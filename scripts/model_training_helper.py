import tensorflow as tf
import time
import os
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from preprocess import create_dataset
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log, create_tensorboard_parms
from create_model import Model
from model_utils import create_padding_mask, create_masks
from training_house_keeping import monitor_eval_metrics, training_results, train_sanity_check
from calculate_metrics import (get_loss_and_accuracy, loss_function, 
                               get_optimizer, tf_write_output_sequence)

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
tf.config.optimizer.set_jit(config.enable_jit)

(train_output_sequence_writer, 
  _, _) = create_tensorboard_parms()
train_loss, train_accuracy = get_loss_and_accuracy()
optimizer = get_optimizer()
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')
avg_task_score = tf.keras.metrics.Mean(name='avg_task_score')
avg_bert_score = tf.keras.metrics.Mean(name='bert_f1_mean')
batch_zero = 'Time taken to feed the input data to the model {} seconds'
batch_run_details = 'Train_Loss {:.4f} Train_Accuracy {:.4f}'
gradient_accumulators = []

train_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                      ]

val_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.string),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                     ]

@tf.function(input_signature=train_step_signature)
def train_step(input_ids, 
               target_ids,
               grad_accum_flag):
    
    target_inp = target_ids[:, :-1]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
                                                        input_ids, 
                                                        target_inp
                                                        )
    with tf.GradientTape() as tape:
        (draft_predictions, draft_attention_weights, 
          refine_predictions, refine_attention_weights) = Model(
                                                         input_ids,
                                                         dec_padding_mask=dec_padding_mask,
                                                         target_ids=target_inp,
                                                         enc_padding_mask=enc_padding_mask, 
                                                         look_ahead_mask=combined_mask, 
                                                         training=True,
                                                         )
        train_variables = Model.trainable_variables
        loss, target = loss_function(target_ids, 
                                     draft_predictions,
                                     refine_predictions, 
                                     Model
                                     )
        predictions = refine_predictions if refine_predictions is not None else draft_predictions
        scaled_loss = optimizer.get_scaled_loss(loss)
    scaled_gradients  = tape.gradient(scaled_loss, train_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    if config.accumulate_gradients:
        # Initialize the shadow variables with same type as the gradients 
        if not gradient_accumulators:
            for tv in gradients:
              gradient_accumulators.append(tf.Variable(tf.zeros_like(tv), 
                                                       trainable=False))
        # accmulate the gradients to the shadow variables
        for (accumulator, grad) in zip(gradient_accumulators, gradients):
            accumulator.assign_add(grad)
        # apply the gradients and reset them to zero if the flag is true
        if grad_accum_flag:
            optimizer.apply_gradients(zip(gradient_accumulators, train_variables))
            for accumulator in (gradient_accumulators):
                accumulator.assign(tf.zeros_like(accumulator))
            train_loss(loss)
            train_accuracy(target, predictions)
    else:
        optimizer.apply_gradients(zip(gradients, train_variables))
        train_loss(loss)
        train_accuracy(target, predictions)

    return predictions

@tf.function(input_signature=val_step_signature)#experimental_relax_shapes=True)
def val_step(
             input_ids,
             target_ids,
             step,
             write_output_seq):

    enc_padding_mask = create_padding_mask(input_ids)
    (draft_predictions, _,  
     refine_predictions, _) = Model( 
                                   input_ids,
                                   decoder_type=config.draft_decoder_type,
                                   beam_size=config.beam_size,
                                   length_penalty=config.length_penalty,
                                   temperature=config.softmax_temperature, 
                                   top_p=config.top_p,
                                   top_k=config.top_k,
                                   enc_padding_mask=enc_padding_mask,
                                   target_ids=None,
                                   dec_padding_mask=None, 
                                   look_ahead_mask=None, 
                                   training=None)
    
    if refine_predictions is not None:
      predictions = refine_predictions
    else:
      predictions = draft_predictions
    task_score, bert_f1 = tf_write_output_sequence(
                                     input_ids,
                                     target_ids[:, 1:], 
                                     predictions[:, 1:], 
                                     step, 
                                     write_output_seq
                                      )

    return (task_score, bert_f1)

def evaluate_validation_set(
                           validation_dataset, 
                           step,
                           decoder_type=config.draft_decoder_type,
                           beam_size=config.beam_size,
                           length_penalty=config.length_penalty,
                           temperature=config.softmax_temperature, 
                           top_p=config.top_p,
                           top_k=config.top_k
                           ):

    avg_task_score.reset_states()
    avg_bert_score.reset_states()
    step='_ '.join([str(i) for i in (decoder_type,
                                     beam_size,
                                     length_penalty,
                                     temperature, 
                                     top_p,
                                     top_k)])
    for (batch, (input_ids, target_ids)) in enumerate(validation_dataset, 1):
        # calculate rouge and bert score for only the first batch
        if batch == 1:
          task_score, bert_f1 = val_step(input_ids,
                                         target_ids,  
                                         step, 
                                         config.write_batch1_predictions
                                         )
        else:
          task_score, bert_f1 =  val_step(input_ids,
                                       target_ids, 
                                       step, 
                                       False
                                       )
        # bleu ranges from 0-100
        if task_score:
            avg_task_score.update_state(task_score if config.task=='summarize' else task_score/100)
        if bert_f1:
            avg_bert_score.update_state(bert_f1)

    return (avg_task_score.result().numpy(), 
            avg_bert_score.result().numpy()
            )

def eval_step(input_ids, 
               target_ids, 
               ):

    target_inp = target_ids[:, :-1]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_ids, target_inp)  
    (draft_predictions, draft_attention_weights, 
      refine_predictions, refine_attention_weights) = Model(
                                                             input_ids,
                                                             dec_padding_mask=dec_padding_mask,
                                                             target_ids=target_inp,
                                                             enc_padding_mask=enc_padding_mask, 
                                                             look_ahead_mask=combined_mask, 
                                                             training=False
                                                             )
    loss, target = loss_function(target_ids, 
                         draft_predictions,
                         refine_predictions, 
                         Model
                         )
    predictions = refine_predictions if refine_predictions is not None else draft_predictions
    train_accuracy(target, predictions)
    train_loss(loss)
    log.info(Model.summary())
    if config.save_initial_weights:
        initial_weights = os.path.join(config.initial_weights,'initial_weights')
        Model.save_weights(initial_weights)

    return loss
    
def check_ckpt(checkpoint_path):

    ckpt = tf.train.Checkpoint(
                               Model=Model,
                               optimizer=optimizer
                              )
    ckpt_manager = tf.train.CheckpointManager(ckpt, 
                                              checkpoint_path, 
                                              max_to_keep=10)
    if tf.train.latest_checkpoint(checkpoint_path):
        ckpt.restore(ckpt_manager.latest_checkpoint)
        log.info(ckpt_manager.latest_checkpoint +' restored')
    else:
        log.warning('No checkpoint found so using the initialized_weights')

    return ckpt_manager
# run every batch
def batch_run_check(batch, start_time):

    if config.run_tensorboard:
        with train_output_sequence_writer.as_default():
          tf.summary.scalar('train_loss', train_loss.result(), step=batch)
          tf.summary.scalar('train_accuracy', train_accuracy.result(), step=batch)
    if config.display_model_summary:
        log.info(Model.summary())
        log.info(batch_zero.format(time.time()-start_time))
        config.display_model_summary = False
    log.info(
             batch_run_details.format(
                                     train_loss.result(), 
                                     train_accuracy.result()
                                     )
            )

    #return train_loss.result()

def save_evaluate_monitor(ck_pt_mgr, val_dataset, 
            target_tokenizer, predictions, 
            target_ids, step, start_time):

    ckpt_save_path = ck_pt_mgr.save()
    # print the detokenized training output of a single sample
    predicted = train_sanity_check(target_tokenizer, predictions, target_ids, log)
    evaluate = train_loss.result() < config.start_evaluate_when and True if predicted  else False
    # Run evaluation only if the predictions made by the teacher forced output is not empty
      # and the train_loss is lesser than start_evaluate_when
    if evaluate:
        (task_score, bert_score) = evaluate_validation_set(       
                                                      val_dataset,
                                                      step
                                                      )
        early_stop_training = monitor_eval_metrics(
                              ckpt_save_path, 
                              bert_score, 
                              task_score,
                              train_loss.result(), 
                              step,
                              log,
                              config
                              )
    else:
        (task_score, bert_score) = (0, 0)
        early_stop_training = False

    training_results(
                      step,
                      train_loss.result(), 
                      train_accuracy.result(), 
                      task_score, 
                      bert_score,
                      (time.time() - start_time),
                      ckpt_save_path,
                      log,
                      config
                      )
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    return early_stop_training
