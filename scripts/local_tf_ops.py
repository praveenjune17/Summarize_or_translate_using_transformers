import tensorflow as tf
import time
import os
import shutil
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from preprocess import create_dataset
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log, create_tensorboard_parms, detokenize
from create_model import Model
from model_utils import create_padding_mask, create_masks
from calculate_metrics import (get_loss_and_accuracy, loss_function, 
                               get_optimizer, tf_write_output_sequence)


(train_output_sequence_writer, 
  valid_output_sequence_writer, _) = create_tensorboard_parms()
avg_rouge = tf.keras.metrics.Mean(name='rouge_f1_mean')
avg_bleu = tf.keras.metrics.Mean(name='bleu_mean')
avg_bert_score = tf.keras.metrics.Mean(name='bert_f1_mean')
calculate_combined_metric = tf.keras.metrics.Mean(name='combined_metric_mean', dtype=None)
tf.config.optimizer.set_jit(config.enable_jit)
optimizer = get_optimizer()
# mixed precision doesn't work for transormers models
if not config.model_architecture == 'bertified_transformer':
    policy = mixed_precision.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

train_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                      ]

val_step_signature = [
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.int32),
                      tf.TensorSpec(shape=(None), dtype=tf.bool)
                     ]
  
model_metrics = 'Step {},\n\
                 Train Loss {:.4f},\n\
                 Train_Accuracy {:.4f},\n\
                 ROUGE_f1 {:4f},\n\
                 Bleu {:4f},\n\
                 BERT_f1 {:4f}\n'
evaluation_step  = 'Time taken for {} step : {} secs' 
checkpoint_details = 'Saving checkpoint at step {} on {}'
batch_zero = 'Time taken to feed the input data to the model {} seconds'
batch_run_details = 'Train_Loss {:.4f} Train_Accuracy {:.4f}'
gradient_accumulators = []
train_loss, train_accuracy = get_loss_and_accuracy()

@tf.function(input_signature=train_step_signature)
def train_step(input_ids, 
               target_ids,
               grad_accum_flag):
    
    target_inp = target_ids[:, :-1]
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_ids, target_inp)
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

@tf.function(input_signature=val_step_signature)
def val_step(
             input_ids,
             target_ids,
             step,
             write_output_seq):

    enc_padding_mask = create_padding_mask(input_ids)
    (draft_predictions, _,  
     refine_predictions, _) = Model( 
                                    input_ids,
                                    enc_padding_mask=enc_padding_mask,
                                    target_ids=None,
                                    dec_padding_mask=None, 
                                    look_ahead_mask=None, 
                                    training=None
                                    )
    
    if refine_predictions is not None:
      predictions = refine_predictions
    else:
      predictions = draft_predictions
    rouge_f1, bert_f1, bleu = tf_write_output_sequence(
                                           input_ids,
                                           target_ids[:, 1:], 
                                           predictions[:, 1:], 
                                           step, 
                                           write_output_seq)  
    return (rouge_f1, bert_f1, bleu)

def evaluate_validation_set(
                           validation_dataset, 
                           step
                           ):
    avg_rouge.reset_states()
    avg_bert_score.reset_states()
    avg_bleu.reset_states()
    for (batch, (input_ids, target_ids)) in enumerate(validation_dataset, 1):
        # calculate rouge and bert score for only the first batch
        if batch == 1:
          rouge_f1, bert_f1, bleu = val_step(input_ids,
                                             target_ids,  
                                             step, 
                                             config.write_summary_op
                                             )
        else:
          rouge_f1, bert_f1, bleu  =  val_step(input_ids,
                                               target_ids, 
                                               step, 
                                               False
                                               )
        if rouge_f1:
            avg_rouge.update_state(rouge_f1)
        if bert_f1:
            avg_bert_score.update_state(bert_f1)
        # bleu ranges from 0-100
        if bleu:
            avg_bleu.update_state(bleu/100)
    return (avg_rouge.result().numpy(), 
            avg_bert_score.result().numpy(), 
            avg_bleu.result().numpy()
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
        log.info('No checkpoint found.')
    return (ckpt_manager)

def monitor_run(ckpt_save_path, 
                bert_score, 
                rouge_score, 
                bleu,
                train_loss,
                step,
                copy_best_ckpt=True,
                to_monitor=config.monitor_metric):
  
    
    if config.run_tensorboard:
        with valid_output_sequence_writer.as_default():
            tf.summary.scalar('ROUGE_f1', rouge_score, step=step)
            tf.summary.scalar('BERT_f1', bert_score, step=step)
            tf.summary.scalar('BLEU', bleu, step=step)
    monitor_metrics = dict()
    monitor_metrics['BERT_f1'] = bert_score
    monitor_metrics['ROUGE_f1'] = rouge_score
    monitor_metrics['bleu'] = bleu
    monitor_metrics['combined_metric'] = [
                                          monitor_metrics['BERT_f1'], 
                                          monitor_metrics['ROUGE_f1'],
                                          monitor_metrics['bleu']
                                          ]
    assert config.monitor_metric in monitor_metrics.keys(), f'Available metrics to monitor are {monitor_metrics.keys()}'
    assert sum(config.combined_metric_weights) == 1, 'weights should sum to 1'
    monitor_metrics['combined_metric'] = calculate_combined_metric(monitor_metrics['combined_metric'], 
                                                                   sample_weight=config.combined_metric_weights)
    log.info(f"combined_metric {monitor_metrics['combined_metric'].numpy()}")
    if config.last_recorded_value <= monitor_metrics[to_monitor]:
        if copy_best_ckpt:
            # reset tolerance to zero if the monitor_metric decreases before the tolerance threshold
            ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
            config.tolerance=0
            config.last_recorded_value =  monitor_metrics[to_monitor]
            ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) \
                                 if ckpt_string in files]
            log.info(f'{to_monitor} is {monitor_metrics[to_monitor]:4f} so checkpoint files {ckpt_string} \
                     will be copied to best checkpoint directory')
            # copy the best checkpoints
            shutil.copy2(os.path.join(ckpt_fold, 'checkpoint'), config.best_ckpt_path)
            for files in ckpt_files_tocopy:
                shutil.copy2(os.path.join(ckpt_fold, files), config.best_ckpt_path)
        else:
            pass
    else:
        config.tolerance+=1
    
    # stop if minimum training loss is reached
    if train_loss < config.min_train_loss:
        log.warning(f'Minimum training loss reached')
        config.tolerance+=1
    # Warn and early stop
    if config.tolerance > config.tolerance_threshold:
        log.warning('Tolerance exceeded')
        if config.early_stop:
            log.info(f'Early stopping since the {to_monitor} reached the tolerance threshold')
            return True
        else:
            return False
    else:
        return False


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
    return train_loss.result()



def train_sanity_check(tokenizer, predictions, target_id):
    # use the last sample in the batch
    predicted, target = detokenize(tokenizer, 
                                   tf.squeeze(tf.argmax(predictions,axis=-1)[-1:]), 
                                   tf.squeeze(target_id[:, :-1][-1:])
                                   )
    log.info(f'the true output_sequence is {target}')
    log.info(f'the predicted output_seq with teacher forcing is\
              {predicted if predicted else "empty hence evaluation will be skipped"}')
    return predicted

def training_results(
                    step, 
                    rouge_score, 
                    bert_score,
                    bleu,
                    timing_info,
                    ckpt_save_path
                    ):

      log.info(
                model_metrics.format(
                                    step, 
                                    train_loss.result(), 
                                    train_accuracy.result(), 
                                    rouge_score*100,
                                    bleu*100, 
                                    bert_score*100
                                    )
              )
      log.info(evaluation_step.format(step, timing_info))
      log.info(checkpoint_details.format(step, ckpt_save_path))
      train_loss.reset_states()
      train_accuracy.reset_states()

def save_evaluate_monitor(ck_pt_mgr, val_dataset, target_tokenizer, predictions, train_loss, target_ids, step, start_time):

    ckpt_save_path = ck_pt_mgr.save()
    # print the detokenized training output of a single sample
    predicted = train_sanity_check(target_tokenizer, predictions, target_ids)
    # Run evaluation only if the predictions made by the teacher forced output is not empty
    (rouge_score, bert_score, bleu) = evaluate_validation_set(       
                                                      val_dataset,
                                                      step
                                                      )  if predicted else 0
    training_results(
                      step, 
                      rouge_score, 
                      bert_score,
                      bleu,
                      (time.time() - start_time),
                      ckpt_save_path
                      )
    early_stop = monitor_run(
                              ckpt_save_path, 
                              bert_score, 
                              rouge_score,
                              bleu,
                              train_loss, 
                              step
                              )
    return early_stop