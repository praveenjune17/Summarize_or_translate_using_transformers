# -*- coding: utf-8 -*-
%tensorflow_version 2.x
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.insert(0, '/content/Summarize_and_translate/scripts')

import tensorflow as tf
tf.random.set_seed(100)
tf.config.optimizer.set_jit(True)
import time
from tqdm import tqdm
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from preprocess import create_dataset
from configuration import config
from calculate_metrics import get_loss_and_accuracy, label_smoothing, loss_function, monitor_run, optimizer, tf_write_output_sequence
from creates import log, train_output_sequence_writer, valid_output_sequence_writer
from create_model import source_tokenizer, target_tokenizer, Model
from decode_text import predict_using_sampling
from local_tf_ops import *

# initialize the policy and the optimizer
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
optimizer = mixed_precision.LossScaleOptimizer(optimizer, loss_scale='dynamic')

train_dataset = create_dataset(
                              'train', 
                              source_tokenizer, 
                              target_tokenizer, 
                              True, 
                              False, 
                              90, 
                              92, 
                              287113, 
                              None, 
                              None, 
                              config.train_batch_size
                              )
train_dataset = train_dataset.take(1)
val_dataset = create_dataset(
                            'train', 
                             source_tokenizer, 
                             target_tokenizer, 
                             True, 
                             False, 
                             0, 
                             0.5, 
                             13368, 
                             None, 
                             None, 
                             config.validation_batch_size
                             )
val_dataset = val_dataset.take(1)
train_loss, train_accuracy = get_loss_and_accuracy()
_, validation_accuracy = get_loss_and_accuracy()
gradient_accumulators = []

@tf.function(input_signature=train_step_signature)
def train_step(input_ids, 
               target_ids_, 
               target_ids, 
               draft_mask, 
               refine_mask,
               grad_accum_flag):
  with tf.GradientTape() as tape:
    (draft_predictions, draft_attention_weights, 
      refine_predictions, refine_attention_weights) = Model(
                                                           input_ids,  
                                                           target_ids_,
                                                           True
                                                           )
    train_variables = Model.trainable_variables
    draft_output_sequence_loss = loss_function(target_ids[:, 1:, :], draft_predictions, draft_mask)
    refine_output_sequence_loss = loss_function(target_ids[:, :-1, :], refine_predictions, refine_mask)
    regularization_loss = tf.add_n(Model.losses)
    loss = draft_output_sequence_loss + refine_output_sequence_loss 
    loss = tf.reduce_mean(loss) + regularization_loss
    scaled_loss = optimizer.get_scaled_loss(loss)
  scaled_gradients  = tape.gradient(scaled_loss, train_variables)
  gradients = optimizer.get_unscaled_gradients(scaled_gradients)
  # Initialize the shadow variables with same type as the gradients 
  if not gradient_accumulators:
    for tv in gradients:
      gradient_accumulators.append(tf.Variable(tf.zeros_like(tv), trainable=False))
  # accmulate the gradients to the shadow variables
  for (accumulator, grad) in zip(gradient_accumulators, gradients):
    accumulator.assign_add(grad)
  # apply the gradients and reset them to zero if the flag is true
  if grad_accum_flag:
    optimizer.apply_gradients(zip(gradient_accumulators, train_variables))
    for accumulator in (gradient_accumulators):
        accumulator.assign(tf.zeros_like(accumulator))
    train_loss(loss)
    train_accuracy(target_ids_[:, :-1], refine_predictions)  
  return refine_predictions

def check_ckpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               Model=Model,
                               optimizer=optimizer
                              )
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=10)
    if tf.train.latest_checkpoint(checkpoint_path):
      ckpt.restore(ckpt_manager.latest_checkpoint)
      log.info(ckpt_manager.latest_checkpoint +' restored')
    else:
        log.info('Training from scratch')
    return (ckpt_manager, ckpt)

def val_step(
             input_ids,
             target_ids_,
             step, 
             write_output_seq):
  validation_accuracy.reset_states()
  (predicted_draft_output_sequence, _,  
   refine_predictions, _) = predict_using_sampling( 
                                                    Model,
                                                    input_ids, 
                                                    refine_decoder_sampling_type='greedy', 
                                                    temperature=0.9, 
                                                    p=0.8, 
                                                    k=7
                                                  )
  
  
  validation_accuracy(target_ids_[:, 1:], tf.one_hot(refine_predictions[:, 1:], depth=config.target_vocab_size))  
  rouge, bert = tf_write_output_sequence(target_ids_[:, 1:], refine_predictions[:, 1:], step, write_output_seq)  
  return (rouge, bert)

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr, ckpt = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)


for (step, (input_ids, target_ids_)) in tqdm(enumerate(train_dataset)):
  start=time.time()
  draft_mask = tf.math.logical_not(tf.math.equal(target_ids_[:, 1:], 0))
  refine_mask = tf.math.logical_not(tf.math.equal(target_ids_[:, :-1], 0))
  target_ids = label_smoothing(tf.one_hot(target_ids_, depth=config.target_vocab_size))
  grad_accum_flag = True if (step+1)%config.gradient_accumulation_steps == 0 else False
  refine_predictions = train_step(
                                  input_ids,  
                                  target_ids_, 
                                  target_ids, 
                                  draft_mask,
                                  refine_mask,
                                  grad_accum_flag
                                  )
  if grad_accum_flag:
    batch_run_check(
                  step+1,  
                  start, 
                  train_output_sequence_writer, 
                  train_loss.result(), 
                  train_accuracy.result(), 
                  Model
                  )
  eval_frequency = ((step+1) * config.train_batch_size) % config.eval_after
  if eval_frequency == 0:
    predicted = target_tokenizer.decode([i for i in tf.squeeze(tf.argmax(refine_predictions,axis=-1)) if i not in [config.CLS_ID, config.SEP_ID, config.PAD_ID]])
    target = target_tokenizer.decode([i for i in tf.squeeze(target_ids_[:, :-1]) if i not in [config.CLS_ID, config.SEP_ID, config.PAD_ID]])
    log.info(f'the true output_sequence is {target}')
    log.info(f'the predicted output_sequence with teacher forcing is {predicted if predicted else "EMPTY"}')
    ckpt_save_path = ck_pt_mgr.save()
    (val_acc, rouge_score, bert_score) = evaluate_validation_set(
                                                              val_dataset, 
                                                              step+1, 
                                                              val_step, 
                                                              valid_output_sequence_writer, 
                                                              validation_accuracy
                                                              )
    log.info(
              model_metrics.format(
                                  step+1, 
                                  train_loss.result(), 
                                  train_accuracy.result(), 
                                  val_acc,
                                  rouge_score, 
                                  bert_score
                                  )
            )
    log.info(evaluation_step.format(step+1, time.time() - start))
    log.info(checkpoint_details.format(step+1, ckpt_save_path))
    if not monitor_run(
                        ckpt_save_path, 
                        val_acc, 
                        bert_score, 
                        rouge_score, 
                        valid_output_sequence_writer, 
                        step+1):
      break