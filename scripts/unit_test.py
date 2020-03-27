# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
import sys
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config
from calculate_metrics import mask_and_one_hot_labels, monitor_run
from creates import log
from create_model import source_tokenizer, target_tokenizer
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set)

train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              batch_size=config.train_batch_size
                              )

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)

def training_loop(dataset):
  for (step, (input_ids, target_ids_)) in tqdm(enumerate(dataset), initial=1):
    max_loss = 10000000
    start=time.time()
    draft_mask, refine_mask, target_ids = mask_and_one_hot_labels(target_ids_)
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
      train_loss = batch_run_check(
                                  step+1,  
                                  start
                                  )
    if config.check_model_capacity:
      if max_loss < train_loss:
        max_loss = train_loss
      else:
        log.warning('Loss not decreasing watch out')

  if config.check_model_capacity:
    if train_loss < config.min_train_loss:
      log.info('Minimum training loss reached')
    else:
      log.info("Loss didn't reach upto the min_train_loss specified, try to increase \
                the parameters of the model and check the run again")

if config.random_results_check:
  training_loop(train_dataset.take(2))
  log.info('First run over. Restart the run time and run the script again')
  sys.exit()

if config.init_loss_check:
  input_ids, target_ids_ = next(iter(train_dataset))
  draft_mask, refine_mask, target_ids = mask_and_one_hot_labels(target_ids_)
  loss =  eval_step(
                    input_ids,  
                    target_ids_, 
                    target_ids, 
                    draft_mask,
                    refine_mask
                    )
  log.info(f"Model's Initial loss {loss}")
  # 2:- Draft and Refine decoders
  log.info(f'Expected initial loss {tf.math.log(tf.cast(config.target_vocab_size, dtype=tf.float32))*2}')
  log.info(f'Initial Loss check run completed')

if config.input_independent_baseline_check:
  input_seq = [[0]*config.input_seq_length]
  output_seq = [[0]*(config.target_seq_length+1)]
  zeroed_train_dataset=tf.data.Dataset.from_tensor_slices((
                                                           tf.convert_to_tensor(input_seq), 
                                                           tf.convert_to_tensor(output_seq)
                                                           )
                                                          )
  zeroed_train_dataset = zeroed_train_dataset.repeat(100)
  for (step, (input_ids, target_ids_)) in tqdm(enumerate(zeroed_train_dataset), initial=1):
    start=time.time()
    input_ids = tf.reshape(input_ids, [1, -1])
    target_ids_ = tf.reshape(target_ids_, [1, -1])
    draft_mask, refine_mask, target_ids = mask_and_one_hot_labels(target_ids_)
    grad_accum_flag = True if (step+1)%config.gradient_accumulation_steps == 0 else False
    refine_predictions = train_step(
                                    input_ids,  
                                    target_ids_, 
                                    target_ids, 
                                    draft_mask,
                                    refine_mask,
                                    grad_accum_flag
                                    )
    batch_run_check(
                  step+1,  
                  start
                  )
  log.info('Input Independent baseline run completed')
  sys.exit()

if config.check_model_capacity:
  training_loop(train_dataset)
  sys.exit()
  