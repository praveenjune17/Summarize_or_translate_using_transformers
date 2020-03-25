# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
tf.config.optimizer.set_jit(True)
import time
import argparse
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config
from calculate_metrics import mask_and_smooth_labels, monitor_run
from creates import log, valid_output_sequence_writer
from create_model import source_tokenizer, target_tokenizer, Model
from local_tf_ops import (check_ckpt, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set)

parser = argparse.ArgumentParser('Set Low Config')
parser.add_argument("--test_script", help="An integer will be increased by 1 and printed.", type=bool, default=True)
parser.add_argument("--no_of_samples_to_test", help="# of samples to test", type=int, default=1)
parser.add_argument("--turnoff_regularizers", help="turns off regularizers.", type=bool, default=True)
args = parser.parse_args()


train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              batch_size=config.train_batch_size
                              )
val_dataset = create_dataset(
                             split='train', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size                          
                             )

# Unit test cases
if args.test_script:
  no_of_samples_to_test = args.no_of_samples_to_test
  train_dataset = train_dataset.take(no_of_samples_to_test)
  val_dataset = val_dataset.take(no_of_samples_to_test)
  config.gradient_accumulation_steps =  config.train_batch_size = no_of_samples_to_test
  config.epochs = 100000
  config.dff = 512                      # feed forward network hidden parameters
  config.num_heads = 4                  # the number of heads in the multi-headed attention unit
  config.num_layers = 2                 # number of transformer blocks
  assert config.d_model % config.num_heads == 0, 'd_model should be a multiple of num_heads'
  if args.turnoff_regularizers:
    config.dropout_rate = config.epsilon_ls = 0.0
    config.grad_clipnorm = None
    config.l2_norm = 0.0

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)

for (step, (input_ids, target_ids_)) in tqdm(enumerate(train_dataset)):
  start=time.time()
  draft_mask, refine_mask, target_ids = mask_and_smooth_labels(target_ids_)
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
                  Model
                  )
  evaluate = ((step+1) * config.train_batch_size) % config.eval_after
  if evaluate == 0:
    train_sanity_check(target_tokenizer, refine_predictions, target_ids_)
    ckpt_save_path = ck_pt_mgr.save()
    (val_acc, rouge_score, bert_score) = evaluate_validation_set(
                                                                  val_dataset, 
                                                                  step+1
                                                                  )
    post_training_results(
                          step+1, 
                          val_acc,
                          rouge_score, 
                          bert_score,
                          (time.time() - start),
                          ckpt_save_path
                          )
    monitor_early_stop=monitor_run(
                                  ckpt_save_path, 
                                  val_acc, 
                                  bert_score, 
                                  rouge_score, 
                                  step+1
                                  )
    if not monitor_early_stop:
      break
