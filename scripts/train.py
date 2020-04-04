# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config
from calculate_metrics import mask_and_one_hot_labels, monitor_run
from creates import log
from create_model import source_tokenizer, target_tokenizer
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set, post_training_results)

train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              batch_size=config.train_batch_size
                              )
val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size                          
                             )

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)
for (step, (input_ids, target_ids_)) in tqdm(enumerate(train_dataset), initial=1):
  start=time.time()
  draft_mask, refine_mask, target_ids = mask_and_one_hot_labels(target_ids_)
  grad_accum_flag = True if (step%config.gradient_accumulation_steps) == 0 else False
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
                                step,  
                                start
                                )
  evaluate = ((step) * config.train_batch_size) % config.eval_after
  if evaluate == 0:
    train_sanity_check(target_tokenizer, refine_predictions, target_ids_)
    ckpt_save_path = ck_pt_mgr.save()
    (val_acc, rouge_score, bert_score) = evaluate_validation_set(
                                                                  val_dataset, 
                                                                  step
                                                                  )
    post_training_results(
                          step, 
                          val_acc,
                          rouge_score, 
                          bert_score,
                          (time.time() - start),
                          ckpt_save_path
                          )
    monitor_early_stop = monitor_run(
                                    ckpt_save_path, 
                                    val_acc, 
                                    bert_score, 
                                    rouge_score,
                                    train_loss, 
                                    step
                                    )
    if not monitor_early_stop:
      break
