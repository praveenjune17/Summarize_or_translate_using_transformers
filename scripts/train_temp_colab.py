# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.insert(0, '/content/models')
import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config, source_tokenizer, target_tokenizer
from calculate_metrics import mask_and_calculate_loss
from utilities import log
from model_training_helper import (check_ckpt, eval_step, train_step, batch_run_check, 
                          save_evaluate_monitor)

train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=0, 
                              to=100, 
                              batch_size=config.train_batch_size,
                              shuffle=True
                              )
val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size,
                             shuffle=True,
                             drop_remainder=True,
                             num_examples_to_select=config.samples_to_validate
                             )

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)

for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=1):
    if step > (217000//config.train_batch_size):
        start_time = time.time()
        grad_accum_flag = (True if (step%config.gradient_accumulation_steps) == 0 else False) if config.accumulate_gradients else None
        predictions = train_step(
                                input_ids,  
                                target_ids,
                                grad_accum_flag
                                )
        if (step % config.steps_to_print_training_info) == 0:
            batch_run_check(
                            step,  
                            start_time
                            )
        if (step % config.eval_after_steps) == 0:
            early_stop = save_evaluate_monitor(ck_pt_mgr, val_dataset, 
                          target_tokenizer, predictions, target_ids, step, start_time)
            if early_stop:
                break
    else:
        continue

if not early_stop:
    _ = save_evaluate_monitor(ck_pt_mgr, val_dataset, 
            target_tokenizer, predictions, target_ids, step, start_time)
log.info(f'Training completed at step {step}')
