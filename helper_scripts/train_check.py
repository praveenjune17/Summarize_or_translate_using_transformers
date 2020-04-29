# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.insert(0, 'D:\\Local_run\\Summarize_and_translate\\scripts')
sys.path.insert(0, 'D:\\Local_run\\models')
import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config, source_tokenizer, target_tokenizer
from calculate_metrics import mask_and_calculate_loss
from utilities import log
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check,
                          train_sanity_check)

train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=0, 
                              to=100, 
                              batch_size=1,
                              shuffle=False
                              )

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)

for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=1):

    start_time = time.time()
    grad_accum_flag = (True if (step%config.gradient_accumulation_steps) == 0 else False) if config.accumulate_gradients else None
    # print('input_ids')
    # print(input_ids)
    # print()
    # print('target_ids')
    # print(target_ids)
    # print()
    # print('grad_accum_flag')
    # print(grad_accum_flag)
    # print()
    predictions = train_step(
                            input_ids,  
                            target_ids,
                            grad_accum_flag
                            )
    if (step % config.steps_to_print_training_info) == 0:
        train_loss = batch_run_check(
                                  step,  
                                  start_time
                                  )
        train_sanity_check(target_tokenizer, predictions, target_ids)
    if (step % config.eval_after_steps) == 0:
        early_stop = True
        if early_stop:
            break

train_sanity_check(target_tokenizer, predictions, target_ids)
log.info(f'Training completed at step {step}')
