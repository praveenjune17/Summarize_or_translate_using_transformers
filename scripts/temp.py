# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
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
from model_training_helper import (check_ckpt, eval_step, train_step, batch_run_check, 
                          save_evaluate_monitor)


train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=0, 
                              to=100, 
                              batch_size=config.train_batch_size,
                              shuffle=False
                              )
# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)

for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=1):
    print(input_ids.numpy().decode('utf-8'))
    print(target_ids.numpy().decode('utf-8'))
    break
# for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=1):
#     for input_id, target_id, in zip(input_ids, target_ids):
#         print(f'source_tokenizer {source_tokenizer.decode(tf.squeeze(input_id), skip_special_tokens=True)}')
#         print(f'target_tokenizer {target_tokenizer.decode(tf.squeeze(target_id), skip_special_tokens=True)}')
#         #if step==100:
#         break
#     break

#எழுத இன்னும் விடியோக்கள் இல்லை    
 எழுத இன்னும் விடியோக்கள் இல்லை