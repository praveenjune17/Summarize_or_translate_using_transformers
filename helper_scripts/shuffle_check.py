# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.insert(0, 'D:\\Local_run\\Summarize_or_translate_using_transformers\\scripts')
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
                          train_sanity_check)

train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=0, 
                              to=100, 
                              batch_size=2,
                              shuffle=False
                              )

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)

for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=1):

    print(inputs_ids)
    if step==100:
        break