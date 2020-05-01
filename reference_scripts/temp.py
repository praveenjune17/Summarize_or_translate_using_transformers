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
from utilities import log
from create_model import source_tokenizer, target_tokenizer
from model_training_helper import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set, training_results)


val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size,
                             drop_remainder=True                          
                             )

count=0
for _ in val_dataset:
    count+=1

print(f'count {count}')
# # if a checkpoint exists, restore the latest checkpoint.
# (rouge_score, bert_score) = evaluate_validation_set(
#                                                       val_dataset, 
#                                                       step+1
#                                                       )
