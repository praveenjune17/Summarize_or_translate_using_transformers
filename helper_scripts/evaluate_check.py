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
                          evaluate_validation_set, training_results)

val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=2,
                             drop_remainder=True
                             )
count=0
step = 1
for _ in val_dataset:
  count+=1
print(f'Total records count is {count}')
#restore checkpoint
ck_pt_mgr = check_ckpt(config.checkpoint_path)
start_time = time.time()
(task_score, bert_score) = evaluate_validation_set(       
                                                    val_dataset.take(2),
                                                    step
                                                    )  
training_results(
                  step,
                  0,
                  0,
                  task_score, 
                  bert_score,
                  (time.time() - start_time),
                  'none',
                  log,
                  config
                  )
sys.exit()
