# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config
from calculate_metrics import mask_and_calculate_loss, monitor_run
from creates import log, detokenize
from create_model import source_tokenizer, target_tokenizer
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set, training_results)


# input_ids, tar = next(iter())
# source, target = detokenize(target_tokenizer, 
#                             tf.squeeze(input_ids[-1,:]), 
#                             tf.squeeze(tar[-1,:]), 
#                             source_tokenizer
#                             )
#print(f'input sequence:- {source}')
#print(f'target sequence:- {target}')

val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=1,
                             drop_remainder=True
                             )

for (step, (input_ids, target)) in tqdm(enumerate(val_dataset, 1), initial=1):
    source, target = detokenize(target_tokenizer, 
                            tf.squeeze(input_ids), 
                            tf.squeeze(target), 
                            source_tokenizer
                            )
    print(f'input sequence:- {source}')
    print(f'target sequence:- {target}')
    