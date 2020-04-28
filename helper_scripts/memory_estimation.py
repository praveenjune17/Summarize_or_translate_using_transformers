# -*- coding: utf-8 -*-

### Not verified yet#######################
from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.insert(0, '/content/Summarize_and_translate/scripts')
import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
import sys
import numpy as np
from io import StringIO
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config
from calculate_metrics import mask_and_calculate_loss, monitor_run
from utilities import log, detokenize
from create_model import source_tokenizer, target_tokenizer, Transformer, Bertified_transformer
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set)

temp_input = tf.random.uniform((64, 100), dtype=tf.int64, minval=0, maxval=200)
temp_target = tf.random.uniform((64, 100), dtype=tf.int64, minval=0, maxval=200)
floating_point_precision = 4
enc_block_op = 0
dec_block_op = 0
batch_size = config.train_batch_size
inp_seq_len = (config.input_seq_length)
tar_seq_len = (config.target_seq_length)
num_heads = (config.num_heads)
d_model = (config.d_model)
dff = (config.dff)
target_vocab_size = float(config.target_vocab_size)
input_vocab_size = float(config.input_vocab_size)
pointer_gen = config.add_pointer_generator
input_shape = np.prod((batch_size, inp_seq_len, input_vocab_size))
encoder_embedding = np.prod((batch_size, inp_seq_len, d_model))
decoder_embedding = np.prod((batch_size, tar_seq_len, d_model))
extra_parms = np.prod((batch_size, tar_seq_len, 1)) if pointer_gen else 0


if config.model_architecture == 'bertified_transformer':
    sample_model = Bertified_transformer(
                            num_layers=config.num_layers, 
                            d_model=config.d_model, 
                            num_heads=config.num_heads, 
                            dff=config.dff, 
                            input_vocab_size=config.input_vocab_size,
                            target_vocab_size=config.target_vocab_size
                            )
else:
    sample_model = Transformer(num_layers=config.num_layers, 
                            d_model=config.d_model, 
                            num_heads=config.num_heads, 
                            dff=config.dff, 
                            input_vocab_size=config.input_vocab_size, 
                            target_vocab_size=config.target_vocab_size,
                            add_pointer_generator=config.add_pointer_generator)
(draft_predictions, draft_attention_weights, 
refine_predictions, refine_attention_weights) = sample_model(temp_input,
                                                    dec_padding_mask=None, 
                                                    enc_padding_mask=None, 
                                                    look_ahead_mask=None,
                                                    target_ids=temp_target, 
                                                    training=False, 
                                                    )
log.info(f'The output shape of the sample model is {tf.shape(draft_predictions if refine_predictions is None else refine_predictions)}')

for i in range(config.num_layers):
  enc_att_1 = np.prod((batch_size, num_heads, inp_seq_len, inp_seq_len))
  enc_ffd1 = np.prod((batch_size, inp_seq_len, dff))
  enc_ffd2 = np.prod((batch_size, inp_seq_len, d_model))
  enc_block_op += enc_att_1+enc_ffd1+enc_ffd2
  dec_att_1 = np.prod((batch_size, num_heads, tar_seq_len, tar_seq_len))
  dec_att_2 = np.prod((batch_size, num_heads, tar_seq_len, inp_seq_len))
  dec_ffd1 = np.prod((batch_size, tar_seq_len, dff))
  dec_ffd2 = np.prod((batch_size, tar_seq_len, d_model))
  dense = np.prod((batch_size, tar_seq_len, target_vocab_size))
  dec_block_op += dec_att_1 + dec_att_2 + dec_ffd1 + dec_ffd2 + dense
total_shape_parms = input_shape + enc_block_op + dec_block_op + extra_parms+ encoder_embedding + decoder_embedding
total_memory = float(floating_point_precision*(sample_model.count_params()+total_shape_parms))
gbytes = np.round(total_memory / (1024.0 ** 3), 3) #+ internal_model_mem_count
gbytes*2 #consider forward and backward