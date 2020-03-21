# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
from functools import partial
from configuration import config
from create_model import tokenizer
from creates import log

AUTOTUNE = tf.data.experimental.AUTOTUNE

def pad(l, n, pad=config.PAD_ID):
    """
    Pad the list 'l' to have size 'n' using 'padding_element'
    """
    pad_with = (0, max(0, n - len(l)))
    return np.pad(l, pad_with, mode='constant', constant_values=pad)


def encode(sent_1, sent_2, tokenizer, input_seq_len, output_seq_len):
    
    input_ids = tokenizer.encode(sent_1.numpy().decode('utf-8'))
    target_ids = tokenizer.encode(sent_2.numpy().decode('utf-8'))
    # Account for [CLS] and [SEP] with "- 2"
    if len(input_ids) > input_seq_len - 2:
        input_ids = input_ids[0:(input_seq_len - 2)]
    if len(target_ids) > (output_seq_len + 1) - 2:
        target_ids = target_ids[0:((output_seq_len + 1) - 2)]
    input_ids = pad(input_ids, input_seq_len)
    target_ids = pad(target_ids, output_seq_len + 1)    
    return input_ids, target_ids


def tf_encode(tokenizer, input_seq_len, output_seq_len):
    """
    Operations inside `.map()` run in graph mode and receive a graph
    tensor that do not have a `numpy` attribute.
    The tokenizer expects a string or Unicode symbol to encode it into integers.
    Hence, you need to run the encoding inside a `tf.py_function`,
    which receives an eager tensor having a numpy attribute that contains the string value.
    """    
    def f(s1, s2):
        encode_ = partial(encode, tokenizer=tokenizer, input_seq_len=input_seq_len, output_seq_len=output_seq_len)
        return tf.py_function(encode_, [s1, s2], [tf.int32, tf.int32])
    return f

# Set threshold for input_sequence and  output_sequence length
def filter_max_length(x, y):
    return tf.logical_and(
                          tf.size(x[0]) <= config.input_seq_length,
                          tf.size(y[0]) <= config.target_seq_length
                         )

def filter_combined_length(x, y):
    return tf.math.less_equal(
                              (tf.math.count_nonzero(x) + tf.math.count_nonzero(y)), 
                              config.max_tokens_per_line
                             )
                        
# this function should be added after padded batch step
def filter_batch_token_size(x, y):
    return tf.math.less_equal(
                              (tf.size(x[0]) + tf.size(y[0])), 
                              config.max_tokens_per_line*config.train_batch_size
                             )
    
def read_csv(path, num_examples):
    df = pd.read_csv(path)
    df.columns = [i.capitalize() for i in df.columns if i.lower() in ['input_sequence', 'output_sequence']]
    assert len(df.columns) == 2, 'column names should be input_sequence and output_sequence'
    df = df[:num_examples]
    assert not df.isnull().any().any(), 'dataset contains  nans'
    return (df["input_sequence"].values, df["output_sequence"].values)

def create_dataset(split, use_tfds, shuffle, from_, to, buffer_size, csv_path, num_examples_to_select, batch_size):

  if use_tfds:
      raw_dataset, _ = tfds.load(
                                 config.tfds_name, 
                                 with_info=True,
                                 as_supervised=True, 
                                 data_dir=config.tfds_data_dir,
                                 builder_kwargs={"version": "2.0.0"},
                                 split=tfds.core.ReadInstruction(split, from_=from_, to=to, unit='%')
                                )        
  else:
    input_seq, output_seq = read_csv(csv_path, num_examples_to_select)
    raw_dataset = tf.data.Dataset.from_tensor_slices((input_seq, output_seq))
    buffer_size = len(input_seq)
  tf_dataset = raw_dataset.map(
                               tf_encode(
                                        tokenizer, 
                                        config.input_seq_length, 
                                        config.target_seq_length
                                        ), 
                               num_parallel_calls=AUTOTUNE
                               )
  tf_dataset = tf_dataset.filter(filter_combined_length)
  tf_dataset = tf_dataset.cache()
  if shuffle:
     tf_dataset = tf_dataset.shuffle(buffer_size, seed = 100)
  tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]), padding_values=tf.convert_to_tensor(config.PAD_ID))
  tf_dataset = tf_dataset.prefetch(buffer_size=AUTOTUNE)
  log.info(f'{split} tf_dataset created')
  return tf_dataset
