# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from functools import partial
from collections import defaultdict
from configuration import config
from utilities import log

AUTOTUNE = tf.data.experimental.AUTOTUNE

def pad(l, n, pad=config.PAD_ID):
    """
    Pad the list 'l' to have size 'n' using 'padding_element'
    """
    pad_with = (0, max(0, n - len(l)))

    return np.pad(l, pad_with, mode='constant', constant_values=pad)

def encode(sent_1, sent_2, source_tokenizer, target_tokenizer, input_seq_len, output_seq_len):
    
    if config.model == 'bertified_transformer':
        input_ids = source_tokenizer.encode(sent_1.numpy().decode('utf-8'), add_special_tokens=True) 
        target_ids = target_tokenizer.encode(sent_2.numpy().decode('utf-8'), add_special_tokens=True) 
        # Account for [CLS] and [SEP] with "- 2"
        if len(input_ids) > input_seq_len - 2:
            input_ids = input_ids[0:(input_seq_len - 2)]
        if len(target_ids) > (output_seq_len + 1) - 2:
            target_ids = target_ids[0:((output_seq_len + 1) - 2)]
        input_ids = pad(input_ids, input_seq_len)
        target_ids = pad(target_ids, output_seq_len + 1)
    else:    
      input_ids = [config.input_CLS_ID] + source_tokenizer.encode(sent_1.numpy()) + [config.input_SEP_ID]
      target_ids = [config.target_CLS_ID] + target_tokenizer.encode(sent_2.numpy()) + [config.target_SEP_ID]

    return input_ids, target_ids

def tf_encode(source_tokenizer, target_tokenizer, input_seq_len, output_seq_len):
    """
    Operations inside `.map()` run in graph mode and receive a graph
    tensor that do not have a `numpy` attribute.
    The tokenizer expects a string or Unicode symbol to encode it into integers.
    Hence, you need to run the encoding inside a `tf.py_function`,
    which receives an eager tensor having a numpy attribute that contains the string value.
    """    
    def f(s1, s2):
        encode_ = partial(
                          encode, 
                          source_tokenizer=source_tokenizer, 
                          target_tokenizer=target_tokenizer, 
                          input_seq_len=input_seq_len, 
                          output_seq_len=output_seq_len
                          )
        return tf.py_function(encode_, [s1, s2], [tf.int32, tf.int32])
    return f

# Set threshold for input_sequence and  output_sequence length
def filter_max_length(x, y):
    return tf.logical_and(
                          tf.math.count_nonzero(x) <= config.input_seq_length,
                          tf.math.count_nonzero(y) <= config.target_seq_length
                         )

def filter_special_max_length(x, y):
    return tf.logical_and(
                          tf.math.count_nonzero(x) <= 2000,
                          tf.math.count_nonzero(y) <= 2000
                         )

# Set threshold for input_sequence and  output_sequence length
def filter_min_length(x, y):
    return tf.logical_and(
                          tf.math.count_nonzero(x) >= 650,
                          tf.math.count_nonzero(y) >= 650
                         )

def filter_tokens_per_batch(x, y):
    return tf.math.less_equal(
                              (tf.size(x) + tf.size(y)), 
                              config.tokens_per_batch
                             )
          
def create_dataset(split, 
                   source_tokenizer, 
                   target_tokenizer, 
                   from_, 
                   to, 
                   batch_size,
                   shuffle=None,
                   drop_remainder=False,
                   num_examples_to_select=config.samples_to_train):

    if config.tfds_name == 'en_tam_parallel_text':
        en_tam_ds = defaultdict(list)
        record_count=0
        #List of available datasets in the package
        dataset_names = ['GNOME_v1_en_to_ta', 'GNOME_v1_en_AU_to_ta', 'GNOME_v1_en_CA_to_ta', 
                 'GNOME_v1_en_GB_to_ta', 'GNOME_v1_en_US_to_ta', 'KDE4_v2_en_to_ta', 
                 'KDE4_v2_en_GB_to_ta', 'Tatoeba_v20190709_en_to_ta', 'Ubuntu_v14.10_en_to_ta_LK', 
                 'Ubuntu_v14.10_en_GB_to_ta_LK', 'Ubuntu_v14.10_en_AU_to_ta_LK', 'Ubuntu_v14.10_en_CA_to_ta_LK', 
                 'Ubuntu_v14.10_en_US_to_ta_LK', 'Ubuntu_v14.10_en_to_ta', 'Ubuntu_v14.10_en_GB_to_ta', 
                 'Ubuntu_v14.10_en_AU_to_ta', 'Ubuntu_v14.10_en_CA_to_ta', 'Ubuntu_v14.10_en_NZ_to_ta', 
                 'Ubuntu_v14.10_en_US_to_ta', 'OpenSubtitles_v2018_en_to_ta', 'OpenSubtitles_v2016_en_to_ta',
                 'en_ta', 'github_joshua_en_ta']
        for name in dataset_names:
            en_tam_ds[(name,'metadata_'+name)] = tfds.load(f'{config.tfds_name}/'+name, 
                                                          with_info=True, 
                                                          as_supervised=True,
                                                          data_dir=config.tfds_data_dir,
                                                          builder_kwargs={'version': config.tfds_data_version},
                                                        )
        for i,j  in en_tam_ds.keys():
            record_count+=(sum([i.num_examples for i in  list(en_tam_ds[(i, j)][1].splits.values())]))
        if not config.test_script:
            log.info(f'Total record count without filtering is {record_count}')
        #initialize the first dataset to the train_examples variable
        #Concatenate all the train datasets
        if split == 'train':
            raw_dataset = en_tam_ds[('GNOME_v1_en_to_ta', 'metadata_GNOME_v1_en_to_ta')][0]['train']
            for typ in list(en_tam_ds.keys())[1:]:
                raw_dataset = raw_dataset.concatenate(en_tam_ds[typ][0]['train'])
          #other splits includes validation and test 
        else:
            raw_dataset = en_tam_ds[('en_ta', 'metadata_en_ta')][0][split]
    else:
        raw_dataset, ds_info = tfds.load(
                             config.tfds_name, 
                             with_info=True,
                             as_supervised=True, 
                             data_dir=config.tfds_data_dir,
                             builder_kwargs={'version': config.tfds_data_version},
                             split=tfds.core.ReadInstruction(split, from_=from_, to=to, unit='%')
                            )
        record_count = (sum([i.num_examples for i in  list(ds_info.splits.values())]))
    tf_dataset = raw_dataset.map(
                                 tf_encode(
                                          source_tokenizer,
                                          target_tokenizer, 
                                          config.input_seq_length, 
                                          config.target_seq_length
                                          ), 
                                 num_parallel_calls=AUTOTUNE
                                 )
    tf_dataset = tf_dataset.filter(filter_max_length)
    tf_dataset = tf_dataset.take(num_examples_to_select) 
    tf_dataset = tf_dataset.cache()
    if shuffle:
        tf_dataset = tf_dataset.shuffle(record_count, seed=100)
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]), drop_remainder=drop_remainder)
    tf_dataset = tf_dataset.prefetch(buffer_size=AUTOTUNE)
    log.info(f'{split} tf_dataset created')

    return tf_dataset
