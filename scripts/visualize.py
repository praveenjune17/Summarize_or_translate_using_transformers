# -*- coding: utf-8 -*-

import tempfile
import tensorflow as tf
import matplotlib
import tensorflow_datasets as tfds
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('seaborn-deep')
import numpy as np
import pandas as pd
import time
from create_model import source_tokenizer, target_tokenizer
from configuration import config
from preprocess import tf_encode
  
# histogram of tokens per batch_size
# arg1 :- must be a padded_batch dataset
def hist_tokens_per_batch(tf_dataset, batch_size, samples_to_try=1000, split='valid', create_hist=True):
    x=[]
    samples_per_batch = int(samples_to_try)//batch_size
    tf_dataset = tf_dataset.padded_batch(batch_size, padded_shapes=([-1], [-1]))
    tf_dataset = tf_dataset.take(samples_per_batch).cache()
    tf_dataset = tf_dataset.prefetch(buffer_size=samples_to_try)
    for (i, j) in (tf_dataset):
        x.append((tf.size(i) + tf.size(j)).numpy())
    print(f'Descriptive statistics on tokens per batch for {split}')
    print(pd.Series(x).describe())
    if create_hist:
      print(f'creating histogram for {samples_to_try} samples')
      plt.hist(x, bins=20)
      plt.xlabel('Total tokens per batch')
      plt.ylabel('No of times')
      plt.savefig('#_of_tokens per batch in '+split+' set.png')
      plt.close() 

# histogram of Summary_lengths
# arg1 :- must be a padded_batch dataset
def hist_summary_length(tf_dataset, samples_to_try=1000, split='valid', create_hist=True):
    output_sequence=[]
    input_sequence=[]
    tf_dataset = tf_dataset.take(samples_to_try).cache()
    tf_dataset = tf_dataset.prefetch(buffer_size=samples_to_try)
    for (ip_sequence, op_sequence) in (tf_dataset):
      input_sequence.append(ip_sequence.shape[0])
      output_sequence.append(op_sequence.shape[0])  
    combined = [i+j for i,j in zip(input_sequence, output_sequence)]
    print(f'Descriptive statistics on input_sequence length based for {split} set')
    print(pd.Series(input_sequence).describe(percentiles=[0.25, 0.5, 0.8, 0.9, 0.95, 0.97] ))
    print(f'Descriptive statistics on output_sequence length based for {split} set')
    print(pd.Series(output_sequence).describe(percentiles=[0.25, 0.5, 0.8, 0.9, 0.95, 0.97] ))
    print(f'Descriptive statistics for the combined length of input sequences and output sequences based for {split} set')
    print(pd.Series(combined).describe(percentiles=[0.25, 0.5, 0.8, 0.9, 0.95, 0.97] ))
    if create_hist:
      print(f'creating histogram for {samples_to_try} samples')
      plt.hist([output_sequence, input_sequence, combined], alpha=0.5, bins=20, label=['output_sequence', 'input_sequence', 'combined'])
      plt.xlabel('lengths of input_sequence and output_sequence')
      plt.ylabel('Counts')
      plt.legend(loc='upper right')
      plt.savefig(split+'_lengths of input_sequence, output_sequence and combined.png')
      plt.close() 
 
if __name__== '__main__':
  examples, metadata = tfds.load(
                                 config.tfds_name, 
                                 with_info=True,
                                 as_supervised=True, 
                                 data_dir=config.tfds_data_dir,
                                 builder_kwargs=config.tfds_data_version
                                ) 
  splits = examples.keys()
  number_of_samples = 500
  batch_size = 2
  tf_datasets = {}
  for split in splits:
    tf_datasets[split] = examples[split].map(
                               tf_encode(
                                        source_tokenizer,
                                        target_tokenizer, 
                                        config.input_seq_length, 
                                        config.target_seq_length
                                        ), 
                               num_parallel_calls=-1
                               )
    number_of_samples = 1000 if split == 'train' else number_of_samples
    #create histogram for summary_lengths and token
    hist_summary_length(tf_datasets[split], split=split)
    hist_tokens_per_batch(tf_datasets[split], batch_size=batch_size, split=split)

  if config.show_detokenized_samples:
    inp, tar = next(iter(examples['train']))
    print(source_tokenizer.decode([i for i in inp.numpy() if i < source_tokenizer.vocab_size]))
    print(target_tokenizer.decode([i for i in tar.numpy() if i < target_tokenizer.vocab_size]))
