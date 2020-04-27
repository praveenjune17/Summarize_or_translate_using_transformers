#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import sys
sys.path.insert(0, 'D:\\Local_run\\Summarize_and_translate\\scripts')
sys.path.insert(0, 'D:\\Local_run\\models')
import io
import os
import numpy as np
import tensorflow as tf
import string
import time
from tensorboard.plugins import projector
from configuration import config
from create_model import source_tokenizer, target_tokenizer, Model
from local_tf_ops import check_ckpt
from creates import embedding_projector_dir

#table = str.maketrans(dict.fromkeys(string.punctuation))  
def tokenize_and_aggregate(tokens, tokenizer, agg, embedding_layer):
    # remove punctuation
    #tokens = tokens.translate(table)
    target_ids = tokenizer.encode(tokens)
    # create sentence embedding by aggregating the tokens
    if agg=='sum':
        embedding_vector = np.sum(embedding_layer[target_ids,:], axis=0)
    elif agg=='mean':
        embedding_vector = np.mean(embedding_layer[target_ids,:], axis=0)

    return embedding_vector

def save_checkpoint_create_config(checkpoint, config, embedding_type, log_dir):
    
    checkpoint.save(os.path.join(log_dir, embedding_type+"_embedding.ckpt"))
    embedding_config = config.embeddings.add()
    embedding_config.tensor_name = embedding_type+"_embedding/.ATTRIBUTES/VARIABLE_VALUE"
    embedding_config.metadata_path = 'metadata_'+embedding_type+'.tsv'

    return config

def display_embedding_shape(sentence_embedding, sentences, embedding_type):
    
    sentence_embedding = np.asarray(sentence_embedding)
    rows, cols = sentence_embedding.shape
    print(f'Shape of the {embedding_type}_embedding tensor created is {rows}, {cols} and the number of sequences are {len(sentences)}')

    return sentence_embedding   

def embedding_projector_files(source_tokenizer, target_tokenizer, model, sentence_pair, log_dir, agg='mean'):
    #words = []
    source_sentence_vector  = []
    target_sentence_vector = []
    souce_sentences = []
    target_sentences = []
    # Remove start and end token embedding
    target_embedding_layer = model.layers[1].get_weights()[0][1:-1, :]  
    source_embedding_layer = model.layers[0].get_weights()[0][1:-1, :]
    with open(os.path.join(log_dir, 'metadata_source.tsv'), "w", encoding='utf-8') as out_meta_source:
        with open(os.path.join(log_dir, 'metadata_target.tsv'), "w", encoding='utf-8') as out_meta_target:
            out_meta_source.write('source'+ "\t"+ 'target' + "\n")
            out_meta_target.write('source'+ "\t"+ 'target' + "\n")
            # Remove tabs, newlines and spaces from the paragraph
            for source, target in sentence_pair:
                source_embedding_vector=tokenize_and_aggregate(source, source_tokenizer, agg, source_embedding_layer)
                target_embedding_vector=tokenize_and_aggregate(target, target_tokenizer, agg, target_embedding_layer)
                #test the above
                out_meta_source.write(source+ "\t"+ target + "\n")
                out_meta_target.write(source+ "\t"+ target + "\n")
                souce_sentences.append(source)
                target_sentences.append(target)
                source_sentence_vector.append(source_embedding_vector)
                target_sentence_vector.append(target_embedding_vector)
            source_sentence_vector = display_embedding_shape(source_sentence_vector, souce_sentences, 'source')
            target_sentence_vector = display_embedding_shape(target_sentence_vector, target_sentences, 'target')
            checkpoint = tf.train.Checkpoint(source_embedding=tf.Variable(source_sentence_vector),
                                             target_embedding=tf.Variable(target_sentence_vector))
            config = projector.ProjectorConfig()
            config = save_checkpoint_create_config(checkpoint, config, 'source', log_dir)
            config = save_checkpoint_create_config(checkpoint, config, 'target', log_dir)
            projector.visualize_embeddings(log_dir, config)

    return (souce_sentences, target_sentences, source_sentence_vector, target_sentence_vector)

if __name__ == '__main__':
    # Instantiate the model
    temp_input = tf.random.uniform((2, 19), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((2, 12), dtype=tf.int64, minval=0, maxval=200)
    _ = Model(temp_input,
            dec_padding_mask=None, 
            enc_padding_mask=None, 
            look_ahead_mask=None,
            target_ids=temp_target, 
            training=False, 
            )
    ck_pt_mgr = check_ckpt(config.checkpoint_path)
    log_dir = os.path.join(config.tensorboard_log, embedding_projector_dir)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    filename = input('Enter the filename:- ')
    file_path = os.path.join(config.output_sequence_write_path, filename)
    input_sentences = []
    hypothesis = []
    with tf.io.gfile.GFile(file_path, 'r') as f:
        for line in f.readlines():
            (source, _, hyp) = line.split('\t')
            input_sentences.append(source)
            hypothesis.append(hyp)
    sentence_pair = zip(input_sentences, hypothesis)
    source, target, vec1, vec2 = embedding_projector_files(source_tokenizer, target_tokenizer, 
                                        Model, sentence_pair, log_dir=log_dir)
    print(f"Model's checkpoint used {config.checkpoint_path}")
    print(f'Tensorboard directory {log_dir}')
    #print('Executing tensorboard..might take few mins to load so please refresh after few mins')
    os.system(f"tensorboard --logdir {log_dir}")
    
