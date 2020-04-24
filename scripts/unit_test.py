# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
tf.keras.backend.clear_session()
tf.random.set_seed(100)
import time
import sys
import GPUtil
from io import StringIO
from tqdm import tqdm
from preprocess import create_dataset
from configuration import config
from calculate_metrics import mask_and_calculate_loss, monitor_run
from creates import log, detokenize
from create_model import source_tokenizer, target_tokenizer, Model
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set)

unit_test_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              batch_size=config.unit_test_dataset_batch_size,
                              drop_remainder=True
                              )

def check_gpu_usage():

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    GPUtil.showUtilization()
    sys.stdout = old_stdout
    gpu_usage = mystdout.getvalue().strip().split('|')[-2].strip()

    return gpu_usage

def change_dataset_and_train(addtional_tokens_per_batch, batch_size):
    
    memory_test_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              shuffle=True,
                              batch_size=batch_size
                              )
    log.info(f'Training with tokens_per_batch set to {addtional_tokens_per_batch}\
               and batch_size set to {batch_size}')
    training_loop(memory_test_dataset.take(1000), False)
    gpu_usage = check_gpu_usage()
    log.info(f'GPU memory utilization is {gpu_usage}')

    return gpu_usage

def training_loop(dataset, check_model_capacity, detokenize_samples=None):

    min_loss = 10000000
    if check_model_capacity:
        dataset = dataset.repeat(670)
    for (step, (input_ids, target_ids)) in tqdm(enumerate(dataset, 1), initial=1):
        start=time.time()
        grad_accum_flag = (True if ((step)%config.gradient_accumulation_steps) == 0 else False) if config.accumulate_gradients else None
        predictions = train_step(
                                  input_ids,  
                                  target_ids, 
                                  grad_accum_flag
                                  )
        if grad_accum_flag is not None:
            if grad_accum_flag:
                if (step)%config.steps_to_print_training_info==0:
                    predicted_ids = train_sanity_check(target_tokenizer, predictions, target_ids)
                    train_loss = batch_run_check(
                                              step,  
                                              start
                                              )
        else:
            if (step)%config.steps_to_print_training_info==0:
                train_loss = batch_run_check(
                                          step,  
                                          start
                                          )
            if check_model_capacity:
                if min_loss > train_loss:
                    min_loss = train_loss
                else:
                    log.warning('Loss not decreasing watch out')
                    monitor_early_stop = monitor_run(
                                    'not saving', 
                                    0, 
                                    0,
                                    0.0, 
                                    1,
                                    copy_best_ckpt=False
                                    )
                    
    if check_model_capacity:
        log.info(f'target_ids are {target_ids}')
        log.info(f'predicted ids are {predicted_ids}')
        if train_loss < config.min_train_loss:
            log.info('Minimum training loss reached')
        else:
            log.info("Loss didn't reach upto the min_train_loss specified, try to increase\
                  the parameters of the model or number of train steps")
        

if config.random_results_check:
    training_loop(unit_test_dataset, False)
    log.info('First run completed')
    training_loop(unit_test_dataset, False)
    log.info('Second run completed')
    training_loop(unit_test_dataset, False)
    log.info('Third run completed')
    log.info('Verify whether the loss stays same for the three runs')

if config.init_loss_check:
    input_ids, target_ids = next(iter(unit_test_dataset))
    loss =  eval_step(
                      input_ids,  
                      target_ids, 
                      )
    log.info(f"Model's Initial loss {loss}")
    log.info(f'Expected initial loss {tf.math.log(tf.cast(config.target_vocab_size, dtype=tf.float32))*config.num_of_decoders}')
    log.info(f'Initial Loss check run completed')

if config.input_independent_baseline_check:
    for (step, (input_ids, target_ids)) in tqdm(enumerate(unit_test_dataset, 1), initial=1):
        start=time.time()
        input_ids = tf.zeros_like(input_ids)
        target_ids = tf.zeros_like(target_ids)
        grad_accum_flag = True if (step)%config.gradient_accumulation_steps == 0 else False
        predictions = train_step(
                            input_ids,  
                            target_ids,
                            grad_accum_flag
                            )
        if grad_accum_flag is not None:
            if grad_accum_flag:
                if (step)%config.steps_to_print_training_info==0:
                    train_loss = batch_run_check(
                                              step,  
                                              start
                                              )
            else:
                if (step)%config.steps_to_print_training_info==0:
                    train_loss = batch_run_check(
                                              step,  
                                              start
                                              )
    log.info('Input Independent baseline run completed')
    

if config.check_model_capacity:

    training_loop(unit_test_dataset, True)
    

if config.check_training_pipeline:

    training_loop(unit_test_dataset, False)
    
if config.check_evaluation_pipeline:

    ck_pt_mgr = check_ckpt(config.checkpoint_path)
    rouge_score, bert_score = evaluate_validation_set(
                              unit_test_dataset, 
                              config.beam_size,
                              config.length_penalty,
                              1
                              )
    monitor_early_stop = monitor_run(
                                    'not saving', 
                                    bert_score, 
                                    rouge_score,
                                    0.0, 
                                    1,
                                    copy_best_ckpt=False
                                    )
    if not monitor_early_stop:

        log.info(f'Validation run completed with ROUGE-avg {rouge_score} and BERT-f1 {bert_score}\
               Check the written summary file in {config.output_sequence_write_path}')
    
    

if config.detokenize_samples:

    training_loop(unit_test_dataset, False, True)
    sample_string = 'Transformer is awesome.'
    tokenized_string = source_tokenizer.encode(sample_string)
    log.info('Tokenized string is {}'.format(tokenized_string))
    original_string = source_tokenizer.decode(tokenized_string)
    log.info('The original string: {}'.format(original_string))
    assert original_string == sample_string, 'Encoding issue with tokenizer'
    

if config.check_predictions_shape:

    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
    (draft_predictions, draft_attention_weights, 
    refine_predictions, refine_attention_weights) = Model(temp_input,
                                                       dec_padding_mask=None, 
                                                       enc_padding_mask=None, 
                                                       look_ahead_mask=None,
                                                       target_ids=temp_target, 
                                                       training=False, 
                                                       )
    log.info(f'The output shape of the sample model is {tf.shape(draft_predictions if refine_predictions is None else refine_predictions)}')
    

if config.gpu_memory_test:

    memory_limit = 85
    gpu_usage = check_gpu_usage()
    while float(gpu_usage[:-1]) < memory_limit:
        gpu_usage = change_dataset_and_train(config.tokens_per_batch, config.train_batch_size)
        config.tokens_per_batch += 50
    log.info(f'GPU memory exceeded {memory_limit}% hence stopping the training')

sys.exit()