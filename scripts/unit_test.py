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
from calculate_metrics import mask_and_one_hot_labels, monitor_run
from creates import log
from create_model import source_tokenizer, target_tokenizer
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set)

unit_test_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              batch_size=config.unit_test_dataset_batch_size
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
                              batch_size=batch_size
                              )
    training_loop(memory_test_dataset.repeat(10), False)
    gpu_usage = check_gpu_usage()
    log.info(f'Training with tokens_per_batch set to {addtional_tokens_per_batch}\
               and batch_size set to {batch_size}')
    log.info(f'GPU memory utilization is {gpu_usage}')
    return gpu_usage

def training_loop(dataset, check_model_capacity):
    if check_model_capacity:
        dataset = dataset.repeat(int(config.epochs))
    for (step, (input_ids, target_ids)) in tqdm(enumerate(dataset), initial=1):
        min_loss = 10000000
        start=time.time()
        draft_mask, refine_mask, target_ids_3D = mask_and_one_hot_labels(target_ids)
        grad_accum_flag = True if (step+1)%config.gradient_accumulation_steps == 0 else False
        predictions = train_step(
                                  input_ids,  
                                  target_ids, 
                                  grad_accum_flag
                                  )

        if grad_accum_flag:
            train_loss = batch_run_check(
                                      step+1,  
                                      start
                                      )
            if check_model_capacity:
                if min_loss > train_loss:
                    min_loss = train_loss
                else:
                    log.warning('Loss not decreasing watch out')

    if config.detokenize_samples:
        _ = train_sanity_check(target_tokenizer, predictions, target_ids)

    if check_model_capacity:
      
        if train_loss < config.min_train_loss:
            log.info('Minimum training loss reached')
        else:
            log.info("Loss didn't reach upto the min_train_loss specified, try to increase \
                  the parameters of the model and check the run again")

if config.random_results_check:
    training_loop(unit_test_dataset, False)
    log.info('First run completed')
    training_loop(unit_test_dataset, False)
    log.info('Second run completed')
    training_loop(unit_test_dataset, False)
    log.info('Third run completed')
    log.info('Verify whether the results are same')
    sys.exit()

if config.init_loss_check:
    input_ids, target_ids = next(iter(unit_test_dataset))
    draft_mask, refine_mask, target_ids_3D = mask_and_one_hot_labels(target_ids)
    loss =  eval_step(
                      input_ids,  
                      target_ids, 
                      )
    log.info(f"Model's Initial loss {loss}")
    log.info(f'Expected initial loss {tf.math.log(tf.cast(config.target_vocab_size, dtype=tf.float32))*config.num_of_decoders}')
    log.info(f'Initial Loss check run completed')

if config.input_independent_baseline_check:
    for (step, (input_ids, target_ids)) in tqdm(enumerate(unit_test_dataset), initial=1):
        start=time.time()
        input_ids = tf.zeros_like(input_ids)
        target_ids = tf.zeros_like(target_ids)
        draft_mask, refine_mask, target_ids_3D = mask_and_one_hot_labels(target_ids)
        grad_accum_flag = True if (step+1)%config.gradient_accumulation_steps == 0 else False
        _ = train_step(
                      input_ids,  
                      target_ids, 
                      target_ids_3D, 
                      draft_mask,
                      refine_mask,
                      grad_accum_flag
                      )
        batch_run_check(
                      step+1,  
                      start
                      )
    log.info('Input Independent baseline run completed')
    sys.exit()

if config.check_model_capacity:
    training_loop(unit_test_dataset, True)
    sys.exit()

if config.check_training_pipeline:
    training_loop(unit_test_dataset, False)
    
if config.check_evaluation_pipeline:
    #ck_pt_mgr = check_ckpt(config.checkpoint_path)
    rouge, bert = evaluate_validation_set(
                              unit_test_dataset, 
                              1
                              )
    monitor_early_stop = monitor_run(
                                    'not saving', 
                                    bert, 
                                    rouge,
                                    0.0, 
                                    1
                                    )
    if not monitor_early_stop:
        break
    log.info(f'Validation run completed with ROUGE-avg {rouge} and BERT-f1 {bert}\
               Check the written summary file in {config.output_sequence_write_path}')
    sys.exit()

if config.detokenize_samples:
    source_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.input_seq_vocab_path)
    target_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.output_seq_vocab_path)
  
    sample_string = 'Transformer is awesome.'

    tokenized_string = source_tokenizer.encode(sample_string)
    print ('Tokenized string is {}'.format(tokenized_string))

    original_string = source_tokenizer.decode(tokenized_string)
    print ('The original string: {}'.format(original_string))

    assert original_string == sample_string, 'Encoding issue with tokenizer'
    sys.exit()

if config.check_predictions_shape:
    temp_input = tf.random.uniform((64, 38), dtype=tf.int64, minval=0, maxval=200)
    temp_target = tf.random.uniform((64, 36), dtype=tf.int64, minval=0, maxval=200)
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
    fn_out, _ = sample_model(temp_input,
                             dec_padding_mask=None, 
                             enc_padding_mask=None, 
                             look_ahead_mask=None,
                             target_ids=temp_target, 
                             training=False, 
                             )
    log.info(f'The output shape of the sample model is {fn_out.shape}')
    sys.exit()

if config.GPU_memory_test:
    gpu_usage = check_gpu_usage()
    while float(gpu_usage[:-1]) < 80:
        gpu_usage = change_dataset_and_train(config.tokens_per_batch, config.train_batch_size)
        config.tokens_per_batch += 500
        log.info('Increasing tokens_per_batch to 500')
    log.info('GPU memory exceeded "80%" hence stopping the training')
    sys.exit()