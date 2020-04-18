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
from creates import log
from create_model import source_tokenizer, target_tokenizer
from local_tf_ops import (check_ckpt, eval_step, train_step, batch_run_check, 
                          train_sanity_check, evaluate_validation_set, training_results)

train_dataset = create_dataset(
                              split='train', 
                              source_tokenizer=source_tokenizer, 
                              target_tokenizer=target_tokenizer, 
                              from_=90, 
                              to=100, 
                              batch_size=config.train_batch_size,
                              buffer_size=420000
                              )
val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size,
                             drop_remainder=True
                             )

# if a checkpoint exists, restore the latest checkpoint.
ck_pt_mgr = check_ckpt(config.checkpoint_path)
total_steps = int(config.epochs * (config.gradient_accumulation_steps))
train_dataset = train_dataset.repeat(total_steps)
for (step, (input_ids, target_ids)) in tqdm(enumerate(train_dataset, 1), initial=1):
    start=time.time()
    grad_accum_flag = (True if ((step)%config.gradient_accumulation_steps) == 0 else False) if config.accmulate_gradients else None
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
    if step == config.eval_after_steps:
        ckpt_save_path = ck_pt_mgr.save()
        predicted = train_sanity_check(target_tokenizer, predictions, target_ids)
        # Run evaluation only if the predictions made by the teacher forced output is not empty
        if predicted:
            (rouge_score, bert_score) = evaluate_validation_set(
                                                              val_dataset, 
                                                              step
                                                              )
        else:
            rouge_score, bert_score = 0
        training_results(
                          step, 
                          rouge_score, 
                          bert_score,
                          (time.time() - start),
                          ckpt_save_path
                          )
        monitor_early_stop = monitor_run(
                                        ckpt_save_path, 
                                        bert_score, 
                                        rouge_score,
                                        train_loss, 
                                        step
                                        )
        if not monitor_early_stop:
            break
log.info(f'Training completed at step {step}')