# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import shutil
import logging
from configuration import config

def detokenize(target_tokenizer, id_1, id_2, source_tokenizer=None):

    if source_tokenizer is None:
        source_tokenizer = target_tokenizer
        cls_id = config.target_CLS_ID
        sep_id = config.target_SEP_ID
    else:
        cls_id = config.input_CLS_ID
        sep_id = config.input_SEP_ID
    detokenized_seq_1 = source_tokenizer.decode([i for i in id_1 if i not in [cls_id, 
                                                                             sep_id, 
                                                                             config.PAD_ID]])
    if id_2 is not None:
        detokenized_seq_2 = target_tokenizer.decode([i for i in id_2 if i not in [config.target_CLS_ID, 
                                                                             config.target_SEP_ID, 
                                                                             config.PAD_ID]])
    else:
        detokenized_seq_2 = None
    
    return (detokenized_seq_1, detokenized_seq_2)

def check_and_create_dir():

    for key in config.keys():
        if key in ['best_ckpt_path', 'initial_weights', 'output_sequence_write_path', 'tensorboard_log']:
            if key == 'tensorboard_log':
                try:
                    shutil.rmtree(config[key])
                except (FileNotFoundError, OSError) as e:
                    continue
                os.makedirs(config[key])
            if not os.path.exists(config[key]):
                os.makedirs(config[key])

def create_logger():

    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(config.log_path, 'w' if config.clear_log else 'a', 'utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.propagate = False

    return log

def create_tensorboard_parms():

    if config.run_tensorboard:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = config.tensorboard_log + current_time + '/train'
        validation_log_dir = config.tensorboard_log + current_time + '/validation'
        embedding_projector_dir = config.tensorboard_log + current_time + '/embedding_projector'
        train_output_sequence_writer = tf.summary.create_file_writer(train_log_dir)
        valid_output_sequence_writer = tf.summary.create_file_writer(validation_log_dir)
    else:
        train_output_sequence_writer = None
        valid_output_sequence_writer = None
        embedding_projector_dir = None

    return (train_output_sequence_writer,
            valid_output_sequence_writer,
            embedding_projector_dir)

def check_recorded_metric_val():
    # Get last_recorded_value of monitor_metric from the log
    try:
        with open(config.log_path, 'r', encoding='utf-8') as f:
            for line in reversed(f.readlines()):
                if ('- tensorflow - INFO - '+ config.monitor_metric in line) and \
                    (line[[i for i,char in enumerate((line)) if char.isdigit()][-1]+1] == '\n'):
                    config.last_recorded_value = float(line.split(
                                                    config.monitor_metric)[1].split('\n')[0].strip())
                    log.info(f"last recorded_value of {config.monitor_metric} retained from last \
                                                            run {config.last_recorded_value}")
                    break
                else:
                    continue
        if config.last_recorded_value is None:
            log.info('setting default value to the last_recorded_value since not \
                        able to find the metrics from the log')
            config.last_recorded_value = 0 if config.monitor_metric != 'validation_loss' else float('inf')
    except FileNotFoundError:
        log.info('setting default value to the last_recorded_value since file was not found')
        config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')



check_and_create_dir()
log = create_logger()
if config.last_recorded_value is None:
    check_recorded_metric_val()