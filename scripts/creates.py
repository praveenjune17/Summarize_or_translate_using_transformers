# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import logging
from configuration import config


def check_and_create_dir():
    for key in config.keys():
        if key in ['best_ckpt_path', 'initial_weights', 'output_sequence_write_path', 'tensorboard_log']:
            if not os.path.exists(config[key]):
                os.makedirs(config[key])

def detokenize(target_tokenizer, id_1, id_2, source_tokenizer=None):
    if source_tokenizer is None:
        source_tokenizer = target_tokenizer
        detokenized_seq_1 = source_tokenizer.decode([i for i in id_1 if i not in [config.target_CLS_ID, 
                                                                                 config.target_SEP_ID, 
                                                                                 config.PAD_ID]])
        detokenized_seq_2 = target_tokenizer.decode([i for i in id_2 if i not in [config.target_CLS_ID, 
                                                                                 config.target_SEP_ID, 
                                                                                 config.PAD_ID]])
    return (detokenized_seq_1, detokenized_seq_2)

# Create logger
def create_logger():
    log = logging.getLogger('tensorflow')
    log.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh = logging.FileHandler(config.log_path, 'w', 'utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    log.addHandler(fh)
    log.propagate = False
    return log


def set_memory_growth(log):
    # Set GPU memory growth
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if not gpu_devices:
    	log.warning("GPU not available so Running in CPU")
    else:
        for device in gpu_devices:
         tf.config.experimental.set_memory_growth(device, True)
         log.info('GPU memory growth set')

def create_tensorboard_parms():
    if config.run_tensorboard:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = config.tensorboard_log + current_time + '/train'
        validation_log_dir = config.tensorboard_log + current_time + '/validation'
        train_output_sequence_writer = tf.summary.create_file_writer(train_log_dir)
        valid_output_sequence_writer = tf.summary.create_file_writer(validation_log_dir)
    else:
        train_output_sequence_writer = None
        valid_output_sequence_writer = None
    return (train_output_sequence_writer,
            valid_output_sequence_writer)

# create metrics dict
def validate_config_parameters():
    monitor_metrics = dict()
    monitor_metrics['validation_loss'] = None
    monitor_metrics['BERT_f1'] = None
    monitor_metrics['ROUGE_f1'] = None
    monitor_metrics['combined_metric'] = (
                                        monitor_metrics['BERT_f1'], 
                                        monitor_metrics['ROUGE_f1']
                                        )
    assert config.monitor_metric in monitor_metrics.keys(), f'Available metrics to monitor are {monitor_metrics.keys()}'
    assert sum(config.combined_metric_weights) == 1, 'weights should sum to 1'
    assert config.d_model % config.num_heads == 0, 'd_model should be a multiple of num_heads'
    if config.task.lower() == 'summarization':
        assert config.input_pretrained_bert_model == config.target_pretrained_bert_model, f'For {config.task}\
        the input and target models must be same'
        assert config.input_CLS_ID ==  config.target_CLS_ID, 'Start Ids must be same'
        assert config.input_SEP_ID ==  config.target_SEP_ID, 'End Ids must be same'
    elif config.task.lower() == 'translation':
        if config.use_BERT:
            assert config.input_pretrained_bert_model != config.target_pretrained_bert_model, f'For {config.task}\
        the input and target models must not be same'
        if (config.input_CLS_ID ==  config.target_CLS_ID) or (config.input_SEP_ID ==  config.target_SEP_ID):
            if not config.use_BERT:
                assert config.target_vocab_size == config.input_vocab_size, 'Vocab size not same so ids should not be same too'
            
    # create folder in input_path if they don't exist
    if not config.use_tfds:
        assert os.path.exists(config.train_csv_path), 'Training dataset not available'
    if config.print_config:
        log.info(f'Configuration used \n {config}')
    if config.test_script:
        log.info(f'Setting Low Configuration to the model parameters since test_script is enabled')

def check_recorded_metric_val():
    # Get last_recorded_value of monitor_metric from the log
    try:
        with open(config.log_path) as f:
            for line in reversed(f.readlines()):
                if ('- tensorflow - INFO - '+ config.monitor_metric in line) and \
                    (line[[i for i,char in enumerate((line)) if char.isdigit()][-1]+1] == '\n'):          
                    config['last_recorded_value'] = float(line.split(
                                                    config.monitor_metric)[1].split('\n')[0].strip())
                    log.info(f"last recorded_value of {config.monitor_metric} retained from last \
                                                            run {config['last_recorded_value']}")
                    break
                else:
                    continue
                if not config['last_recorded_value']:
                    log.info('setting default value to the last_recorded_value since not \
                                able to find the metrics from the log')
                    config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')
    except FileNotFoundError:
        log.info('setting default value to the last_recorded_value since file was not found')
        config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')


log = create_logger()
set_memory_growth(log)
check_and_create_dir()
validate_config_parameters()
(train_output_sequence_writer,
valid_output_sequence_writer) = create_tensorboard_parms()
if config.use_last_recorded_value:
    check_recorded_metric_val()