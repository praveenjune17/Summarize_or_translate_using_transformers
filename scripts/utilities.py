# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import shutil
import logging
from configuration import config

def set_memory_growth(log):
    # Set GPU memory growth
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    if not gpu_devices:
        log.warning("GPU not available so Running in CPU")
    else:
        for device in gpu_devices:
         tf.config.experimental.set_memory_growth(device, True)
         log.info('GPU memory growth set')

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
            

# Create logger
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
# create metrics dict
def validate_config_parameters():
    allowed_decoder_types = ['topktopp','greedy', 'only_beam_search']
    allowed_model_architectures = ['transformer', 'bertified_transformer']
    if config.add_bias:
        if not config.target_language == 'ta':
            assert config.target_pretrained_bert_model == 'bert-base-multilingual-cased', 'Bias is available only for en-ta translation and bert-multilingual model'
    assert config.d_model % config.num_heads == 0, 'd_model should be a multiple of num_heads'
    assert config.eval_after_steps%config.steps_to_print_training_info == 0, 'steps_to_print_training_info must be a factor of eval_after_steps'
    assert config.draft_decoder_type  in allowed_decoder_types, f'available decoding types are {allowed_decoder_types}'
    assert config.model_architecture  in allowed_model_architectures, f'available model_architectures are {allowed_model_architectures}'
    if config.task.lower() == 'summarize':
        assert config.input_pretrained_bert_model == config.target_pretrained_bert_model, f'For {config.task}\
        the input and target models must be same'
        assert config.input_CLS_ID ==  config.target_CLS_ID, 'Start Ids must be same'
        assert config.input_SEP_ID ==  config.target_SEP_ID, 'End Ids must be same'
    elif config.task.lower() == 'translate':
        if config.model_architecture == 'bertified_transformer':
            assert config.input_pretrained_bert_model != config.target_pretrained_bert_model, f'For {config.task}\
        the input and target models must not be same'
        if (config.input_CLS_ID ==  config.target_CLS_ID) or (config.input_SEP_ID ==  config.target_SEP_ID):
            if not config.model_architecture == 'bertified_transformer':
                assert config.target_vocab_size == config.input_vocab_size, 'Vocab size not same so ids should not be same too'
    else:
        raise ValueError('Incorrect task.. please change it to summarize or translate only')  
    # create folder in input_path if they don't exist
    if not config.use_tfds:
        assert os.path.exists(config.train_csv_path), 'Training dataset not available'
    if config.print_config:
        log.info(f'Configuration used \n {config}')
    if config.test_script:
        log.info(f'Setting Low Configuration to the model parameters since test_script is enabled')


check_and_create_dir()
log = create_logger()
set_memory_growth(log)
validate_config_parameters()
if config.last_recorded_value is None:
    check_recorded_metric_val()