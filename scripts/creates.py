# -*- coding: utf-8 -*-
import datetime
import tensorflow as tf
import os
import logging
from configuration import config


gpu_devices = tf.config.experimental.list_physical_devices('GPU')

def check_and_create_dir(path):
    if not os.path.exists(path):
      os.makedirs(path)
      print(f'directory {path} created ')
        
# create folder in input_path if they don't exist
if not config.use_tfds:
  assert (os.path.exists(config.train_csv_path)), 'Training dataset not available'
for key in config.keys():
  if key in ['best_ckpt_path', 'output_sequence_write_path', 'tensorboard_log']:
    check_and_create_dir(config[key])
              
# Create logger
log = logging.getLogger('tensorflow')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh = logging.FileHandler(config.log_path)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
log.addHandler(fh)
log.propagate = False

# Set GPU memory growth
if not gpu_devices:
    log.warning("GPU not available so Running in CPU")
else:
  for device in gpu_devices:
      tf.config.experimental.set_memory_growth(device, True)
      log.info('GPU memory growth set')

if config.run_tensorboard:
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = config.tensorboard_log + current_time + '/train'
    validation_log_dir = config.tensorboard_log + current_time + '/validation'
    train_output_sequence_writer = tf.summary.create_file_writer(train_log_dir)
    valid_output_sequence_writer = tf.summary.create_file_writer(validation_log_dir)
else:
    train_output_sequence_writer = None
    valid_output_sequence_writer = None
        
# create metrics dict
monitor_metrics = dict()
monitor_metrics['validation_loss'] = None
monitor_metrics['BERT_f1'] = None
monitor_metrics['ROUGE_f1'] = None
monitor_metrics['validation_accuracy'] = None
monitor_metrics['combined_metric'] = (
                                      monitor_metrics['BERT_f1'], 
                                      monitor_metrics['ROUGE_f1'], 
                                      monitor_metrics['validation_accuracy']
                                      )
assert (config.monitor_metric in monitor_metrics.keys()), f'Available metrics to monitor are {monitor_metrics.keys()}'
assert (tf.reduce_sum(config.combined_metric_weights) == 1), 'weights should sum to 1'
assert config.PAD_ID == 0, 'Change the padding values in the tf_dataset.padded_batch line of preprocess script'
log.info(f'Configuration used \n {config}')
if config.test_script:
  log.info(f'Setting Low Configuration to the model parameters since test_script is enabled')
# Get last_recorded_value of monitor_metric from the log
try:
  with open(config.log_path) as f:
    for line in reversed(f.readlines()):
        if ('- tensorflow - INFO - '+ config.monitor_metric in line) and \
          (line[[i for i,char in enumerate((line)) if char.isdigit()][-1]+1] == '\n'):          
          config['last_recorded_value'] = float(line.split(config.monitor_metric)[1].split('\n')[0].strip())
          print(f"last_recorded_value of {config.monitor_metric} retained from last run {config['last_recorded_value']}")
          break
        else:
          continue
  if not config['last_recorded_value']:
    print('setting default value to the last_recorded_value since not able to find the metrics from the log')
    config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')
except FileNotFoundError:
  print('setting default value to the last_recorded_value since file was not found')
  config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')
