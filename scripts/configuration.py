# -*- coding: utf-8 -*-
from bunch import Bunch
from input_path import file_path

model_parms = {
     'copy_gen':True,
     'doc_length': 512,
     'd_model': 768,                  # the projected word vector dimension
     'dff': 2048,                      # feed forward network hidden parameters
     'input_vocab_size': 30522,        # total vocab size + start and end token
     'num_heads': 8,                  # the number of heads in the multi-headed attention unit
     'num_layers': 8,                 # number of transformer blocks
     'pretrained_bert_model': 'bert-base-uncased',
     'summ_length': 72,
     'target_vocab_size': 30522,       # total vocab size + start and end token
     }                                    

training_parms = {
     'early_stop' : False,
     'eval_after' : 5000,              # Evaluate once this many samples are trained 
     'last_recorded_value': None,
     'monitor_metric' : 'combined_metric',
     'max_tokens_per_line' : 1763,      # filter documents based on this many tokens
     'print_chks': 50,                  # print training progress per number of batches specified
     'run_tensorboard': False,
     'show_detokenized_samples' : False,
     'tfds_name' : 'cnn_dailymail',     # tfds dataset to be used
     'tolerance_threshold': 5,          # Stop training after the threshold is reached
     'use_tfds' : True,                 # use tfds datasets as to train the model else use the given csv file
     'valid_samples_to_eval' : 100,     # number of samples used for validation
     'write_summary_op': True           # write the first batch of validation set summary to a file
     }                                    

# Use the csv dataset if tfds is false
if not training_parms['use_tfds']:
  training_parms['num_examples_to_train'] = None     # If None then all the examples in the dataset will be used to train
  training_parms['num_examples_to_infer'] = None
  training_parms['test_size'] = 0.05                 # test set split size

config = Bunch(model_parms)
config.update(training_parms)


# Get last_recorded_value of monitor_metric from the log
try:
  with open(file_path.log_path) as f:
    for line in reversed(f.readlines()):
        if ('- tensorflow - INFO - '+ config.monitor_metric in line) and \
          (line[[i for i,char in enumerate((line)) if char.isdigit()][-1]+1] == '\n'):          
          config['last_recorded_value'] = float(line.split(config.monitor_metric)[1].split('\n')[0].strip())
          print(f"last_recorded_value of {config.monitor_metric} retained from last run {config['last_recorded_value']}")
          break
        else:
          continue
  if not config['last_recorded_value']:
    print('setting default value to last_recorded_value')
    config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')
except FileNotFoundError:
  print('setting default value to last_recorded_value')
  config['last_recorded_value'] = 0 if config.monitor_metric != 'validation_loss' else float('inf')
