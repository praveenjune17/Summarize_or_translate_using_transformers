# -*- coding: utf-8 -*-
import os
from bunch import Bunch

core_path = os.getcwd() 

unit_test = {
      'test_script' : True,
      'init_loss_check' : True,
      'samples_to_test' : 1,
      'save_initial_weights' : False,
      'input_independent_baseline_check' : True, 
      'check_model_capacity' : True,
      'random_results_check' : True
      } 
model_parms = {
     'activation' : 'relu',
     'copy_gen': True,
     'input_seq_length': 60,
     'd_model': 768,                  # the projected word vector dimension
     'dff': 2048,                      # feed forward network hidden parameters
     'input_vocab_size': 30522,        # total vocab size + start and end token
     'num_heads': 8,                  # the number of heads in the multi-headed attention unit
     'num_layers': 8,                 # number of transformer blocks
     'input_pretrained_bert_model': 'bert-base-uncased',
     'target_pretrained_bert_model' : 'bert-base-multilingual-cased',
     'target_seq_length': 40,
     'target_vocab_size': 119547,       # total vocab size + start and end token
     }                                    
training_parms = {
     'display_model_summary' : True,
     'early_stop' : False,
     'enable_jit' : True,
     'eval_after' : 5000,              # Evaluate once this many samples are trained 
     'last_recorded_value': None,
     'max_tokens_per_line' : model_parms['input_seq_length']+model_parms['target_seq_length'],      # filter documents based on this many tokens
     'min_train_loss' : 0.5,
     'monitor_metric' : 'combined_metric',
     'print_chks': 50,                  # print training progress per number of batches specified
     'run_tensorboard': False,
     'tfds_name' : 'en_tam_parallel_text',     # tfds dataset to be used
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

# Special Tokens
special_tokens = {
          'CLS_ID' : 101,
          'MASK_ID' : 103,
          'PAD_ID' : 0,
          'SEP_ID' : 102,
          'UNK_ID' : 100
          }


h_parms = {
   'gradient_accumulation_steps': 36,                                                                                   
   'train_batch_size': 1,
   'beam_sizes': [2, 3, 4],              # Used  during inference                                                 
   'combined_metric_weights': [0.6, 0.3, 0.1], #(bert_score, rouge, validation accuracy)
   'dropout_rate': 0.0,
   'epochs': 1,
   'epsilon_ls': 0.0,                    # label_smoothing hyper parameter
   'grad_clipnorm':None,
   'l2_norm':0.0,
   'learning_rate': None,                # change to None to set learning rate decay
   'length_penalty' : 1,                       # Beam search hyps . Used  during inference                                                 
   'mean_attention_heads':True,                # if False then the attention parameters of the last head will be used
   'mean_attention_parameters_of_layers':True,           # if False then the attention parameters of the last layer will be used
   'validation_batch_size' : 8
   }                                    

dataset_name = training_parms['tfds_name']
file_path = {
        'best_ckpt_path' : f"/content/drive/My Drive/best_checkpoints/{dataset_name}/",  
        'checkpoint_path' : f"/content/drive/My Drive/checkpoints/{dataset_name}/",
        'initial_weights' : f"/content/drive/My Drive/initial_weights/{dataset_name}/",
        'infer_csv_path' : None,
        'infer_ckpt_path' : None,
        'log_path' : f"/content/drive/My Drive/created_files/{dataset_name}/tensorflow.log",
        'output_sequence_write_path' : f"/content/drive/My Drive/created_files/{dataset_name}/summaries/{dataset_name}/",
        'tensorboard_log' : f"/content/drive/My Drive/created_files/{dataset_name}/tensorboard_logs/",
        'tfds_data_dir' : f'/content/drive/My Drive/Tensorflow_datasets/{dataset_name}_dataset',
        'tfds_data_version' : None,#{"version": "2.0.0"},
        'train_csv_path' : None,
            }

config = Bunch(model_parms)
config.update(unit_test)
config.update(training_parms)
config.update(special_tokens)
config.update(h_parms)
config.update(file_path)

if config.test_script:
  config.gradient_accumulation_steps =  config.samples_to_test
  config.epochs = 1000
  config.dff = 512                      # feed forward network hidden parameters
  config.num_heads = 4                  # the number of heads in the multi-headed attention unit
  config.num_layers = 2                 # number of transformer blocks
  assert config.d_model % config.num_heads == 0, 'd_model should be a multiple of num_heads'
  config.dropout_rate = config.epsilon_ls = 0.0
  config.grad_clipnorm = None
  config.l2_norm = 0.0
  config.copy_gen = False
else:
  config.samples_to_test = -1
  config.save_initial_weights = False
  config.run_init_eval = False
  config.init_loss_check = False
  config.input_independent_baseline_check = False
  config.check_model_capacity = False
  config.random_results_check = False
