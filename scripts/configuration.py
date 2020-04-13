# -*- coding: utf-8 -*-
import os
from bunch import Bunch

unit_test = {
      'check_evaluation_pipeline' : True,
      'check_model_capacity' : False,
      'check_training_pipeline' : False,
      'detokenize_samples' : True,
      'init_loss_check' : False,
      'input_independent_baseline_check' : False, 
      'print_config' : True,
      'random_results_check' : False,
      'samples_to_test' : 4,
      'save_initial_weights' : False,
      'test_script' : False,
      'unit_test_dataset_batch_size' : 2
          }


model_parms = {
     'add_bias' : None,               # set values as True|None Increases the inital bias of Tamil vocabs
     'activation' : 'relu',
     'bert_score_model' : 'bert-base-multilingual-cased',
     'copy_gen': True,
     'input_seq_length': 60,
     'd_model': 768,                  # the projected word vector dimension
     'dff': 2048,                      # feed forward network hidden parameters
     'input_vocab_size': 8247+2,        # total vocab size + start and end token
     'num_heads': 8,                  # the number of heads in the multi-headed attention unit
     'num_layers': 8,                 # number of transformer blocks
     'input_pretrained_bert_model': 'bert-base-uncased',
     'task':'translation',
     'target_pretrained_bert_model' : 'bert-base-multilingual-cased',
     'target_seq_length': 40,
     'target_vocab_size': 8294+2,
     'use_BERT' : False
     }   

training_parms = {
     'display_model_summary' : True,
     'early_stop' : True,
     'enable_jit' : True,
     'eval_after' : 5000,              # Evaluate once this many samples are trained 
     'last_recorded_value': 0.00,
     'max_tokens_per_line' : model_parms['input_seq_length']+model_parms['target_seq_length'],      # filter documents based on this many tokens
     'min_train_loss' : 1,
     'monitor_metric' : 'combined_metric',
     'print_chks': 50,                  # print training progress per number of batches specified
     'run_tensorboard': True,
     'tfds_name' : 'en_tam_parallel_text',     # tfds dataset to be used
     'tolerance' : 0,
     'tolerance_threshold': 5,          # Stop training after the threshold is reached
     'use_tfds' : True,                 # use tfds datasets as to train the model else use the given csv file
     'use_last_recorded_value' : True,
     'valid_samples_to_eval' : 100,     # number of samples used for validation
     'write_summary_op': True           # write the first batch of validation set summary to a file
     }                                    

# Special Tokens
special_tokens = {
    'input_CLS_ID' : 101,
    'input_SEP_ID' : 102,
    'MASK_ID' : 103,
    'PAD_ID' : 0,
    'target_CLS_ID' : 101,
    'target_SEP_ID' : 102,
    'UNK_ID' : 100
    }

h_parms = {
   'gradient_accumulation_steps': 2,                                                                                   
   'train_batch_size': 32,
   'beam_sizes': [2, 3, 4],              # Used  during inference                                                 
   'combined_metric_weights': [0.4, 0.6], #(bert_score, rouge)
   'dropout_rate': 0.1,
   'epochs': 1,
   'epsilon_ls': 0.2,                    # label_smoothing hyper parameter
   'grad_clipnorm':None,
   'l2_norm':0.001,
   'learning_rate': None,                # change to None to set learning rate decay
   'length_penalty' : 1,                       # Beam search hyps . Used  during inference                                                 
   'mean_attention_heads':True,                # if False then the attention parameters of the last head will be used
   'mean_attention_parameters_of_layers':True,           # if False then the attention parameters of the last layer will be used
   'validation_batch_size' : 32
   }                                    

dataset_name = training_parms['tfds_name']
core_path = 'D:\\Local_run'

file_path = {
        'best_ckpt_path' : os.path.join(core_path, f"best_checkpoints\\{dataset_name}\\"),  
        'checkpoint_path' : os.path.join(core_path, f"checkpoints\\{dataset_name}\\"),
        'initial_weights' : os.path.join(core_path, f"initial_weights\\{dataset_name}\\"),
        'infer_csv_path' : None,
        'infer_ckpt_path' : None,
        'log_path' : os.path.join(core_path, f"created_files\\{dataset_name}\\tensorflow.log"),
        'output_sequence_write_path' : os.path.join(core_path, f"created_files\\{dataset_name}\\summaries\\{dataset_name}\\"),
        'serialized_tensor_path' : os.path.join("/content/drive/My Drive/", 'saved_serialized_tensor_3'),
        'tensorboard_log' : os.path.join(core_path, f"created_files\\{dataset_name}\\tensorboard_logs/"),
        'tfds_data_dir' : 'D:\\Local_run\\Tensorflow_datasets\\en_tam_parallel_text_dataset',
        'tfds_data_version' : None,#{"version": "2.0.0"},
        'train_csv_path' : None,
        'input_seq_vocab_path' : 'D:\\Local_run\\TFDS_vocab_files\\vocab_en',
        'output_seq_vocab_path' : 'D:\\Local_run\\TFDS_vocab_files\\vocab_ta',
            }
if not training_parms['use_tfds']:
    training_parms['num_examples_to_train'] = None     # If None then all the examples in the dataset will be used to train
    training_parms['num_examples_to_infer'] = None
    training_parms['test_size'] = 0.05                 # test set split size

if not model_parms['use_BERT']:
    model_parms['use_refine_decoder'] = False
    model_parms['num_of_decoders'] = 1
    model_parms['input_pretrained_bert_model'] = None
    model_parms['target_pretrained_bert_model'] = None
    special_tokens['input_CLS_ID'] = model_parms['input_vocab_size']-2
    special_tokens['input_SEP_ID'] = model_parms['input_vocab_size']-2+1
    special_tokens['target_CLS_ID'] = model_parms['target_vocab_size']-2
    special_tokens['target_SEP_ID'] = model_parms['target_vocab_size']-2+1
else:
    model_parms['use_refine_decoder'] = True
    model_parms['num_of_decoders'] = 2

config = Bunch(model_parms)
config.update(unit_test)
config.update(training_parms)
config.update(special_tokens)
config.update(h_parms)
config.update(file_path)


if config.test_script:
    if config.unit_test_dataset_batch_size < config.gradient_accumulation_steps:
        config.gradient_accumulation_steps =  config.unit_test_dataset_batch_size
    if not config.check_model_capacity:
        config.d_model = 256
        config.dff = 512
        config.num_heads = 8
        config.num_layers = 2
    config.epochs = 1000
    config.dropout_rate = config.epsilon_ls = config.l2_norm = 0.0
    config.grad_clipnorm = None
    config.run_tensorboard = False


else:
    config.samples_to_test = -1
    config.save_initial_weights = False
    config.run_init_eval = False
    config.init_loss_check = False
    config.input_independent_baseline_check = False
    config.check_model_capacity = False
    config.random_results_check = False
    config.print_config = True
