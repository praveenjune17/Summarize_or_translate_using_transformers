# -*- coding: utf-8 -*-
import os
import platform
from bunch import Bunch

unit_test = {
      'check_evaluation_pipeline' : False,
      'check_model_capacity' : False,
      'check_training_pipeline' : False,
      'check_predictions_shape' : False,
      'clear_log' : True,
      'detokenize_samples' : False,
      'gpu_memory_test' : False,
      'init_loss_check' : False,
      'input_independent_baseline_check' : False, 
      'print_config' : True,
      'random_results_check' : False,
      'samples_to_test' : 128,
      'save_initial_weights' : False,
      'test_script' : False,
      'unit_test_dataset_batch_size' : 1
          }


model_parms = {
     'add_bias' : None,               # set values as True|None Increases the inital bias of Tamil vocabs
     'activation' : 'relu',
     'add_pointer_generator': True,
     'bert_score_model' : 'bert-base-multilingual-cased',
     'd_model': 256,                  # the projected word vector dimension
     'dff': 1024,                      # feed forward network hidden parameters
     'input_pretrained_bert_model': 'bert-base-uncased',
     'input_seq_length': 100,
     'input_vocab_size': 8247+2,        # total vocab size + start and end token
     'model_architecture' : 'transformer',   #bertified_transformer or transformer
     'num_heads': 4,                  # the number of heads in the multi-headed attention unit
     'num_layers': 4,                 # number of transformer blocks
     'target_pretrained_bert_model' : 'bert-base-multilingual-cased',
     'target_seq_length': 100,
     'target_vocab_size': 8294+2,
     'task':'translate'            # must be translate or summarize
     }

training_parms = {
     'accumulate_gradients' : True,
     'display_model_summary' : True,
     'early_stop' : True,
     'enable_jit' : True,
     'eval_after_steps' : 1000,              # Evaluate after these many training steps
     'gradient_accumulation_steps': 2,   
     'last_recorded_value': 0.6259,
     'min_train_loss' : 1.0,
     'monitor_metric' : 'combined_metric',
     'run_tensorboard': True,
     'steps_to_print_training_info': 100,      # print training progress per number of batches specified
     'tfds_name' : 'en_tam_parallel_text',     # tfds dataset to be used
     'tolerance' : 0,
     'tolerance_threshold': 3,          # Stop training after the threshold is reached
     'tokens_per_batch' : 4050,
     'use_tfds' : True,                 # use tfds datasets as to train the model else use the given csv file
     'valid_samples_to_eval' : 10000,     # number of samples used for validation
     'write_summary_op': True           # write the first batch of validation set summary to a file
     }                                    

# Special Tokens
special_tokens = {
    'input_CLS_ID' : 101,                   # Please donot change these
    'input_SEP_ID' : 102,
    'MASK_ID' : 103,
    'PAD_ID' : 0,
    'target_CLS_ID' : 101,
    'target_SEP_ID' : 102
    }

inference_decoder_parms = {
    'decoder_type' : 'beam_search',   # or topktopp
    'softmax_temperature'  : 0.9, 
    'topp' : 0.9, 
    'topk' : 5
    }    
h_parms = {
   'beam_size': 1,              # Used  during inference                                                 
   'combined_metric_weights': [0.8, 0.1, 0.1], #(bert_score, rouge, bleu)
   'dropout_rate': 0.1,
   'epochs': 4,
   'epsilon_ls': 0.1,                    # label_smoothing hyper parameter
   'grad_clipnorm':None,
   'l2_norm':0.0,
   'learning_rate': None,                # change to None to set learning rate decay
   'length_penalty' : 1,               # Beam search hyps . Used  during inference                                                 
   'train_batch_size': 128,
   'validation_batch_size' : 32
   }                                    

dataset_name = training_parms['tfds_name']
core_path = os.getcwd()
path_seperator = '\\' if platform.system() == 'Windows' else '/'
file_path = {
        'best_ckpt_path' : os.path.join(core_path, f"best_checkpoints{path_seperator}{dataset_name}{path_seperator}"),  
        'checkpoint_path' : os.path.join(core_path, f"checkpoints{path_seperator}{dataset_name}{path_seperator}"),
        'initial_weights' : os.path.join(core_path, f"initial_weights{path_seperator}{dataset_name}{path_seperator}"),
        'infer_csv_path' : None,
        'infer_ckpt_path' : 'D:\\Local_run\\best_checkpoints\\en_tam_parallel_text\\ckpt-213',
        'input_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_en"),
        'log_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}tensorflow.log"),
        'output_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_ta"),
        'output_sequence_write_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}summaries{path_seperator}"),
        'serialized_tensor_path' : os.path.join("/content/drive/My Drive/", 'saved_serialized_tensor_3'),
        'tensorboard_log' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}tensorboard_logs{path_seperator}"),
        'tfds_data_dir' : os.path.join(core_path, f'Tensorflow_datasets{path_seperator}{dataset_name}_dataset'),
        'tfds_data_version' : None,
        'train_csv_path' : None
            }

if not training_parms['use_tfds']:
    training_parms['num_examples_to_train'] = None     # If None then all the examples in the dataset will be used to train
    training_parms['num_examples_to_infer'] = None
    training_parms['test_size'] = 0.05                 # test set split size
if training_parms['accumulate_gradients']==False:
    training_parms['gradient_accumulation_steps'] = 1
if training_parms['gradient_accumulation_steps'] == 1:
    training_parms['accumulate_gradients']==False

if inference_decoder_parms['decoder_type'] == 'greedy':
    inference_decoder_parms['decoder_type'] = 'beam_search'
    h_parms['beam_size'] = 1

if not (model_parms['model_architecture']=='bertified_transformer'):
    model_parms['num_of_decoders'] = 1
    special_tokens['input_CLS_ID'] = model_parms['input_vocab_size']-2
    special_tokens['input_SEP_ID'] = model_parms['input_vocab_size']-2+1
    special_tokens['target_CLS_ID'] = model_parms['target_vocab_size']-2
    special_tokens['target_SEP_ID'] = model_parms['target_vocab_size']-2+1
else:
    model_parms['num_of_decoders'] = 2

if model_parms['add_bias']:
    read_tensor = tf.io.read_file(file_path.serialized_tensor_path, name=None)
    output_bias_tensor = tf.io.parse_tensor(read_tensor, tf.float32, name=None)
    model_parms['add_bias'] = tf.keras.initializers.Constant(output_bias_tensor.numpy())

config = Bunch(model_parms)
config.update(unit_test)
config.update(training_parms)
config.update(special_tokens)
config.update(inference_decoder_parms)
config.update(h_parms)
config.update(file_path)


if config.test_script:
    if config.unit_test_dataset_batch_size < config.gradient_accumulation_steps:
        config.gradient_accumulation_steps =  config.unit_test_dataset_batch_size
    if config.unit_test_dataset_batch_size > config.samples_to_test:
        config.unit_test_dataset_batch_size = config.samples_to_test
    if config.steps_to_print_training_info > config.unit_test_dataset_batch_size:
        config.steps_to_print_training_info = config.unit_test_dataset_batch_size
    config.grad_clipnorm = None
    config.run_tensorboard = False
    config.dropout_rate = config.epsilon_ls = config.l2_norm = 0

else:
    config.samples_to_test = -1
    config.save_initial_weights = False
    config.run_init_eval = False
    config.init_loss_check = False
    config.input_independent_baseline_check = False
    config.check_model_capacity = False
    config.random_results_check = False
    config.print_config = True
    config.clear_log = False
