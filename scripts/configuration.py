# -*- coding: utf-8 -*-
import os
import platform
from bunch import Bunch

unit_test = {
      'check_evaluation_pipeline' : False,
      'check_model_capacity' : True,
      'check_training_pipeline' : False,
      'check_predictions_shape' : True,
      'detokenize_samples' : False,
      'gpu_memory_test' : True,
      'init_loss_check' : False,
      'input_independent_baseline_check' : False, 
      'print_config' : True,
      'random_results_check' : False,
      'samples_to_test' : -1,
      'save_initial_weights' : False,
      'test_script' : True,
      'unit_test_dataset_batch_size' : 32
          }


model_parms = {
     'add_bias' : None,               # set values as True|None Increases the inital bias of Tamil vocabs
     'activation' : 'relu',
     'bert_score_model' : 'bert-base-multilingual-cased',
     'add_pointer_generator': True,
     'input_seq_length': 60,
     'd_model': 768,                  # the projected word vector dimension
     'dff': 2048,                      # feed forward network hidden parameters
     'input_vocab_size': 8247+2,        # total vocab size + start and end token
     'num_heads': 8,                  # the number of heads in the multi-headed attention unit
     'num_layers': 8,                 # number of transformer blocks
     'input_pretrained_bert_model': 'bert-base-uncased',
     'model_architecture' : 'transformer',   #bertified_transformer or transformer
     'task':'translation',
     'target_pretrained_bert_model' : 'bert-base-multilingual-cased',
     'target_seq_length': 40,
     'target_vocab_size': 8294+2
     }   

training_parms = {
     'accmulate_gradients' : True,
     'display_model_summary' : True,
     'early_stop' : True,
     'enable_jit' : True,
     'eval_steps' : 5000,              # Evaluate once this many samples are trained
     'gradient_accumulation_steps': 2,   
     'last_recorded_value': None,
     'min_train_loss' : 1.0,
     'monitor_metric' : 'combined_metric',
     'run_tensorboard': True,
     'steps_to_print_training_info': 50,                  # print training progress per number of batches specified
     'tfds_name' : 'en_tam_parallel_text',     # tfds dataset to be used
     'tolerance' : 0,
     'tolerance_threshold': 3,          # Stop training after the threshold is reached
     'tokens_per_batch' : 3200,
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

inference_decoder_parms = {
    'decoder_type' : 'beam_search',   # or topktopp
    'softmax_temperature'  : 0.9, 
    'topp' : 0.9, 
    'topk' : 5
    }    
h_parms = {
   'train_batch_size': 32,
   'beam_size': 1,              # Used  during inference                                                 
   'combined_metric_weights': [0.98, 0.02], #(bert_score, rouge)
   'dropout_rate': 0.1,
   'epochs': 7,
   'epsilon_ls': 0.2,                    # label_smoothing hyper parameter
   'grad_clipnorm':None,
   'l2_norm':0.0,
   'learning_rate': None,                # change to None to set learning rate decay
   'length_penalty' : 1,                       # Beam search hyps . Used  during inference                                                 
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
        'infer_ckpt_path' : None,
        'log_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}tensorflow.log"),
        'output_sequence_write_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}\
                                                                 summaries{path_seperator}"),
        'serialized_tensor_path' : os.path.join("/content/drive/My Drive/", 'saved_serialized_tensor_3'),
        'tensorboard_log' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}tensorboard_logs/"),
        'tfds_data_dir' : os.path.join(core_path, f'Tensorflow_datasets{path_seperator}{dataset_name}_dataset'),
        'tfds_data_version' : None,
        'train_csv_path' : None,
        'input_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_en"),
        'output_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_ta"),
            }

if not training_parms['use_tfds']:
    training_parms['num_examples_to_train'] = None     # If None then all the examples in the dataset will be used to train
    training_parms['num_examples_to_infer'] = None
    training_parms['test_size'] = 0.05                 # test set split size
if training_parms['accmulate_gradients']==False:
    training_parms['gradient_accumulation_steps'] = 1

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
    if not config.check_model_capacity:
        config.d_model = 256
        config.dff = 512
        config.num_heads = 8
        config.num_layers = 2
    config.epochs = 10000000
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
