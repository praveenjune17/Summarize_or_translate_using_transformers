# -*- coding: utf-8 -*-
import os
import platform
from bunch import Bunch
from check_rules import adhere_task_rules, assert_config_values

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
      'samples_to_test' : 1,
      'save_initial_weights' : False,
      'test_script' : False,
      'unit_test_dataset_batch_size' : 1
          }

model_parms = {
     'add_bias' : None,               # set values as True|None Increases the inital bias of Tamil vocabs 
     'activation' : 'relu',
     'add_pointer_generator': True,
     'd_model': 256,                  # the projected word vector dimension
     'dff': 1024,                      # feed forward network hidden parameters
     'input_pretrained_bert_model': 'bert-base-uncased',
     'input_seq_length': 300,
     'model' : 'transformer',#bertified_transformer or transformer
     'num_heads': 4,                  # the number of heads in the multi-headed attention unit
     'num_layers': 4,                 # number of transformer blocks
     'target_language' : 'ta',
     'target_pretrained_bert_model' : 'bert-base-uncased',#'bert-base-uncased',#'bert-base-multilingual-cased',
     'target_seq_length': 50,
     'task':'translate'            # must be translate or summarize
     }

training_parms = {
     'accumulate_gradients' : True,
     'samples_to_train' : -1,
     'display_model_summary' : True,
     'early_stop' : True,
     'enable_jit' : True,
     'eval_after_steps' : 1000,              # Evaluate after these many training steps
     'gradient_accumulation_steps': 2,   
     'last_recorded_value': 0.6259,
     'min_train_loss' : 1.0,
     'monitor_metric' : 'unified_metric',
     'run_tensorboard': True,
     'steps_to_print_training_info': 100,      # print training progress per number of batches specified
     'tfds_name' : 'en_tam_parallel_text',            #cnn_dailymail,en_tam_parallel_text     # tfds dataset to be used
     'tolerance' : 0,
     'tolerance_threshold': 3,          # Stop training after the threshold is reached
     'tokens_per_batch' : 4050,
     'use_tfds' : True,                 # use tfds datasets as to train the model else use the given csv file
     'write_batch1_predictions': True           # write the first batch of validation set summary to a file
     }                                    

# Special Tokens
special_tokens = {
    'CLS_ID' : 101,
    'SEP_ID' : 102,
    'MASK_ID' : 103,
    'PAD_ID' : 0,
    }

inference_decoder_parms = {
    'beam_size': 7,              
    'draft_decoder_type' : 'only_beam_search',     # 'greedy', 'only_beam_search', 'topktopp' --> topktopp filtering + beam search
    'length_penalty' : 1,
    'refine_decoder_type' : 'greedy',
    'softmax_temperature' : 1,
    'top_p' : 0.9, 
    'top_k' : 5                         
    }

h_parms = {
   'metric_weights': {'bert_f1_score':0.8, 'task_score':0.2}, #(task_score <- rouge if summarize else bleu)
   'dropout_rate': 0.1,
   'epochs': 4,
   'epsilon_ls': 0.1,                  # label_smoothing hyper parameter
   'grad_clipnorm':None,
   'l2_norm':0.0,
   'learning_rate': None,              # set None to create decayed learning rate schedule
   'train_batch_size': 64,
   'validation_batch_size' : 20
   }                                    

dataset_name = training_parms['tfds_name']
core_path = os.getcwd()
path_seperator = '\\' if platform.system() == 'Windows' else '/'
file_path = {
        'best_ckpt_path' : os.path.join(core_path, f"best_checkpoints{path_seperator}{dataset_name}{path_seperator}"),  
        'checkpoint_path' : os.path.join(core_path, f"temp{path_seperator}checkpoints{path_seperator}{dataset_name}{path_seperator}"),
        #'checkpoint_path' : 'D:\\Local_run\\checkpoints\\temp_ckpt',
        'initial_weights' : os.path.join(core_path, f"initial_weights{path_seperator}{dataset_name}{path_seperator}"),
        'infer_csv_path' : None,
        'infer_ckpt_path' : 'D:\\Local_run\\best_checkpoints\\en_tam_parallel_text\\ckpt-213',
        #'input_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_en"),
        'input_seq_vocab_path' : '/root/ckpt_dir/vocab_en',
        'log_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}tensorflow.log"),
        #'output_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_ta"),
        'output_seq_vocab_path' : '/root/ckpt_dir/vocab_ta',
        'output_sequence_write_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}summaries{path_seperator}"),
        #'output_sequence_write_path' : '/root/ckpt_dir/vocab_ta',
        'serialized_tensor_path' : os.path.join("/content/drive/My Drive/", 'saved_serialized_tensor_'+ model_parms['target_language']),
        'tensorboard_log' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name}{path_seperator}tensorboard_logs{path_seperator}"),
        'tfds_data_dir' : os.path.join(core_path, f'Tensorflow_datasets{path_seperator}{dataset_name}_dataset'),
        'tfds_data_version' : None,
        'train_csv_path' : None
            }

config = Bunch(model_parms)
config.update(unit_test)
config.update(training_parms)
config.update(special_tokens)
config.update(inference_decoder_parms)
config.update(h_parms)
config.update(file_path)

config, source_tokenizer, target_tokenizer = adhere_task_rules(config)
config = assert_config_values(config)