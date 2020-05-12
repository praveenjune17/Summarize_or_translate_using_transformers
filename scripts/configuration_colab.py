# -*- coding: utf-8 -*-
import os
import platform
from bunch import Bunch
from check_rules import (adhere_task_rules, 
        assert_config_values, set_memory_growth)

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
     'add_bias' : True,               # set values as True|None Increases the inital bias of Tamil vocabs 
     'activation' : 'relu',
     'add_pointer_generator': True,
     'd_model': 768,                  # the projected word vector dimension
     'dff': 1024,                      # feed forward network hidden parameters
     'input_pretrained_bert_model': 'distilroberta-base',  #distilroberta-base, #bert-base-uncased , #google/electra-small-discriminator
     'input_seq_length': 50,
     'model' : 'bertified_transformer',#bertified_transformer or transformer
     'num_heads': 12,                  # the number of heads in the multi-headed attention unit
     'num_layers': 6,                 # number of transformer blocks
     'target_language' : 'ta',
     'target_pretrained_bert_model' : 'distilbert-base-multilingual-cased',#'bert-base-uncased',
                                                                     #'bert-base-multilingual-cased',
                                                                    #'distilbert-base-multilingual-cased'
     'target_seq_length': 20,
     'task':'translate'            # must be translate or summarize
     }

training_parms = {
     'accumulate_gradients' : True,
     'display_model_summary' : True,
     'early_stop' : False,
     'enable_jit' : False,
     'eval_after_steps' : 5000,              # Evaluate after these many training steps
     'gradient_accumulation_steps': 9,   
     'last_recorded_value': None,
     'min_train_loss' : 1.0,
     'monitor_metric' : 'unified_metric',
     'run_tensorboard': True,
     'samples_to_train' : -1,
     'samples_to_validate' : 100,
     'start_evaluate_when' : 5.0,           # run evaluation when loss reaches 10
     'steps_to_print_training_info': 100,      # print training progress per number of batches specified
     'tfds_name' : 'en_tam_parallel_text',            #cnn_dailymail,en_tam_parallel_text     # tfds dataset to be used
     'tolerance' : 0,
     'tolerance_threshold': 7,          # Stop training after the threshold is reached
     'tokens_per_batch' : 4050,
     'use_tfds' : True,                 # use tfds datasets as to train the model else use the given csv file
     'write_batch1_predictions': True           # write the first batch of validation set summary to a file
     }                                    

# Special Tokens
special_tokens = {
    'input_CLS_ID' : 101,                            #assumed to be BERT ids
    'input_SEP_ID' : 102,                            #assumed to be BERT ids
    'MASK_ID' : 103,
    'PAD_ID' : 0,
    }

inference_decoder_parms = {
    'beam_size': 1,              
    'draft_decoder_type' : 'greedy',     # 'greedy', 'only_beam_search', 'topktopp' --> topktopp filtering + beam search
    'length_penalty' : 1,
    'num_parallel_calls' : -1,
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
   'train_batch_size': 4,
   'validation_batch_size' : 8
   }                                    

dataset_name = training_parms['tfds_name']
model = model_parms['model']
core_path = "/content/drive/My Drive/"#os.getcwd()
path_seperator = '\\' if platform.system() == 'Windows' else '/'
file_path = {
        'best_ckpt_path' : os.path.join(core_path, f"best_checkpoints{path_seperator}{dataset_name+'_'+model}{path_seperator}"),  
        'checkpoint_path' : os.path.join(core_path, f"checkpoints{path_seperator}{dataset_name+'_'+model}{path_seperator}"),
        'initial_weights' : os.path.join(core_path, f"initial_weights{path_seperator}{dataset_name+'_'+model}{path_seperator}"),
        'infer_csv_path' : None,
        'infer_ckpt_path' : None,
        'input_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_en"),
        'log_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name+'_'+model}{path_seperator}tensorflow.log"),
        'output_seq_vocab_path' : os.path.join(core_path, f"TFDS_vocab_files{path_seperator}{dataset_name}{path_seperator}vocab_ta"),
        'output_sequence_write_path' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name+'_'+model}{path_seperator}summaries{path_seperator}"),
        'serialized_tensor_path' : os.path.join("/content/drive/My Drive/saved_serialized_tensor_ta"),
        'tensorboard_log' : os.path.join(core_path, f"created_files{path_seperator}{dataset_name+'_'+model}{path_seperator}tensorboard_logs{path_seperator}"),
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

#set GPU memory growth
_ = set_memory_growth()
config, source_tokenizer, target_tokenizer = adhere_task_rules(config)
config = assert_config_values(config)

# Special Tokens 
config['PAD_ID']  = target_tokenizer.pad_token_id
config['CLS_ID']  = target_tokenizer.cls_token_id
config['MASK_ID'] = target_tokenizer.mask_token_id
config['SEP_ID']  = target_tokenizer.sep_token_id