# -*- coding: utf-8 -*-
import os
from bunch import Bunch

core_path = os.getcwd() 
dataset_name = 'cnn'

model_parms = {
     'copy_gen':True,
     'input_seq_length': 512,
     'd_model': 768,                  # the projected word vector dimension
     'dff': 2048,                      # feed forward network hidden parameters
     'input_vocab_size': 30522,        # total vocab size + start and end token
     'num_heads': 8,                  # the number of heads in the multi-headed attention unit
     'num_layers': 8,                 # number of transformer blocks
     'pretrained_bert_model': 'bert-base-uncased',
     'target_seq_length': 72,
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

# Special Tokens
special_tokens = {
          'CLS_ID' : 101,
          'MASK_ID' : 103,
          'PAD_ID' : 0,
          'SEP_ID' : 102,
          'UNK_ID' : 100
          }


h_parms = {
   'accumulation_steps': 36,                                                                                   
   'train_batch_size': 1,
   'beam_sizes': [2, 3, 4],              # Used only during inference                                                 
   'combined_metric_weights': [0.4, 0.3, 0.3], #(bert_score, rouge, validation accuracy)
   'dropout_rate': 0.0,
   'epochs': 4,
   'epsilon_ls': 0.0,                    # label_smoothing hyper parameter
   'grad_clipnorm':None,
   'l2_norm':0,
   'learning_rate': None,                # change to None to set learning rate decay
   'length_penalty' : 1,                       # Beam search hyps . Used only during inference                                                 
   'mean_attention_heads':True,                # if False then the attention parameters of the last head will be used
   'mean_parameters_of_layers':True,           # if False then the attention parameters of the last layer will be used
   'validation_batch_size' : 8
   }                                    


file_path = {
        'best_ckpt_path' : "/content/drive/My Drive/Text_summarization/BERT_text_summarisation/created_files/training_summarization_model_ckpts/cnn/best_checkpoints",  
        'checkpoint_path' : "/content/cnn_checkpoints",
        'infer_csv_path' : None,
        'infer_ckpt_path' : "/content/drive/My Drive/Text_summarization/BERT_text_summarisation/cnn_checkpoints/ckpt-1",
        'log_path' : "/content/drive/My Drive/Text_summarization/BERT_text_summarisation/created_files/tensorflow.log",
        'subword_vocab_path' : os.path.join(core_path, "input_files/vocab_file_summarization_"+dataset_name),
        'output_sequence_write_path' : os.path.join(core_path, "created_files/summaries/"+dataset_name+"/"),
        'tensorboard_log' : os.path.join(core_path, "created_files/tensorboard_logs/"+dataset_name+"/"),
        'tfds_data_dir' : '/content/drive/My Drive/Text_summarization/cnn_dataset',
        'train_csv_path' : None,
        
    }

config = Bunch(model_parms)
config.update(training_parms)
config.update(special_tokens)
config.update(h_parms)
config.update(file_path)
