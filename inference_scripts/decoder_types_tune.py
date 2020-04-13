# -*- coding: utf-8 -*-
'''
Try out combinations of different decoders on the test set
'''
import tensorflow as tf
tf.random.set_seed(100)
import tensorflow_datasets as tfds
import numpy as np
import os
from rouge import Rouge
from bert_score import score as b_score
from scripts.model import tokenizer
from scripts.preprocess import map_batch_shuffle
from scripts.configuration import config
from scripts.calculate_metrics import convert_wordpiece_to_words
from scripts.decode_text import *


rouge_all = Rouge()
infer_template = '''Draft_decoder_type <--- {}\nRefine_decoder_type <--- {}\nROUGE-f1  <--- {}\nBERT-f1   <--- {}'''



def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               model=model
                               )
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'Incorrect checkpoint directory'
    ckpt.restore(checkpoint_path).expect_partial()
    print(f'{checkpoint_path} restored')

draft_and_refine_decoder_combinations = [
                                         ('greedy', 'greedy'), 
                                         ('greedy', 'topk'), 
                                         ('topk', 'greedy'), 
                                         ('topk', 'topk') ,
                                         ('greedy', 'nucleus') ,
                                         ('nucleus', 'greedy') ,
                                         ('topk', 'nucleus') ,
                                         ('nucleus', 'topk') ,
                                         ('nucleus', 'nucleus') ,
                                         ('beam_search', 'greedy') ,
                                         ('beam_search', 'topk') ,
                                         ('beam_search', 'nucleus') ,
                                          ]

# Beam size is set to 3 by default
# Other hyperparameters include temperature, p (nucleus sampling) and k (top k sampling)
# Please refer decode_text script
def run_inference(dataset, print_output=False):

  for draft_type, refine_type in draft_and_refine_decoder_combinations:
    ref_sents = []
    hyp_sents = []
    for (input_id, (input_ids, _, _, target_ids, _, _)) in enumerate(dataset, 1):
      start_time = time.time()
      if draft_type != 'beam_search':
          _, _, refined_output_sequence, _ = predict_using_sampling(input_ids, draft_type, refine_type, k=10)
      else:
          _, refined_output_sequence, _ = predict_using_beam_search(input_ids, refine_decoder_sampling_type=refine_type)
      sum_ref = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(target_ids) if i not in [config.PAD_ID, 
                                                                                                config.target_CLS_ID, 
                                                                                                config.target_SEP_ID]])
      sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(refined_output_sequence) if i not in [config.PAD_ID, 
                                                                                                             config.target_CLS_ID, 
                                                                                                             config.target_SEP_ID]])
      sum_ref = convert_wordpiece_to_words(sum_ref)
      sum_hyp = convert_wordpiece_to_words(sum_hyp)
      if print_output:
        print('Original output_sequence: {}'.format(sum_ref))
        print('Predicted output_sequence: {}'.format(sum_hyp))
      ref_sents.append(sum_ref)
      hyp_sents.append(sum_hyp)
    print(f'Calculating scores for {len(ref_sents)} true output_sequences and {len(hyp_sents)} predicted output_sequences')
    try:
      rouges = rouge_all.get_scores(ref_sents , hyp_sents)
      avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], 
                                       rouge_scores['rouge-2']["f"], 
                                       rouge_scores['rouge-l']["f"]]) for rouge_scores in rouges])
      _, _, bert_f1 = b_score(ref_sents, hyp_sents, lang='en', model_type=config.pretrained_bert_model)
      avg_bert_f1 = np.mean(bert_f1.numpy())
    except:
      avg_rouge_f1 = 0
      avg_bert_f1 = 0
    print(infer_template.format(draft_type, refine_type, avg_rouge_f1, avg_bert_f1))
    print(f'time to process input_sequence {input_id} : {time.time()-start_time}') 

if __name__ == '__main__':
  #Restore the model's checkpoints
  restore_chkpt(config.infer_ckpt_path)
  if config.use_tfds:
    test_dataset = create_dataset('test', True, False, '0', '2', 11490, None, None, config.validation_batch_size)
  else:
    test_dataset = create_dataset('test', False, True, None, None, None, config.infer_ckpt_path, config.num_examples_to_infer, config.validation_batch_size)
  run_inference(test_dataset)
