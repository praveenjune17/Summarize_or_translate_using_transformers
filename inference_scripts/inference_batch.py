# -*- coding: utf-8 -*-
import tensorflow as tf
tf.random.set_seed(100)
import time
import os
import numpy as np
from transformer import create_masks
from configuration import config
from create_model import Model, tokenizer
from beam_search import beam_search
from preprocess import infer_data_from_df
from calculate_metrics import convert_wordpiece_to_words
from rouge import Rouge
from bert_score import score as b_score

rouge_all = Rouge()
infer_template = '''Beam size <--- {}\
                    ROUGE-f1  <--- {}\
                    BERT-f1   <--- {}'''

def with_column(x, i, column):
    """
    Given a tensor `x`, change its i-th column with `column`
    x :: (N, T)
    return :: (N, T)
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    left = x[:, :i]
    right = x[:, i+1:]
        
    return tf.concat([left, column, right], axis=1)

def mask_timestamp(x, i, mask_with):
    """
    Masks each word in the summary draft one by one with the [MASK] token
    At t-th time step the t-th word of input summary is
    masked, and the decoder predicts the refined word given other
    words of the summary.
    
    x :: (N, T)
    return :: (N, T)
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    left = x[:, :i]
    right = x[:, i+1:]
    
    mask = tf.ones([N, 1], dtype=x.dtype) * mask_with
    
    masked = tf.concat([left, mask, right], axis=1)

    return masked

def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               Model=Model
                               )
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'Incorrect checkpoint direcotry'
    ckpt.restore(checkpoint_path).expect_partial()
    print(f'{checkpoint_path} restored')

#@tf.function
def draft_decoded_summary(model, input_ids, target_ids, beam_size):
    batch = tf.shape(input_ids)[0]
    start = [101] * batch
    end = [102]
    # (batch_size, seq_len, d_bert)
    enc_output_ = model.bert_model(input_ids)[0]
    enc_output = tf.tile(enc_output_, multiples=[beam_size,1, 1])
    input_ids = tf.tile(input_ids, multiples=[beam_size, 1])
    # (batch_size, 1, 1, seq_len), (_), (batch_size, 1, 1, seq_len)
    def beam_search_decoder(target_ids):
      _, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids)    
      draft_logits, _ = model.draft_summary(
                                            input_ids=input_ids,
                                            enc_output=enc_output,
                                            look_ahead_mask=combined_mask,
                                            padding_mask=dec_padding_mask,
                                            target_ids=target_ids,
                                            training=False
                                          )
      # (batch_size, 1, target_vocab_size)
      return (draft_logits[:,-1:,:])
    return (beam_search(
                    beam_search_decoder, 
                    start, 
                    beam_size, 
                    config.target_seq_length, 
                    config.input_vocab_size, 
                    config.length_penalty, 
                    stop_early=True, 
                    eos_id=[end]
                    ),
            enc_output_)

def refined_summary_greedy(model, input_ids, enc_output, draft_summary, padding_mask, training=False):
        """
        Inference call, builds a refined summary
        
        It first masks each word in the summary draft one by one,
        then feeds the draft to BERT to generate context vectors.
        """                
        refined_summary = draft_summary
        refined_summary_mask = tf.cast(tf.math.equal(draft_summary, 0), tf.float32)
        refined_summary_segment_ids = tf.zeros(tf.shape(draft_summary))
                
        N = tf.shape(draft_summary)[0]            
        T = tf.shape(draft_summary)[1]
        
        dec_outputs, attention_dists = [], []
        for i in range(1, model.output_seq_len):
            
            # (batch_size, seq_len)
            refined_summary_ = mask_timestamp(refined_summary, i, config.MASK_ID)
            
            # (batch_size, seq_len, d_bert)
            context_vectors = model.bert_model(refined_summary_)[0]
            
            # (batch_size, seq_len, vocab_len), (_)
            dec_output, attention_dist = model.decoder(
                                                        input_ids,
                                                        context_vectors,
                                                        enc_output,
                                                        training=training,
                                                        look_ahead_mask=None,
                                                        padding_mask=padding_mask
                                                      )
            
            # (batch_size, 1, vocab_len)
            dec_output_i = dec_output[:, i:i+1 ,:]
            dec_outputs += [dec_output_i]
            # (batch_size, 1) 
            preds = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)

            if tf.squeeze(preds, axis=0) == 102:
              refined_summary = with_column(refined_summary, i, preds)
              return refined_summary, attention_dist       
            
            # (batch_size, seq_len)
            refined_summary = with_column(refined_summary, i, preds)
        # (batch_size, seq_len), (_)        
        return refined_summary, attention_dist

def run_inference(model, dataset, beam_sizes_to_try=config.beam_sizes):
    for beam_size in beam_sizes_to_try:
      ref_sents = []
      hyp_sents = []
      for (doc_id, (input_ids, _, _, target_ids, _, _)) in enumerate(dataset, 1):
        start_time = time.time()
        # translated_output_temp[0] (batch, beam_size, summ_length+1)
        translated_output_temp, enc_output = draft_decoded_summary(model, input_ids, target_ids[:, :-1], beam_size)
        draft_predictions = translated_output_temp[0][:,0,:]
        _, _, dec_padding_mask = create_masks(input_ids, target_ids[:, :-1])
        refined_summary, attention_dists = refined_summary_greedy(model, input_ids, enc_output, draft_predictions, dec_padding_mask, training=False)
        sum_ref = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(target_ids) if i not in [0, 101, 102]])
        sum_hyp = tokenizer.convert_ids_to_tokens([i for i in tf.squeeze(refined_summary) if i not in [0, 101, 102]])
        sum_ref = convert_wordpiece_to_words(sum_ref)
        sum_hyp = convert_wordpiece_to_words(sum_hyp)
        print('Original summary: {}'.format(sum_ref))
        print('Predicted summary: {}'.format(sum_hyp))
        if sum_ref and sum_hyp:
          ref_sents.append(sum_ref)
          hyp_sents.append(sum_hyp)
      try:
        rouges = rouge_all.get_scores(ref_sents , hyp_sents)
        avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], rouge_scores['rouge-2']["f"], rouge_scores['rouge-l']["f"]]) for rouge_scores in rouges])
        _, _, bert_f1 = b_score(ref_sents, hyp_sents, lang='en', model_type=config.pretrained_bert_model)
        avg_bert_f1 = np.mean(bert_f1.numpy())
      except:
        avg_rouge_f1 = 0
        avg_bert_f1 = 0
      print(infer_template.format(beam_size, avg_rouge_f1, avg_bert_f1))
      print(f'time to process document {doc_id} : {time.time()-start_time}') 

if __name__ == '__main__':
  #Restore the model's checkpoints
  restore_chkpt(config.infer_ckpt_path)
  infer_dataset = infer_data_from_df()
  run_inference(Model, infer_dataset)
