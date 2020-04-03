#code adapted from a) https://github.com/ShenakhtPajouh/GPT-language-model-tf.keras/blob/master/utils.py
#                  b)https://github.com/raufer/bert-summarization/tree/master/models
import tensorflow as tf
tf.random.set_seed(100)
import tensorflow_addons as tfa
from configuration import config
from creates import log
from beam_search import beam_search
from transformer import create_masks


def with_column(x, i, column):
    """
    Given a tensor `x`, change its i-th column with `column`
    x :: (N, T)
    return :: (N, T)
    """
    left = x[:, :i]
    right = x[:, i+1:]
        
    return tf.concat([left, column, right], axis=1)

def mask_timestamp(x, i, mask_with):
    """
    Masks each word in the output_sequence draft one by one with the [MASK] token
    At t-th time step the t-th word of input output_sequence is
    masked, and the decoder predicts the refined word given other
    words of the output_sequence.
    
    x :: (N, T)
    return :: (N, T)
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]

    left = x[:, :i]
    right = x[:, i+1:]
    
    mask = tf.ones([N, 1], dtype=x.dtype) * mask_with
    
    masked = tf.concat([left, mask, right], axis=1)

    return masked

def create_padding_mask(seq):
    """
    Mask all the pad tokens in the batch of sequence.
    It ensures that the model does not treat padding as the input.
    The mask indicates where pad value 0 is present:
    it outputs a 1 at those locations, and a 0 otherwise.    
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

    # add extra dimensions so that we can add the padding
    # to the attention logits.
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

def sampling(logits):
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample

def top_k_sampling(logits, k=25):
    'k must be greater than 0'
    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)
    logits = tf.reshape(logits, (config.validation_batch_size, -1))
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample
  
def nucleus_sampling(logits, p=0.9):
    sorted_logits = tf.sort(logits, direction='DESCENDING')
    sorted_indices = tf.argsort(logits, direction='DESCENDING')
    cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
    t_sorted_indices_to_remove = cumulative_probs > p
    ''' Shift the indices to the right to keep also the first token above the threshold '''
    indices = tf.range(1, tf.shape(logits)[0], 1)
    sorted_indices_to_remove = tf.scatter_nd(
                                             tf.expand_dims(indices, 1), 
                                             t_sorted_indices_to_remove[:-1], 
                                             logits.shape
                                             )
    logits = tf.where(
        sorted_indices_to_remove,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits
    )
    logits = tf.reshape(logits, (config.validation_batch_size, -1))
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
    return sample

def topp_topk(logits, p, k):
  sorted_logits = tf.sort(logits, direction='DESCENDING')
  sorted_indices = tf.argsort(logits, direction='DESCENDING')
  cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits))
  t_sorted_indices_to_remove = cumulative_probs > p
  ''' Shift the indices to the right to keep also the first token above the threshold '''
  indices = tf.range(1, tf.shape(logits)[0], 1)
  sorted_indices_to_remove = tf.scatter_nd(
                                           tf.expand_dims(indices, 1), 
                                           t_sorted_indices_to_remove[:-1], 
                                           logits.shape
                                           )
  logits = tf.where(
      sorted_indices_to_remove,
      tf.ones_like(logits, dtype=logits.dtype) * -1e10,
      logits
  )
  values, _ = tf.nn.top_k(logits, k=k)
  min_value = tf.reduce_min(values)
  logits = tf.where(
      logits < min_value,
      tf.ones_like(logits, dtype=logits.dtype) * -1e10,
      logits)
  logits = tf.reshape(logits, (config.validation_batch_size, -1))
  sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
  return sample

def draft_output_sequence_sampling(model,
                           inp, 
                           enc_output, 
                           look_ahead_mask, 
                           padding_mask, 
                           decoder_type='greedy', 
                           temperature=0.9, 
                           p=0.9, 
                           k=25, 
                           training=False
                           ):
    """
    Inference call, builds a draft output_sequence auto-regressively
    """
    log.info(f"Building: 'Draft {decoder_type} decoder'")
    N = tf.shape(enc_output)[0]

    # (batch_size, 1)
    dec_input = tf.ones([N, 1], dtype=tf.int32) * config.CLS_ID
    output_sequence, dec_outputs, dec_logits, attention_dists = [], [], [], []
    output_sequence += [dec_input]
    for i in (range(0, config.target_seq_length)):
        _, _, dec_padding_mask = create_masks(inp, dec_input)
        # (batch_size, i+1, d_bert)
        embeddings = model.decoder_embedding(dec_input)    

        # (batch_size, i+1, vocab), (_)            
        dec_output, attention_dist = model.decoder(inp,
                                                   embeddings, 
                                                   enc_output, 
                                                   training, 
                                                   look_ahead_mask, 
                                                   padding_mask
                                                   )        

        # (batch_size, 1, vocab)
        dec_output_i = dec_output[:, -1: ,:]
        if decoder_type == 'nucleus':
          predictions = tf.cast(nucleus_sampling(((dec_output_i)/ temperature), p=p), tf.int32)
        elif decoder_type == 'topk':
          predictions = tf.cast(top_k_sampling(((dec_output_i)/ temperature), k=k), tf.int32)
        elif decoder_type == 'random_sampling':
          predictions = tf.cast(sampling((dec_output_i)/ temperature), tf.int32)
        elif decoder_type == 'topktopp':
          predictions = tf.cast(topp_topk(((dec_output_i)/ temperature), p=p,k=k), tf.int32)
        elif decoder_type == 'greedy':
          predictions = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
        else:
          raise RuntimeError('Incorrect Decoder type specified')
        dec_outputs += [dec_output_i]
        #dec_logits_i = dec_logits_i[:, -1:, :]
        #dec_logits += [dec_logits_i]
        output_sequence += [predictions]
        dec_input = with_column(dec_input, i+1, predictions)
    output_sequence = tf.concat(output_sequence, axis=1)  
    # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
    return output_sequence, attention_dist

def draft_output_sequence_beam_search(model,
                              input_ids, 
                              enc_output, 
                              dec_padding_mask, 
                              beam_size
                              ):

    log.info(f"Building: 'Draft beam search decoder'")
    input_ids = tfa.seq2seq.tile_batch(input_ids, multiplier=beam_size)
    enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_size)
    dec_padding_mask = tfa.seq2seq.tile_batch(dec_padding_mask, multiplier=beam_size)
    def beam_search_decoder(output):
      # (batch_size, seq_len, d_bert)    
      embeddings = model.decoder_embedding(output)
      predictions, attention_weights = model.decoder(input_ids,
                                                     embeddings, 
                                                     enc_output, 
                                                     False, 
                                                     None, 
                                                     dec_padding_mask
                                                     )
      # (batch_size, 1, target_vocab_size)
      return (predictions[:,-1:,:])
    return beam_search(
                        beam_search_decoder, 
                        [config.CLS_ID] * config.train_batch_size, 
                        beam_size, 
                        config.target_seq_length, 
                        config.input_vocab_size, 
                        config.length_penalty, 
                        stop_early=False, 
                        eos_id=[[config.SEP_ID]]
                        )
            

def refined_output_sequence_sampling(model,
                             inp, 
                             enc_output, 
                             draft_output_sequence, 
                             padding_mask, 
                             decoder_type='greedy', 
                             temperature=0.9, 
                             p=0.9, 
                             k=25,
                             training=False):
        """
        Inference call, builds a refined output_sequence
        
        It first masks each word in the output_sequence draft one by one,
        then feeds the draft to BERT to generate context vectors.
        """
        
        log.info(f"Building: 'Refined {decoder_type} decoder'")
        tf.shape(enc_output)[0]
        refined_output_sequence = draft_output_sequence
        for i in (range(1, config.target_seq_length)):
            
            # (batch_size, seq_len)
            refined_output_sequence_ = mask_timestamp(refined_output_sequence, i, config.MASK_ID)
            
            # (batch_size, seq_len, d_bert)
            context_vectors = model.decoder_bert_model(refined_output_sequence_)[0]
            
            # (batch_size, seq_len, d_bert), (_)
            dec_output,  attention_dist =  model.decoder(inp,
                                                        context_vectors,
                                                        enc_output,
                                                        training=training,
                                                        look_ahead_mask=None,
                                                        padding_mask=padding_mask
                                                      )
            
            # (batch_size, 1, vocab_len)
            dec_output_i = dec_output[:, i:i+1 ,:]
            if decoder_type == 'nucleus':
              predictions = tf.cast(nucleus_sampling((dec_output_i/ temperature), p=p), tf.int32)
            elif decoder_type == 'topk':
              predictions = tf.cast(top_k_sampling(((dec_output_i)/ temperature), k=k), tf.int32)
            elif decoder_type == 'topktopp':
              predictions = tf.cast(topp_topk(((dec_output_i)/ temperature), p=p,k=k), tf.int32)
            elif decoder_type == 'random_sampling':
              predictions = tf.cast(sampling((dec_output_i)/ temperature), tf.int32)
            elif decoder_type == 'greedy':
              predictions = tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32)
            else:
              raise RuntimeError('Incorrect decoder_type given')
            refined_output_sequence = with_column(refined_output_sequence, i, predictions)
        # (batch_size, seq_len, vocab_len), (_)        
        return refined_output_sequence, attention_dist

def predict_using_sampling(
                           model,
                           inp, 
                           draft_decoder_sampling_type='topk',
                           refine_decoder_type='topk', 
                           temperature=0.9, 
                           p=0.9, 
                           k=25):

  #restore the latest checkpoint.
  #ckpt.restore(ckpt_manager.latest_checkpoint)
  
  dec_padding_mask = create_padding_mask(inp)
  
  # (batch_size, seq_len, d_bert)
  enc_output = model.encoder_bert_model(inp)[0]
  # (batch_size, seq_len, vocab_len), (_)
  predicted_draft_output_sequence, draft_attention_dist = draft_output_sequence_sampling( model,
                                                                      inp,
                                                                      enc_output=enc_output,
                                                                      look_ahead_mask=None,
                                                                      padding_mask=dec_padding_mask,
                                                                      decoder_type=draft_decoder_sampling_type,
                                                                      temperature=temperature,
                                                                      p=p, 
                                                                      k=k,
                                                                    )
  # (batch_size, seq_len, vocab_len), ()
  predicted_refined_output_sequence, refined_attention_dist = refined_output_sequence_sampling( model,
                                                                            inp,
                                                                            enc_output=enc_output,
                                                                            padding_mask=dec_padding_mask,
                                                                            draft_output_sequence=predicted_draft_output_sequence,
                                                                            decoder_type=refine_decoder_type, 
                                                                            temperature=temperature, 
                                                                            p=p, 
                                                                            k=k
                                                                            )


  return predicted_draft_output_sequence, draft_attention_dist, predicted_refined_output_sequence, refined_attention_dist

def predict_using_beam_search(
                              model,
                              inp, 
                              beam_size=3, 
                              refine_decoder_type='nucleus', 
                              temperature=0.9, 
                              p=0.9, 
                              k=25):
  
  dec_padding_mask = create_padding_mask(inp)
  # (batch_size, seq_len, d_bert)
  enc_output = model.encoder_bert_model(inp)[0]
  
  #[batch_size*beam_size, input_Seq_len, d_bert]
  translated_output_temp = draft_output_sequence_beam_search(
                                                      model, 
                                                      inp, 
                                                      enc_output, 
                                                      dec_padding_mask, 
                                                      beam_size
                                                      )
  # Take the sequence with high score (the last one)
  predicted_draft_output_sequence = translated_output_temp[0][:,0,:] 
  
  predicted_refined_output_sequence, refined_attention_dist = refined_output_sequence_sampling(model,
                                                                          inp,
                                                                          enc_output=enc_output,
                                                                          padding_mask=dec_padding_mask,
                                                                          draft_output_sequence=predicted_draft_output_sequence, 
                                                                          decoder_type=refine_decoder_type, 
                                                                          temperature=temperature, 
                                                                          p=p, 
                                                                          k=k
                                                                          )
  return predicted_draft_output_sequence, predicted_refined_output_sequence, refined_attention_dist