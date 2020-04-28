import tensorflow as tf
import numpy as np
tf.random.set_seed(100)
import tensorflow_addons as tfa
from configuration import config
from utilities import log
from tensor2tensor.utils.beam_search import beam_search

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
    N, _ = tf.shape(x)[0], tf.shape(x)[1]
    mask = tf.ones([N, 1], dtype=x.dtype) * mask_with
    masked = with_column(x, i, mask)
    N, T = tf.shape(x)[0], tf.shape(x)[1]
    left = x[:, :i]
    right = x[:, i+1:]
    mask = tf.ones([N, 1], dtype=x.dtype) * mask_with
    masked = tf.concat([left, mask, right], axis=1)
    return masked
  
def tile_and_mask_diagonal(x, mask_with):
    """    
    Masks each word in the summary draft one by one with the [MASK] token
    At t-th time step the t-th word of input summary is
    masked, and the decoder predicts the refined word given other
    words of the summary.
    
    x :: (N, T)
    returrn :: (N, T-1, T)
    
    We do not mask the first and last postition (corresponding to [CLS]
    """

    N, T = tf.shape(x)[0], tf.shape(x)[1]
    first = tf.reshape(tf.tile(x[:, 0], [T-1]), [N, T-1, 1])    
    x = x[:, 1:]
    T = T - 1    
    masked = tf.reshape(tf.tile(x, [1, T]), [N, T, T])    
    diag = tf.ones([N, T], dtype=masked.dtype) * mask_with
    masked = tf.linalg.set_diag(masked, diag)    
    masked = tf.concat([first, masked], axis=2)    
    masked = tf.reshape(masked, [N*T, T+1])

    return masked

def get_angles(pos, i, d_model):
    '''Get angle rate for the projected embedding output (d_model)
       and multiply that with the target vocab size
    '''
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))

    return pos * angle_rates

def positional_encoding(position, d_model):

    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def create_padding_mask(seq):
    '''The mask indicates where pad value 0 is present.
       it outputs a 1 at those locations, and a 0 otherwise.
    '''
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # add extra dimensions so that we can add the padding
    # to the attention logits.

    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]  

def create_look_ahead_mask(size):
    '''look-ahead mask is used to mask the future tokens in a sequence
       i.e to predict the third word, only the first and second word will be used
    '''
              #lower_triangular_matrix
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)  
    # (seq_len, seq_len)
    return mask  

def create_masks(input_ids, target_ids):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(input_ids)  
    dec_padding_mask = create_padding_mask(input_ids)
    dec_target_padding_mask = create_padding_mask(target_ids)
    look_ahead_mask = create_look_ahead_mask(tf.shape(target_ids)[1])
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

def sampling(logits):

    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)

    return sample

def top_k_sampling(logits, batch_size, k=25):
    'k must be greater than 0'
    values, _ = tf.nn.top_k(logits, k=k)
    min_value = tf.reduce_min(values)
    logits = tf.where(
        logits < min_value,
        tf.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits)
    logits = tf.reshape(logits, (batch_size, -1))
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)

    return sample
  
def nucleus_sampling(logits, batch_size, p=0.9):

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
    logits = tf.reshape(logits, (batch_size, -1))
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)

    return sample

def topp_topk(logits, batch_size, p, k):
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
    logits = tf.reshape(logits, (batch_size, -1))
    sample = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)

    return sample

def sampling_decoder(decoder_type, decoder_op, batch_size, temperature, p, k):

    if decoder_type == 'nucleus':
        predictions = tf.cast(nucleus_sampling((decoder_op/ temperature), batch_size, p=p), tf.int32)
    elif decoder_type == 'topk':
        predictions = tf.cast(top_k_sampling(((decoder_op)/ temperature), batch_size, k=k), tf.int32)
    elif decoder_type == 'topktopp':
        predictions = tf.cast(topp_topk(((decoder_op)/ temperature), batch_size, p=p, k=k), tf.int32)
    elif decoder_type == 'random_sampling':
        predictions = tf.cast(sampling(decoder_op/ temperature), tf.int32)
    else:
        raise RuntimeError('Incorrect decoder_type')

    return predictions

def query_decoder(self, enc_output, input_ids, dec_input, decoder_type, beam_size, training=False):

    _, combined_mask, dec_padding_mask = create_masks(input_ids, dec_input)
    embeddings = self.decoder_embedding(dec_input) if config.model_architecture == 'bertified_transformer' else dec_input
    # (batch_size, i+1, vocab), (_)            
    dec_output, attention_dist = self.decoder(input_ids,
                                               embeddings, 
                                               enc_output, 
                                               training, 
                                               combined_mask, 
                                               dec_padding_mask
                                               )        

    # (batch_size, 1, vocab)
    if decoder_type == 'beam_search':
        return dec_output[:, -1: ,:]
    else:
        return (dec_output[:, -1: ,:], attention_dist)

def draft_decoder(self,
                 input_ids, 
                 enc_output,
                 beam_size,
                 length_penalty,
                 decoder_type, 
                 temperature, 
                 top_p, 
                 top_k,
                 batch_size, 
                 training=False
                 ):

        """
        Inference call, builds a draft output_sequence auto-regressively
        """
        log.info(f"Building: '{decoder_type} decoder'")
        start_ids = tf.repeat(config.target_CLS_ID, repeats=batch_size)
        if decoder_type == 'beam_search':
            input_ids = tfa.seq2seq.tile_batch(input_ids, multiplier=beam_size)
            enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_size)
            
            def perform_beam_search(dec_input):

                return query_decoder(self, enc_output, input_ids, dec_input, 
                  decoder_type, beam_size, training=False)

            predicted_beam_search_op = beam_search(
                                                  perform_beam_search, 
                                                  initial_ids=start_ids, 
                                                  beam_size=beam_size, 
                                                  decode_length=config.target_seq_length, 
                                                  vocab_size=config.target_vocab_size, 
                                                  alpha=length_penalty,
                                                  stop_early=False,
                                                  eos_id=config.target_SEP_ID
                                                  )
            predicted_output_sequence = predicted_beam_search_op[0][:,0,:]
            attention_dist = None
        else:
            predicted_output_sequence = tf.expand_dims(start_ids, 1)
            for i in (range(0, config.target_seq_length)):
                # (batch_size, i+1, d_bert)
                dec_output,attention_dist = query_decoder(self,
                                                            enc_output, 
                                                            input_ids,
                                                            predicted_output_sequence,
                                                            decoder_type,
                                                            beam_size=None, 
                                                            training=False
                                                            )
                predictions = sampling_decoder(decoder_type, dec_output, batch_size, temperature, 
                                              top_p, top_k)
                predicted_output_sequence = tf.concat([predicted_output_sequence, predictions], 
                                                      axis=-1
                                                      )
        #(batch_size, seq_len, vocab_len), (_)
        return predicted_output_sequence, attention_dist
