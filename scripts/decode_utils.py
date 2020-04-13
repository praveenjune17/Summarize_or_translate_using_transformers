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
    add extra dimensions so that padding can be added
    to the attention logits.
    """
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    # (batch_size, 1, 1, seq_len)
    return seq[:, tf.newaxis, tf.newaxis, :]  

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

def sampling_decoder(decoder_type, decoder_op, batch_size, temperature, p, k):

    if decoder_type == 'nucleus':
        predictions = tf.cast(nucleus_sampling((decoder_op/ temperature), batch_size, p=p), tf.int32)
    elif decoder_type == 'topk':
        predictions = tf.cast(top_k_sampling(((decoder_op)/ temperature), batch_size, k=k), tf.int32)
    elif decoder_type == 'topktopp':
        predictions = tf.cast(topp_topk(((decoder_op)/ temperature), batch_size, p=p, k=k), tf.int32)
    elif decoder_type == 'random_sampling':
        predictions = tf.cast(sampling(decoder_op/ temperature), tf.int32)
    elif decoder_type == 'greedy':
        predictions = tf.cast(tf.argmax(decoder_op, axis=-1), tf.int32)
    else:
        raise RuntimeError('Incorrect decoder_type given')

    return predictions
