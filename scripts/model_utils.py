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

def set_tensor_by_indices_to_value(tensor, indices, value):
    # create value_tensor since tensor value assignment is not possible in TF
    value_tensor = tf.zeros_like(tensor) + value

    return tf.where(indices, value_tensor, tensor)

def scatter_values_on_batch_indices(values, batch_indices):
    shape = tf.shape(batch_indices)
    # broadcast batch dim to shape
    broad_casted_batch_dims = tf.reshape(tf.broadcast_to(tf.expand_dims(tf.range(shape[0]), axis=-1), shape), [1, -1])
    # transform batch_indices to pair_indices
    pair_indices = tf.transpose(tf.concat([broad_casted_batch_dims, tf.reshape(batch_indices, [1, -1])], 0))
    # scatter values to pair indices
    return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), shape)

def topp_topk(logits, batch_size, temperature, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size, vocabulary size)
            if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            Make sure we keep at least min_tokens_to_keep per batch example in the output
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    logits = tf.squeeze(logits, 1)
    logits = tf.divide(logits, temperature)
    logits_shape = tf.shape(logits)
    if top_k > 0:
        #top_k = min(max(top_k, min_tokens_to_keep), logits_shape[-1])  # Safety check
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < tf.math.top_k(logits, k=top_k)[0][..., -1, None]
        logits = set_tensor_by_indices_to_value(logits, indices_to_remove, filter_value)

    if top_p < 1.0:
        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        sorted_logits = tf.gather(
            logits, sorted_indices, axis=-1, batch_dims=1
        )  # expects logits to be of dim (batch_size, vocab_size)

        cumulative_probs = tf.math.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > top_p

        if min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove = tf.concat(
                [
                    tf.zeros_like(sorted_indices_to_remove[:, :min_tokens_to_keep]),
                    sorted_indices_to_remove[:, min_tokens_to_keep:],
                ],
                -1,
            )

        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = tf.concat(
            [tf.zeros_like(sorted_indices_to_remove[:, :1]), sorted_indices_to_remove[:, 1:]], -1,
        )
        # scatter sorted tensors to original indexing
        indices_to_remove = scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
        logits = set_tensor_by_indices_to_value(logits, indices_to_remove, filter_value)

    return logits

def query_decoder(self, enc_output, input_ids, 
      dec_input, batch_size, temperature, 
      top_p, top_k, training=False):

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
    logits = tf.divide(dec_output[:, -1: ,:], temperature)
    predictions = topp_topk(logits=logits,
                            batch_size=batch_size,
                            temperature=temperature,
                            top_k=top_k, 
                            top_p=top_p)
    # (batch_size, 1, vocab)
    return predictions

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
        input_ids = tfa.seq2seq.tile_batch(input_ids, multiplier=beam_size)
        enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_size)
        def perform_beam_search(dec_input):

            return query_decoder(self, enc_output, input_ids, dec_input, 
                                batch_size, temperature, top_p, 
                                top_k, training=training)

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

        return predicted_output_sequence, attention_dist
