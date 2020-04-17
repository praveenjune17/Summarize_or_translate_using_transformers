# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from configuration import config

'''Positional encoding is added to give the model some information about the 
   relative position of the words in the sentence.Nearby tokens will have 
   similar position-encoding vectors. Any relative position encoding can be written 
   as a linear function of the current position. Raw angles are not a good model 
   input (they're either unbounded, or discontinuous) so take the sine and cosine'''
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

def create_masks(input_ids, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(input_ids)  
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(input_ids)
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by 
    # the decoder.allows decoder to attend to all positions in the decoder up to and 
    # including that position(refer architecture)
    dec_target_padding_mask = create_padding_mask(tar)
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    return enc_padding_mask, combined_mask, dec_padding_mask



def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
      
    Returns:
      output, attention_weights
    """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)
    return output, attention_weights

'''
MultiHeadAttention:-

The scaled_dot_product_attention defined above is applied to each head (broadcasted for efficiency). 
An appropriate mask must be used in the attention step. The attention output for each head is then concatenated 
and put through a final Dense layer.Instead of one single attention head, Q, K, and V are split into multiple 
heads because it allows the model to jointly attend to information at different positions from different 
representational spaces. After the split each head has a reduced dimensionality, 
so the total computation cost is the same as a single head attention with full dimensionality.
'''
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0, 'd_model should be a multiple of num_heads'
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(
                                        d_model, 
                                        kernel_regularizer = tf.keras.regularizers.l2(config.l2_norm),
                                        )
        self.wk = tf.keras.layers.Dense(
                                        d_model, 
                                        kernel_regularizer = tf.keras.regularizers.l2(config.l2_norm),
                                        )
        self.wv = tf.keras.layers.Dense(
                                        d_model, 
                                        kernel_regularizer = tf.keras.regularizers.l2(config.l2_norm),
                                        )
        self.dense = tf.keras.layers.Dense(
                                           d_model, 
                                           kernel_regularizer = tf.keras.regularizers.l2(config.l2_norm),
                                           )
          
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
      
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        # (batch_size, seq_len, d_model)
        q = self.wq(q)  
        # (batch_size, seq_len, d_model)
        k = self.wk(k)  
        # (batch_size, seq_len, d_model)
        v = self.wv(v)  
        # (batch_size, num_heads, seq_len_q, depth)
        q = self.split_heads(q, batch_size)  
        # (batch_size, num_heads, seq_len_k, depth)
        k = self.split_heads(k, batch_size)  
        # (batch_size, num_heads, seq_len_v, depth)
        v = self.split_heads(v, batch_size)  
        
        # scaled_attention (batch_size, num_heads, seq_len_q, depth)
        # attention_weights (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)
        # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  
        # (batch_size, seq_len_q, d_model)
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model)
                                      )  
        # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  
            
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    # d_model (batch_size, seq_len, dff)
    # dff (batch_size, seq_len, d_model)
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, 
                              activation=config.activation, 
                              kernel_regularizer = tf.keras.regularizers.l2(config.l2_norm),
                              ),
        tf.keras.layers.Dense(d_model, 
                              kernel_regularizer = tf.keras.regularizers.l2(config.l2_norm),
                              )
    ])

def _embedding_from_bert():

    log.info("Extracting pretrained word embeddings weights from BERT")
    with tf.device("CPU:0"):  
        encoder = TFBertModel.from_pretrained(config.input_pretrained_bert_model, 
                                              trainable=False, 
                                              name=config.input_pretrained_bert_model)
        decoder = TFBertModel.from_pretrained(config.target_pretrained_bert_model, 
                                              trainable=False, 
                                              name=config.target_pretrained_bert_model)
    decoder_embedding = decoder.get_weights()[0]
    log.info(f"Decoder_Embedding matrix shape '{decoder_embedding.shape}'")
    return (decoder_embedding, encoder, decoder)

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=config.dropout_rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.dropout1 = tf.keras.layers.Dropout(rate, dtype='float32')
        self.dropout2 = tf.keras.layers.Dropout(rate, dtype='float32')
      
    def call(self, x, training, mask):
        # (batch_size, input_seq_len, d_model)
        
        attn_output, _ = self.mha(x, x, x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        x = tf.cast(x, tf.float32)
        out1 = self.layernorm1(x + attn_output)  
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        out2 = self.layernorm2(out1 + ffn_output)  
        
        return out2

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=config.dropout_rate):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
      
      
    def call(self, x, enc_output, training, 
             look_ahead_mask, padding_mask):
      
      
        attn1, attn_weights_block1 = self.mha1(x, 
                                               x, 
                                               x, 
                                               look_ahead_mask
                                               )  
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        attn2, attn_weights_block2 = self.mha2(
                                               enc_output, 
                                               enc_output, 
                                               out1, 
                                               padding_mask
                                               )  
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        out2 = self.layernorm2(attn2 + out1)  
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(out2)  
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        out3 = self.layernorm3(ffn_output + out2)  
        return (out3, attn_weights_block1, attn_weights_block2)


class Pointer_Generator(tf.keras.layers.Layer):
    
    def __init__(self):
        super(Pointer_Generator, self).__init__()
        self.pointer_generator_layer = tf.keras.layers.Dense(
                                                             1, 
                                                             kernel_regularizer = tf.keras.regularizers.l2(
                                                                                            config.l2_norm)
                                                             )
        self.pointer_generator_vec   = tf.keras.layers.Activation('sigmoid', dtype='float32')
      
    def call(self, dec_output, final_output, 
            attention_weights, encoder_input, 
            inp_shape, tar_shape, training):

        batch = tf.shape(encoder_input)[0]
        # p_gen (batch_size, tar_seq_len, 1)
        p_gen = self.pointer_generator_vec(self.pointer_generator_layer(dec_output))
        # vocab_dist (batch_size, tar_seq_len, target_vocab_size)   
        vocab_dist_ = tf.math.softmax(final_output, axis=-1)              #cand1
        vocab_dist = p_gen * vocab_dist_ 
        # attention_weights is 4D so taking mean of the second dimension(i.e num_heads)
        if config.mean_attention_heads:
            attention_weights_ = tf.reduce_mean(attention_weights, axis=1)
        else:
            attention_weights_ = attention_weights[:, -1, :, :]
        # attention_dist (batch_size, tar_seq_len, inp_seq_len)
        attention_dist = tf.math.softmax(attention_weights_, axis=-1)
        # updates (batch_size, tar_seq_len, inp_seq_len)
        updates = (1 - p_gen) * attention_dist
        shape = tf.shape(final_output)
        # represent the tokens indices in 3D using meshgrid and tile
        # https://stackoverflow.com/questions/45162998/proper-usage-of-tf-scatter-nd-in-tensorflow-r1-2
        i1, i2 = tf.meshgrid(tf.range(batch), tf.range(tar_shape), indexing="ij")
        i1 = tf.tile(i1[:, :, tf.newaxis], [1, 1, inp_shape])
        i2 = tf.tile(i2[:, :, tf.newaxis], [1, 1, inp_shape])
        # convert to int32 since they are compatible with scatter_nd
        indices_ = tf.cast(encoder_input, dtype=tf.int32)
        #tile on tar_seq_len so that the input vocab can be copied to output
        indices_x = tf.tile(indices_[:, tf.newaxis,: ], [1, tar_shape, 1])
        indices = tf.stack([i1, i2, indices_x], axis=-1)
        # copy_probs (batch_size, tar_seq_len, target_vocab_size)
        copy_probs = tf.scatter_nd(indices, updates, shape)   
        combined_probs = vocab_dist + copy_probs
        # ensures numerical stability
        combined_probs = tf.math.maximum(combined_probs, 0.0000000001)
        combined_logits = tf.math.log(combined_probs)
        return combined_logits

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 rate=config.dropout_rate):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(input_vocab_size, self.d_model)
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate, dtype='float32')
          
    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]
        # adding embedding and position encoding.
        # (batch_size, input_seq_len, d_model)
        x = self.encoder_embedding(x)  
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        #x (batch_size, input_seq_len, d_model)
        for i in range(self.num_layers):
          x = self.enc_layers[i](tf.cast(x, tf.float32), training, mask)
        x = tf.cast(x, tf.float32)
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
                 rate=config.dropout_rate,  add_pointer_generator=None):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(target_vocab_size, self.d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) 
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)
        self.pointer_generator = Pointer_Generator() if add_pointer_generator else None
        self.final_layer = tf.keras.layers.Dense(
                                         target_vocab_size,
                                         dtype='float32',
                                         kernel_regularizer=tf.keras.regularizers.l2(config.l2_norm),
                                         name='final_dense_layer',
                                         bias_initializer=config.add_bias
                                         )

    def call(self, input_ids, x, enc_output, training, 
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(x)[1]
        attention_weights = {}
        x = self.decoder_embedding(x)
        # (batch_size, target_seq_len, d_model) 
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x,
                                                 enc_output,
                                                 training,
                                                 look_ahead_mask,
                                                 padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # take the attention weights of the final layer 
        block2_attention_weights = attention_weights[f'decoder_layer{self.num_layers}_block2']
        # final_layer(x) <- (batch_size, tar_seq_len, target_vocab_size)
        # x              <- (batch_size, tar_seq_len, d_model)
        predictions = self.final_layer(x) #if add_dense else x
        predictions = self.pointer_generator(
                                            x, 
                                            predictions, 
                                            block2_attention_weights, 
                                            input_ids, 
                                            tf.shape(input_ids)[1], 
                                            seq_len, 
                                            training
                                            )          if self.pointer_generator  else predictions
        return predictions, block2_attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=config.dropout, add_pointer_generator=None):
    super(Transformer, self).__init__()

    self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                           input_vocab_size, rate)
    self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                           target_vocab_size, rate, add_pointer_generator)

    def fit(self, input_ids, tar, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask)

    # (batch_size, inp_seq_len, d_model)
    enc_output = self.encoder(input_ids, training, enc_padding_mask)  
    # (batch_size, tar_seq_len, target_vocab_size), ()
    final_output, attention_weights = self.decoder(
                                                input_ids,
                                                tar, 
                                                enc_output, 
                                                training, 
                                                look_ahead_mask, 
                                                dec_padding_mask
                                                )
    
    return (final_output, attention_weights, None, None)


    def predict(self,
               input_ids,
               dec_padding_mask,
               decoder_sampling_type='greedy',
               temperature=0.9, 
               p=0.9, 
               k=25):

        # (batch_size, inp_seq_len, d_model)
        # Both dec_padding_mask and enc_padding_mask are same
        enc_output = self.encoder(input_ids, False, dec_padding_mask)
        if decoder_sampling_type=='beam_search':
            predicted_beam_search_op = self.draft_output_sequence_beam_search(
                                                                        input_ids, 
                                                                        enc_output, 
                                                                        dec_padding_mask, 
                                                                        config.beam_size,
                                                                        tf.shape(input_ids)[0]
                                                                        )
            predicted_draft_output_sequence = predicted_beam_search_op[0][:,0,:]
            draft_attention_dist = None
        else:
            (predicted_draft_output_sequence, 
                        draft_attention_dist) = self.draft_output_sequence_sampling(
                                                                  input_ids,
                                                                  enc_output=enc_output,
                                                                  look_ahead_mask=None,
                                                                  padding_mask=dec_padding_mask,
                                                                  decoder_type=draft_decoder_sampling_type,
                                                                  temperature=temperature,
                                                                  p=p, 
                                                                  k=k,
                                                                )
        return (predicted_draft_output_sequence, draft_attention_dist, None, None)

    @tf.function
    def call(self, input_ids, dec_padding_mask, enc_padding_mask=None, 
           look_ahead_mask=None, tar=None, training=None):

    if training is not None:
        return self.fit(self, input_ids, tar, training, enc_padding_mask, 
                        look_ahead_mask, dec_padding_mask)
    else:
        return self.predict(self, input_ids, dec_padding_mask)
