# -*- coding: utf-8 -*-
import tensorflow as tf
import tensorflow_addons as tfa
from configuration import config
from model_utils import positional_encoding, draft_decoder

call_signature = [
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None), dtype=tf.bool)
                ]

def scaled_dot_product_attention(q, k, v, mask):
    # (..., seq_len_q, seq_len_k)
    matmul_qk = tf.matmul(q, k, transpose_b=True)  
    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)  
    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    # (..., seq_len_q, seq_len_k)
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # (..., seq_len_q, depth_v)
    output = tf.matmul(attention_weights, v)  

    return output, attention_weights

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

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=config.dropout_rate):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6, dtype='float32')
        self.dropout1 = tf.keras.layers.Dropout(rate, dtype='float32')
        self.dropout2 = tf.keras.layers.Dropout(rate, dtype='float32')
      
    def call(self, input_ids, training, mask):
        # (batch_size, input_seq_len, d_model)
        
        attn_output, _ = self.mha(input_ids, input_ids, input_ids, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        input_ids = tf.cast(input_ids, tf.float32)
        layer_norm_out1 = self.layernorm1(input_ids + attn_output)  
        # (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn(layer_norm_out1)  
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, d_model)
        encoder_output = self.layernorm2(layer_norm_out1 + ffn_output)  
        
        return encoder_output

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
      
    def call(self, target_ids, enc_output, training, 
             look_ahead_mask, padding_mask):
      
        attn1, attn_weights_block1 = self.mha1(target_ids, 
                                               target_ids, 
                                               target_ids, 
                                               look_ahead_mask
                                               )  
        attn1 = self.dropout1(attn1, training=training)
        layer_norm_out1 = self.layernorm1(attn1 + target_ids)
        attn2, attn_weights_block2 = self.mha2(
                                               enc_output, 
                                               enc_output, 
                                               layer_norm_out1, 
                                               padding_mask
                                               )  
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, d_model)
        layer_norm_out2 = self.layernorm2(attn2 + layer_norm_out1)  
        # (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn(layer_norm_out2)  
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, d_model)
        decoder_output = self.layernorm3(ffn_output + layer_norm_out2)  
        return (decoder_output, attn_weights_block1, attn_weights_block2)


class Pointer_Generator(tf.keras.layers.Layer):
    
    def __init__(self):
        super(Pointer_Generator, self).__init__()
        self.pointer_generator_layer = tf.keras.layers.Dense(
                                             1, 
                                             kernel_regularizer = tf.keras.regularizers.l2(
                                                                            config.l2_norm)
                                             )
        self.pointer_generator_vector = tf.keras.layers.Activation('sigmoid', dtype='float32')
      
    def call(self, dec_output, final_output, 
            attention_weights, encoder_input, 
            input_shape, target_shape, training):

        batch = tf.shape(encoder_input)[0]
        # pointer_generator (batch_size, tar_seq_len, 1)
        pointer_generator = self.pointer_generator_vector(
                                self.pointer_generator_layer(dec_output)
                                                         )
        vocab_dist = tf.math.softmax(final_output, axis=-1)
        # weighted_vocab_dist (batch_size, tar_seq_len, target_vocab_size)
        weighted_vocab_dist = pointer_generator * vocab_dist
        # attention_weights is 4D so taking mean of the second dimension(i.e num_heads)
        final_attention_weights = tf.reduce_mean(attention_weights, axis=1)
        # attention_dist (batch_size, tar_seq_len, inp_seq_len)
        attention_dist = tf.math.softmax(final_attention_weights, axis=-1)
        # weighted_attention_dist (batch_size, tar_seq_len, inp_seq_len)
        weighted_attention_dist = (1 - pointer_generator) * attention_dist
        attention_dist_shape = tf.shape(final_output)
        # represent the tokens indices in 3D using meshgrid and tile
        batch_indices, target_indices = tf.meshgrid(tf.range(batch), 
                                                    tf.range(target_shape), 
                                                    indexing="ij")
        tiled_batch_indices = tf.tile(batch_indices[:, :, tf.newaxis], 
                                      [1, 1, input_shape]
                                      )
        tiled_target_indices = tf.tile(target_indices[:, :, tf.newaxis], 
                                       [1, 1, input_shape]
                                       )
        # convert to int32 since they are compatible with scatter_nd
        encoder_input = tf.cast(encoder_input, dtype=tf.int32)
        #tile on tar_seq_len so that the input vocab can be copied to output
        tiled_encoder_input = tf.tile(encoder_input[:, tf.newaxis,: ], 
                                      [1, target_shape, 1]
                                      )
        gather_attention_indices = tf.stack([tiled_batch_indices, 
                                            tiled_target_indices, 
                                            tiled_encoder_input
                                            ], 
                                            axis=-1
                                            )
        # selected_attention_dist (batch_size, tar_seq_len, target_vocab_size)
        selected_attention_dist = tf.scatter_nd(gather_attention_indices, 
                                                weighted_attention_dist, 
                                                attention_dist_shape
                                                )   
        total_distribution = weighted_vocab_dist + selected_attention_dist
        # ensures numerical stability
        total_distribution = tf.math.maximum(total_distribution, 0.0000000001)
        logits = tf.math.log(total_distribution)
        return logits

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
          
    def call(self, input_ids, training, mask):
        seq_len = tf.shape(input_ids)[1]
        # (batch_size, input_seq_len, d_model)
        input_ids = self.encoder_embedding(input_ids)  
        input_ids *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))  
        input_ids += self.pos_encoding[:, :seq_len, :]
        input_ids = self.dropout(input_ids, training=training)
        #input_ids (batch_size, input_seq_len, d_model)
        for i in range(self.num_layers):
          input_ids = self.enc_layers[i](tf.cast(input_ids, tf.float32), training, mask)
        input_ids = tf.cast(input_ids, tf.float32)
        return input_ids

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

    def call(self, input_ids, target_ids, enc_output, training, 
             look_ahead_mask, padding_mask):
        seq_len = tf.shape(target_ids)[1]
        attention_weights = {}
        if not config.model_architecture=='bertified_transformer':
            target_ids = self.decoder_embedding(target_ids)
        # (batch_size, target_seq_len, d_model) 
        target_ids *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        target_ids += self.pos_encoding[:, :seq_len, :]
        target_ids = self.dropout(target_ids, training=training)
        # target_ids  <- (batch_size, tar_seq_len, d_model)
        for i in range(self.num_layers):
            target_ids, block1, block2 = self.dec_layers[i](target_ids,
                                                 enc_output,
                                                 training,
                                                 look_ahead_mask,
                                                 padding_mask)
            attention_weights['decoder_layer{}_block1'.format(i+1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i+1)] = block2
        
        # take the attention weights of the final layer 
        block2_attention_weights = attention_weights[f'decoder_layer{self.num_layers}_block2']
        # predictions <- (batch_size, tar_seq_len, target_vocab_size)
        predictions = self.final_layer(target_ids) 
        predictions = self.pointer_generator(
                                            target_ids, 
                                            predictions, 
                                            block2_attention_weights, 
                                            input_ids, 
                                            tf.shape(input_ids)[1], 
                                            seq_len, 
                                            training
                                            )      if self.pointer_generator  else predictions
        return predictions, block2_attention_weights


class Transformer(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
               target_vocab_size, rate=config.dropout_rate, add_pointer_generator=None):
        super(Transformer, self).__init__()

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, 
                               input_vocab_size, rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, 
                               target_vocab_size, rate, add_pointer_generator)

    def fit(self, input_ids, target_ids, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):

        # (batch_size, inp_seq_len, d_model)
        enc_output = self.encoder(input_ids, training, enc_padding_mask)  
        # (batch_size, tar_seq_len, target_vocab_size), ()
        final_output, attention_weights = self.decoder(
                                                    input_ids,
                                                    target_ids, 
                                                    enc_output, 
                                                    training, 
                                                    look_ahead_mask, 
                                                    dec_padding_mask
                                                    )
        
        return (final_output, attention_weights, None, None)


    def predict(self,
               input_ids,
               dec_padding_mask,
               decoder_sampling_type=config.decoder_type,
               beam_size=config.beam_size,
               temperature=config.softmax_temperature, 
               top_p=config.topp,
               top_k=config.topk):

        # (batch_size, inp_seq_len, d_model)
        # Both dec_padding_mask and enc_padding_mask are same
        batch_size = tf.shape(input_ids)[0]
        if decoder_sampling_type=='beam_search':
            input_ids = tfa.seq2seq.tile_batch(input_ids, multiplier=beam_size)
            dec_padding_mask = tfa.seq2seq.tile_batch(dec_padding_mask, multiplier=beam_size)
        enc_output = self.encoder(input_ids, False, dec_padding_mask)
        # (batch_size, seq_len, vocab_len), 
        # ()
        (predicted_draft_output_sequence, 
          draft_attention_dist) = draft_decoder(self,
                                                input_ids,
                                                enc_output=enc_output,
                                                beam_size=beam_size,
                                                decoder_type=decoder_sampling_type,
                                                temperature=temperature,
                                                top_p=top_p, 
                                                top_k=top_k,
                                                batch_size=batch_size
                                                )

        return (predicted_draft_output_sequence, draft_attention_dist, None, None)

    #@tf.function(input_signature=call_signature)
    def call(self, input_ids, target_ids, dec_padding_mask, 
                enc_padding_mask, look_ahead_mask, training):

        if training is not None:
            return self.fit(input_ids, target_ids, training, enc_padding_mask, 
                            look_ahead_mask, dec_padding_mask)
        else:
            return self.predict(input_ids, dec_padding_mask)
