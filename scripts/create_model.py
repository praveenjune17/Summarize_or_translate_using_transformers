import tensorflow as tf
from tensorflow.keras.initializers import Constant
from transformers import TFAutoModel
from transformer import Decoder, Transformer
from utilities import log
from configuration import config
from model_utils import (tile_and_mask_diagonal, create_masks, topp_topk,
                         with_column, mask_timestamp, draft_decoder)

def _embedding_from_bert():

    with tf.device("CPU:0"):  
        input_pretrained_bert = TFAutoModel.from_pretrained(config.input_pretrained_model, 
                                              trainable=False, 
                                              name=config.input_pretrained_model)
        target_pretrained_bert = TFAutoModel.from_pretrained(config.target_pretrained_model, 
                                              trainable=False, 
                                              name=config.target_pretrained_model)
    decoder_embedding = target_pretrained_bert.get_weights()[0]
    log.info(f"Decoder_Embedding matrix shape '{decoder_embedding.shape}'")

    return (decoder_embedding, input_pretrained_bert, target_pretrained_bert)

class Bertified_transformer(tf.keras.Model):
    """
    Pretraining-Based Natural Language Generation for Text Summarization 
    https://arxiv.org/pdf/1902.09243.pdf
    """
    def __init__(
                  self, 
                  num_layers, 
                  d_model, 
                  num_heads, 
                  dff, 
                  input_vocab_size, 
                  target_vocab_size,
                  rate=config.dropout_rate, 
                  add_pointer_generator=None):
        super(Bertified_transformer, self).__init__()

        self.target_vocab_size = target_vocab_size
        (decoder_embedding, self.encoder, 
        self.decoder_bert_model) = _embedding_from_bert()
        self.decoder_embedding = tf.keras.layers.Embedding(
                                       target_vocab_size, 
                                       d_model, 
                                       trainable=False,
                                       embeddings_initializer=Constant(decoder_embedding),
                                       name='Decoder-embedding'
                                       )
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate, 
                               add_pointer_generator=add_pointer_generator)
    def draft_summary(self,
                      input_ids,
                      enc_output,
                      look_ahead_mask,
                      padding_mask,
                      target_ids,
                      training):
        # (batch_size, seq_len, d_bert)
        dec_ip = self.decoder_embedding(target_ids)
        # (batch_size, seq_len, vocab_len), (_)            
        draft_logits, draft_attention_dist = self.decoder(
                                                          input_ids,
                                                          dec_ip, 
                                                          enc_output, 
                                                          training, 
                                                          look_ahead_mask, 
                                                          padding_mask
                                                          )
        # (batch_size, seq_len, vocab_len)
        return draft_logits, draft_attention_dist

    def refine_summary(self,
                       input_ids, 
                       enc_output, 
                       target, 
                       padding_mask, 
                       training):

        batch_size = tf.shape(enc_output)[0]
        # exclued CLS_ID from tiling and masking
        max_time_steps = config.target_seq_length - 1
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_ids = tile_and_mask_diagonal(target, batch_size, 
                                max_time_steps, mask_with=config.MASK_ID)
        # (batch_size x (seq_len - 1), seq_len, d_bert)
        context_vectors = self.decoder_bert_model(dec_inp_ids)[0]
        # (batch_size x (seq_len - 1), seq_len)
        input_ids = tf.tile(input_ids, [max_time_steps, 1])
        # (batch_size x (seq_len - 1), seq_len, d_bert) 
        enc_output = tf.tile(enc_output, [max_time_steps, 1, 1])
        # (batch_size x (seq_len - 1), 1, 1, seq_len) 
        padding_mask = tf.tile(padding_mask, [max_time_steps, 1, 1, 1])
        # (batch_size x (seq_len - 1), seq_len, vocab_len), (_)
        refined_logits, refine_attention_dist = self.decoder(
                                                           input_ids,
                                                           context_vectors,
                                                           enc_output,
                                                           training,
                                                           look_ahead_mask=None,
                                                           padding_mask=padding_mask
                                                         )
        # (batch_size x (seq_len - 1), seq_len - 1, vocab_len)
        refined_logits = refined_logits[:, 1:, :]
        # tile an identity matrix (batch_size*(seq_len - 1), (seq_len - 1))
        mark_masked_indices = tf.tile(tf.eye(max_time_steps, dtype=tf.bool), [batch_size, 1])
        # (batch_size x (seq_len - 1), vocab_len)
        refined_logits = tf.gather_nd(refined_logits, indices=tf.where(mark_masked_indices))
        # (batch_size, seq_len - 1, vocab_len)
        refined_logits = tf.reshape(refined_logits, [batch_size, max_time_steps, -1])
        # (batch_size, 1, vocab_len)
        cls_logits = tf.tile(tf.one_hot(
                                    [config.CLS_ID], 
                                    self.target_vocab_size)[tf.newaxis,:,:], 
                            [batch_size, 1, 1]
                            )
        # (batch_size, seq_len, vocab_len)
        total_refine_logits = tf.concat([cls_logits, refined_logits], axis=1)
        # (batch_size, seq_len, vocab_len)
        return total_refine_logits, refine_attention_dist

    def refined_output_sequence_sampling(self,
                                         input_ids, 
                                         enc_output, 
                                         draft_output_sequence, 
                                         batch_size, 
                                         decoder_type, 
                                         temperature, 
                                         top_p, 
                                         top_k,
                                         training=False):
        """
        Inference call, builds a refined output_sequence
        
        It first masks each word in the output_sequence draft one by one,
        then feeds the draft to BERT to generate context vectors.
        """
        
        #log.info(f"Building: 'Refined {decoder_type} decoder'")
        dec_input = draft_output_sequence
        for i in (range(1, config.target_seq_length)):    

            # (batch_size, seq_len)
            dec_input = mask_timestamp(dec_input, i, config.MASK_ID)
            _, _, dec_padding_mask = create_masks(input_ids, dec_input)
            # (batch_size, seq_len, d_bert)
            context_vectors = self.decoder_bert_model(dec_input)[0]
            # (batch_size, seq_len, d_bert), (_)
            dec_output,  attention_dist =  self.decoder(input_ids,
                                                        context_vectors,
                                                        enc_output,
                                                        training=training,
                                                        look_ahead_mask=None,
                                                        padding_mask=dec_padding_mask
                                                      )
            # (batch_size, 1, vocab_len)
            dec_output_i = dec_output[:, i:i+1 ,:]
            truncated_logits = topp_topk(logits=dec_output_i, 
                                batch_size=batch_size,
                                temperature=temperature, 
                                top_k=top_k, 
                                top_p=top_p)
            if decoder_type == 'greedy':
                predictions = tf.expand_dims(tf.math.argmax(truncated_logits, axis=-1, output_type=tf.int32), 1)
            else:
                predictions = tf.random.categorical(truncated_logits, num_samples=1, dtype=tf.int32, seed=1)
            dec_input = with_column(dec_input, i, predictions)
        # (batch_size, seq_len, vocab_len), (_)        
        return dec_input, attention_dist

    def fit(self, input_ids, target_ids, training, enc_padding_mask, 
           look_ahead_mask, dec_padding_mask):
        
        # (batch_size, seq_len, d_bert)
        enc_output = self.encoder(input_ids)[0]
        # (batch_size, seq_len, vocab_len), _
        draft_logits, draft_attention_dist = self.draft_summary(
                                                                input_ids,
                                                                enc_output=enc_output,
                                                                look_ahead_mask=look_ahead_mask,
                                                                padding_mask=dec_padding_mask,
                                                                target_ids=target_ids,
                                                                training=training
                                                               )
        # (batch_size, seq_len, vocab_len), _
        refine_logits, refine_attention_dist = self.refine_summary(
                                                                input_ids,
                                                                enc_output=enc_output,
                                                                target=target_ids,            
                                                                padding_mask=dec_padding_mask,
                                                                training=training
                                                                )
              
        return draft_logits, draft_attention_dist, refine_logits, refine_attention_dist

    def predict(self,
               input_ids,
               draft_decoder_type,
               beam_size,
               length_penalty, 
               temperature, 
               top_p, 
               top_k,
               refine_decoder_type=config.refine_decoder_type):

        # (batch_size, seq_len, d_bert)
        batch_size = tf.shape(input_ids)[0]
        enc_output = self.encoder(input_ids)[0]
        # (batch_size, seq_len, vocab_len), 
        # ()
        (predicted_draft_output_sequence, 
          draft_attention_dist) = draft_decoder(self,
                                                input_ids,
                                                enc_output=enc_output,
                                                beam_size=beam_size,
                                                length_penalty=length_penalty,
                                                temperature=temperature,
                                                top_p=top_p, 
                                                top_k=top_k,
                                                batch_size=batch_size
                                                )
        # (batch_size, seq_len, vocab_len), 
        # ()
        (predicted_refined_output_sequence, 
          refined_attention_dist) = self.refined_output_sequence_sampling(
                                            input_ids,
                                            enc_output=enc_output,
                                            draft_output_sequence=predicted_draft_output_sequence,
                                            decoder_type=refine_decoder_type,
                                            batch_size=batch_size, 
                                            temperature=temperature, 
                                            top_p=top_p, 
                                            top_k=top_k
                                            )
        
        return (predicted_draft_output_sequence, draft_attention_dist, 
               predicted_refined_output_sequence, refined_attention_dist)

    

    def call(self, input_ids, target_ids, dec_padding_mask, 
                 enc_padding_mask, look_ahead_mask, training,
                 decoder_type=config.draft_decoder_type,
                 beam_size=config.beam_size,
                 length_penalty=config.length_penalty,
                 temperature=config.softmax_temperature, 
                 top_p=config.top_p,
                 top_k=config.top_k):

        if training is not None:
            return self.fit(input_ids, target_ids, training, enc_padding_mask, 
                            look_ahead_mask, dec_padding_mask)
        else:
            return self.predict(input_ids,
                               draft_decoder_type=decoder_type,
                               beam_size=beam_size,
                               length_penalty=length_penalty,
                               temperature=temperature, 
                               top_p=top_p,
                               top_k=top_k)

if config.model == 'transformer':
    Model = Transformer(
                       num_layers=config.num_layers, 
                       d_model=config.d_model, 
                       num_heads=config.num_heads, 
                       dff=config.dff, 
                       input_vocab_size=config.input_vocab_size, 
                       target_vocab_size=config.target_vocab_size,
                       add_pointer_generator=config.add_pointer_generator
                       )
        
elif config.model == 'bertified_transformer':
    Model = Bertified_transformer(
                                  num_layers=config.num_layers, 
                                  d_model=config.d_model, 
                                  num_heads=config.num_heads, 
                                  dff=config.dff, 
                                  input_vocab_size=config.input_vocab_size,
                                  target_vocab_size=config.target_vocab_size,
                                  add_pointer_generator=config.add_pointer_generator
                                  )

if config.print_config:
    if config['add_bias']:
          config['add_bias'] = True
    log.info(f'Configuration used \n {config}')
    
