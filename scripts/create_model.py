import tensorflow as tf
from tensorflow.keras.initializers import Constant
from transformers import TFBertModel
from transformer import Decoder, Transformer
from utilities import log
from configuration import config
from model_utils import (tile_and_mask_diagonal, create_masks, topp_topk,
                         with_column, mask_timestamp, draft_decoder)

def _embedding_from_bert():

    with tf.device("CPU:0"):  
        input_pretrained_bert = TFBertModel.from_pretrained(config.input_pretrained_bert_model, 
                                              trainable=False, 
                                              name=config.input_pretrained_bert_model)
        target_pretrained_bert = TFBertModel.from_pretrained(config.target_pretrained_bert_model, 
                                              trainable=False, 
                                              name=config.target_pretrained_bert_model)
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

        N = tf.shape(enc_output)[0]
        T = config.target_seq_length
        # (batch_size x (seq_len - 1), seq_len) 
        dec_inp_ids = tile_and_mask_diagonal(target, mask_with=config.MASK_ID)
        # (batch_size x (seq_len - 1), seq_len, d_bert)
        context_vectors = self.decoder_bert_model(dec_inp_ids)[0]
        # (batch_size x (seq_len - 1), seq_len, d_bert) 
        enc_output = tf.tile(enc_output, [T-1, 1, 1])
        # (batch_size x (seq_len - 1), 1, 1, seq_len) 
        padding_mask = tf.tile(padding_mask, [T-1, 1, 1, 1])
        # (batch_size x (seq_len - 1), seq_len, vocab_len), (_)
        refined_logits, refine_attention_dist = self.decoder(
                                                           tf.tile(input_ids, [T-1, 1]),
                                                           context_vectors,
                                                           enc_output,
                                                           training,
                                                           look_ahead_mask=None,
                                                           padding_mask=padding_mask
                                                         )
        # (batch_size x (seq_len - 1), seq_len - 1, vocab_len)
        refined_logits = refined_logits[:, 1:, :]
        # (batch_size x (seq_len - 1), (seq_len - 1))
        diag = tf.linalg.set_diag(tf.zeros([T-1, T-1]), tf.ones([T-1]))
        diag = tf.tile(diag, [N, 1])
        where = tf.not_equal(diag, 0)
        indices = tf.where(where)
        # (batch_size x (seq_len - 1), vocab_len)
        refined_logits = tf.gather_nd(refined_logits, indices)
        # (batch_size, seq_len - 1, vocab_len)
        refined_logits = tf.reshape(refined_logits, [N, T-1, -1])
        # (batch_size, seq_len, vocab_len)
        refine_logits = tf.concat(
                           [tf.tile(
                                    tf.expand_dims(
                                                  tf.one_hot(
                                                    [config.target_CLS_ID], 
                                                    self.target_vocab_size
                                                            ), 
                                                  axis=0
                                                  ), 
                                                  [N, 1, 1]
                                    ), 
                                    refined_logits],
                                    axis=1
                                  )


        # (batch_size, seq_len, vocab_len)
        return refine_logits, refine_attention_dist

    def refined_output_sequence_sampling(self,
                                         input_ids, 
                                         enc_output, 
                                         draft_output_sequence, 
                                         batch_size, 
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
            logits = topp_topk(logits=dec_output_i, 
                                batch_size=batch_size,
                                temperature=temperature, 
                                top_k=k, 
                                top_p=p)
            if decoder_type == 'greedy':
                predictions = tf.expand_dims(tf.math.argmax(logits, axis=-1, output_type=tf.int32), 1)
            else:
                predictions = tf.random.categorical(logits, num_samples=1, dtype=tf.int32, seed=1)
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
               draft_decoder_sampling_type=config.draft_decoder_type,
               refine_decoder_type=config.refine_decoder_type,
               beam_size=config.beam_size,
               length_penalty=config.length_penalty, 
               temperature=config.softmax_temperature, 
               top_p=config.topp, 
               top_k=config.topk):

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
                                                decoder_type=draft_decoder_sampling_type,
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
                                            p=top_p, 
                                            k=top_k
                                            )
        
        return (predicted_draft_output_sequence, draft_attention_dist, 
               predicted_refined_output_sequence, refined_attention_dist)

    def call(self, input_ids, target_ids, dec_padding_mask, 
             enc_padding_mask, look_ahead_mask, training):

        if training is not None:
            return self.fit(input_ids, target_ids, training, enc_padding_mask, 
                            look_ahead_mask, dec_padding_mask)
        else:
            return self.predict(input_ids)

if config.model_architecture == 'transformer':
    Model = Transformer(
                       num_layers=config.num_layers, 
                       d_model=config.d_model, 
                       num_heads=config.num_heads, 
                       dff=config.dff, 
                       input_vocab_size=config.input_vocab_size, 
                       target_vocab_size=config.target_vocab_size,
                       add_pointer_generator=config.add_pointer_generator
                       )
        
elif config.model_architecture == 'bertified_transformer':
    Model = Bertified_transformer(
                                  num_layers=config.num_layers, 
                                  d_model=config.d_model, 
                                  num_heads=config.num_heads, 
                                  dff=config.dff, 
                                  input_vocab_size=config.input_vocab_size,
                                  target_vocab_size=config.target_vocab_size,
                                  add_pointer_generator=config.add_pointer_generator
                                  )
