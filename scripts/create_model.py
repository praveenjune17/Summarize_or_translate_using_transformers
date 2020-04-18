import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import Constant
from transformers import TFBertModel, BertTokenizer
from transformer import Decoder, Encoder, Transformer
from creates import log, create_vocab
from configuration import config
from model_utils import (tile_and_mask_diagonal, sampling_decoder, 
                         with_column, mask_timestamp, draft_decoder)

call_signature = [
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.int32),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None, None), dtype=tf.bool),
                tf.TensorSpec(shape=(None), dtype=tf.bool)
                ]

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
                  rate):
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
                               add_pointer_generator=True)
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
        enc_output = tf.tile(enc_output, [T-1, 1, 1])
        # (batch_size x (seq_len - 1), 1, 1, seq_len) 
        padding_mask = tf.tile(padding_mask, [T-1, 1, 1, 1])
        # (batch_size x (seq_len - 1), seq_len, d_bert)
        context_vectors = self.decoder_bert_model(dec_inp_ids)[0]
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
        #tf.shape(enc_output)[0]
        refined_output_sequence = draft_output_sequence
        for i in (range(1, config.target_seq_length)):    
            # (batch_size, seq_len)
            masked_refined_output_sequence = mask_timestamp(refined_output_sequence, i, config.MASK_ID)
            # (batch_size, seq_len, d_bert)
            context_vectors = self.decoder_bert_model(masked_refined_output_sequence)[0]
            # (batch_size, seq_len, d_bert), (_)
            dec_output,  attention_dist =  self.decoder(input_ids,
                                                        context_vectors,
                                                        enc_output,
                                                        training=training,
                                                        look_ahead_mask=None,
                                                        padding_mask=padding_mask
                                                      )
            # (batch_size, 1, vocab_len)
            dec_output_i = dec_output[:, i:i+1 ,:]
            predictions = sampling_decoder(decoder_type, dec_output_i, temperature, p, k)
            refined_output_sequence = with_column(refined_output_sequence, i, predictions)
        # (batch_size, seq_len, vocab_len), (_)        
        return refined_output_sequence, attention_dist

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
               dec_padding_mask, 
               draft_decoder_sampling_type=config.decoder_type,
               refine_decoder_type='topk',
               beam_size=config.beam_size, 
               temperature=config.softmax_temperature, 
               top_p=config.topp, 
               top_k=config.topk):

        # (batch_size, seq_len, d_bert)
        enc_output = self.encoder(input_ids)[0]
        # (batch_size, seq_len, vocab_len), 
        # ()
        (predicted_draft_output_sequence, 
          draft_attention_dist) = draft_decoder(self,
                                                input_ids,
                                                enc_output=enc_output,
                                                beam_size=beam_size,
                                                decoder_type=draft_decoder_sampling_type,
                                                temperature=temperature,
                                                top_p=top_p, 
                                                top_k=top_k,
                                                )
        
        
        # (batch_size, seq_len, vocab_len), 
        # ()
        (predicted_refined_output_sequence, 
          refined_attention_dist) = self.refined_output_sequence_sampling(
                                            input_ids,
                                            enc_output=enc_output,
                                            padding_mask=dec_padding_mask,
                                            draft_output_sequence=predicted_draft_output_sequence,
                                            decoder_type=refine_decoder_type, 
                                            temperature=temperature, 
                                            p=p, 
                                            k=k
                                            )
        
        return (predicted_draft_output_sequence, draft_attention_dist, 
               predicted_refined_output_sequence, refined_attention_dist)

    #@tf.function(input_signature=call_signature)
    def call(self, input_ids, target_ids, dec_padding_mask, 
             enc_padding_mask, look_ahead_mask, training):

        if training is not None:
            return self.fit(input_ids, target_ids, training, enc_padding_mask, 
                            look_ahead_mask, dec_padding_mask)
        else:
            return self.predict(input_ids, dec_padding_mask)
        

if not (config.model_architecture == 'bertified_transformer'):
    source_tokenizer = create_vocab(config.input_seq_vocab_path, 'source', log)
    target_tokenizer = create_vocab(config.output_seq_vocab_path, 'target', log)
    Model = Transformer(
                       num_layers=config.num_layers, 
                       d_model=config.d_model, 
                       num_heads=config.num_heads, 
                       dff=config.dff, 
                       input_vocab_size=config.input_vocab_size, 
                       target_vocab_size=config.target_vocab_size,
                       add_pointer_generator=config.add_pointer_generator
                       )
else:
    source_tokenizer = BertTokenizer.from_pretrained(config.input_pretrained_bert_model)
    target_tokenizer = BertTokenizer.from_pretrained(config.target_pretrained_bert_model)
    Model = Bertified_transformer(
                                  num_layers=config.num_layers, 
                                  d_model=config.d_model, 
                                  num_heads=config.num_heads, 
                                  dff=config.dff, 
                                  input_vocab_size=config.input_vocab_size,
                                  target_vocab_size=config.target_vocab_size,
                                  add_pointer_generator=config.add_pointer_generator
                                  )
