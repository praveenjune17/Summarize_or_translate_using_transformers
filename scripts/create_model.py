import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tensorflow.keras.initializers import Constant
from transformer import create_masks, Decoder
from creates import log
from configuration import config

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

def _embedding_from_bert():

  log.info("Extracting pretrained word embeddings weights from BERT")  
  encoder = TFBertModel.from_pretrained(config.input_pretrained_bert_model, trainable=False)
  decoder = TFBertModel.from_pretrained(config.target_pretrained_bert_model, trainable=False)
  decoder_embedding = decoder.get_weights()[0]
  log.info(f"Decoder_Embedding matrix shape '{decoder_embedding.shape}'")
  return (decoder_embedding, encoder, decoder)

class AbstractiveSummarization(tf.keras.Model):
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
                  target_vocab_size, 
                  output_seq_len, 
                  rate=0.1):
        super(AbstractiveSummarization, self).__init__()
        
        self.output_seq_len = output_seq_len
        self.target_vocab_size = target_vocab_size
        (decoder_embedding, self.encoder_bert_model, 
          self.decoder_bert_model) = _embedding_from_bert()
        self.decoder_embedding = tf.keras.layers.Embedding(
                                                           target_vocab_size, 
                                                           d_model, 
                                                           trainable=False,
                                                           embeddings_initializer=Constant(decoder_embedding)
                                                           )        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)

    def draft_summary(self,
                      input_ids,
                      enc_output,
                      look_ahead_mask,
                      padding_mask,
                      target_ids,
                      training):
        # (batch_size, seq_len, d_bert)
        embeddings = self.decoder_embedding(target_ids) 
        # (batch_size, seq_len, vocab_len), (_)            
        draft_logits, draft_attention_dist = self.decoder(
                                                          input_ids,
                                                          embeddings, 
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
        T = self.output_seq_len
        # since we are using teacher forcing we do not need an autoregressice mechanism here
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
                                                    [config.CLS_ID], 
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

    def call(self, 
             input_ids, 
             target_ids, 
             training):

           # (batch_size, 1, 1, seq_len), (batch_size, 1, 1, seq_len)
        _, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids[:, :-1])

        # (batch_size, seq_len, d_bert)
        enc_output = self.encoder_bert_model(input_ids)[0]             #Eng bert

        # (batch_size, seq_len, vocab_len), _
        draft_logits, draft_attention_dist = self.draft_summary(
                                                                input_ids,
                                                                enc_output=enc_output,
                                                                look_ahead_mask=combined_mask,
                                                                padding_mask=dec_padding_mask,
                                                                target_ids=target_ids[:, :-1],
                                                                training=True
                                                               )

        # (batch_size, seq_len, vocab_len), _
        refine_logits, refine_attention_dist = self.refine_summary(
                                                                  input_ids,
                                                                  enc_output=enc_output,
                                                                  target=target_ids[:, :-1],            
                                                                  padding_mask=dec_padding_mask,
                                                                  training=True
                                                                  )
              
        return draft_logits, draft_attention_dist, refine_logits, refine_attention_dist

Model = AbstractiveSummarization(
                                num_layers=config.num_layers, 
                                d_model=config.d_model, 
                                num_heads=config.num_heads, 
                                dff=config.dff, 
                                target_vocab_size=config.target_vocab_size,
                                output_seq_len=config.target_seq_length, 
                                rate=config.dropout_rate
                                )

source_tokenizer = BertTokenizer.from_pretrained(config.input_pretrained_bert_model)
target_tokenizer = BertTokenizer.from_pretrained(config.target_pretrained_bert_model)
