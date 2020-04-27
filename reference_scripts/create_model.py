import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from tensorflow.keras.initializers import Constant
from beam_search import beam_search
from transformers import TFBertModel, BertTokenizer
from transformer import create_masks, Decoder, Encoder
from creates import log
from configuration import config
from decode_utils import (tile_and_mask_diagonal, sampling_decoder, 
                          with_column, mask_timestamp)


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
                  input_vocab_size, 
                  target_vocab_size, 
                  output_seq_len, 
                  rate):
        super(AbstractiveSummarization, self).__init__()
        if config.use_refine_decoder:
            self.output_seq_len = output_seq_len
            self.target_vocab_size = target_vocab_size
        else:
            self.output_seq_len = None
            self.target_vocab_size = None
        if config.use_BERT:
            (decoder_embedding, self.encoder, 
            self.decoder_bert_model) = _embedding_from_bert()
            self.decoder_embedding = tf.keras.layers.Embedding(
                                                     target_vocab_size, 
                                                     d_model, 
                                                     trainable=False,
                                                     embeddings_initializer=Constant(decoder_embedding),
                                                     name='Decoder-embedding'
                                                     )
        else:
            self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, rate)
            self.decoder_bert_model = None
            self.decoder_embedding = None
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, rate)
    @tf.function
    def draft_summary(self,
                      input_ids,
                      enc_output,
                      look_ahead_mask,
                      padding_mask,
                      target_ids,
                      training):
        # (batch_size, seq_len, d_bert)
        if config.use_BERT:
            dec_ip = self.decoder_embedding(target_ids)
        else:
            dec_ip = target_ids
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
    @tf.function
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
        if config.use_BERT:
            context_vectors = self.decoder_bert_model(dec_inp_ids)[0]
        else:
            context_vectors = dec_inp_ids

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
    @tf.function
    def draft_output_sequence_sampling(self,
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
        dec_input = tf.expand_dims([config.target_CLS_ID]*N, 0)#tf.ones([N, 1], dtype=tf.int32) * config.target_CLS_ID
        #dec_outputs, dec_logits, attention_dists = [], [], []
        for i in (range(0, config.target_seq_length)):
            # (batch_size, i+1, d_bert)
            if config.use_BERT:
                embeddings = self.decoder_embedding(dec_input)
            else:
                embeddings = dec_input
            _, combined_mask, dec_padding_mask = create_masks(
                                                      inp, embeddings)
            # (batch_size, i+1, vocab), (_)            
            dec_output, attention_dist = self.decoder(inp,
                                                       embeddings, 
                                                       enc_output, 
                                                       training, 
                                                       combined_mask, 
                                                       dec_padding_mask
                                                       )        

            # (batch_size, 1, vocab)
            dec_output_i = dec_output[:, -1: ,:]
            predictions = sampling_decoder(decoder_type, dec_output_i, N, temperature, p, k)
            # return the result if the predicted_id is equal to the end token
            # if predictions == config.target_SEP_ID:
            #     return dec_input, attention_dist
            dec_input = tf.concat([dec_input, predictions], axis=-1)
            
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len), (_)
        return dec_input, attention_dist

    @tf.function
    def draft_output_sequence_beam_search(self,
                                          input_ids, 
                                          enc_output, 
                                          dec_padding_mask,
                                          beam_size,
                                          batch_size
                                          ):

        log.info(f"Building: 'Draft beam search decoder'")
        input_ids = tfa.seq2seq.tile_batch(input_ids, multiplier=beam_size)
        enc_output = tfa.seq2seq.tile_batch(enc_output, multiplier=beam_size)
        dec_padding_mask = tfa.seq2seq.tile_batch(dec_padding_mask, multiplier=beam_size)
        def beam_search_decoder(output):
            # (batch_size, seq_len, d_bert)    
            if config.use_BERT:
                embeddings = self.decoder_embedding(output)
            else:
                embeddings = output

            predictions, attention_weights = self.decoder(input_ids,
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
                            [config.target_CLS_ID] * batch_size, 
                            beam_size, 
                            config.target_seq_length, 
                            config.target_vocab_size, 
                            config.length_penalty,
                            stop_early=False,
                            eos_id=[[config.target_SEP_ID]]
                            )
    @tf.function
    def refined_output_sequence_sampling(self,
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
        #tf.shape(enc_output)[0]
        refined_output_sequence = draft_output_sequence
        for i in (range(1, config.target_seq_length)):    
            # (batch_size, seq_len)
            refined_output_sequence_ = mask_timestamp(refined_output_sequence, i, config.MASK_ID)
            # (batch_size, seq_len, d_bert)
            if config.use_BERT:
                context_vectors = self.decoder_bert_model(refined_output_sequence_)[0]
            else:
                context_vectors = refined_output_sequence_
            # (batch_size, seq_len, d_bert), (_)
            dec_output,  attention_dist =  self.decoder(inp,
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

    def fit(self, 
            input_ids, 
            target_ids, 
            training):
        # (batch_size, 1, 1, seq_len), (batch_size, 1, 1, seq_len)
        enc_padding_mask, combined_mask, dec_padding_mask = create_masks(input_ids, target_ids[:, :-1])
        # (batch_size, seq_len, d_bert)
        if config.use_BERT:
            enc_output = self.encoder(input_ids)[0]             #Eng bert
        else:
            enc_output = self.encoder(input_ids, training, enc_padding_mask)
        # (batch_size, seq_len, vocab_len), _
        draft_logits, draft_attention_dist = self.draft_summary(
                                                                input_ids,
                                                                enc_output=enc_output,
                                                                look_ahead_mask=combined_mask,
                                                                padding_mask=dec_padding_mask,
                                                                target_ids=target_ids[:, :-1],
                                                                training=training
                                                               )

        if config.use_refine_decoder:
          # (batch_size, seq_len, vocab_len), _
            refine_logits, refine_attention_dist = self.refine_summary(
                                                                    input_ids,
                                                                    enc_output=enc_output,
                                                                    target=target_ids[:, :-1],            
                                                                    padding_mask=dec_padding_mask,
                                                                    training=training
                                                                    )
        else:
            refine_logits, refine_attention_dist = 0, 0
              
        return draft_logits, draft_attention_dist, refine_logits, refine_attention_dist

    @tf.function    
    def predict(self,
               inp, 
               dec_padding_mask,
               draft_decoder_sampling_type='greedy',
               refine_decoder_type='topk', 
               temperature=0.9, 
               p=0.9, 
               k=25):

        # (batch_size, seq_len, d_bert)
        if config.use_BERT:
            enc_output = self.encoder(inp)[0]            
        else:
            enc_output = self.encoder(inp, False, dec_padding_mask)
        # (batch_size, seq_len, vocab_len), 
        #  ()

        if draft_decoder_sampling_type=='beam_search':
            predicted_beam_search_op = self.draft_output_sequence_beam_search(inp, 
                                                                              enc_output, 
                                                                              dec_padding_mask, 
                                                                              config.beam_size,
                                                                              tf.shape(inp)[0]
                                                                              )
            predicted_draft_output_sequence = predicted_beam_search_op[0][:,0,:]
            draft_attention_dist = None
        else:
            (predicted_draft_output_sequence, 
                        draft_attention_dist) = self.draft_output_sequence_sampling(
                                                                          inp,
                                                                          enc_output=enc_output,
                                                                          look_ahead_mask=None,
                                                                          padding_mask=dec_padding_mask,
                                                                          decoder_type=draft_decoder_sampling_type,
                                                                          temperature=temperature,
                                                                          p=p, 
                                                                          k=k,
                                                                        )
        
        if config.use_refine_decoder:
          # (batch_size, seq_len, vocab_len), 
          # ()
            (predicted_refined_output_sequence, 
                    refined_attention_dist) = self.refined_output_sequence_sampling(
                                                                      inp,
                                                                      enc_output=enc_output,
                                                                      padding_mask=dec_padding_mask,
                                                                      draft_output_sequence=predicted_draft_output_sequence,
                                                                      decoder_type=refine_decoder_type, 
                                                                      temperature=temperature, 
                                                                      p=p, 
                                                                      k=k
                                                                      )
        else:
            predicted_refined_output_sequence, refined_attention_dist = 0, 0

        return (predicted_draft_output_sequence, draft_attention_dist, 
               predicted_refined_output_sequence, refined_attention_dist)

    @tf.function
    def call(self, 
             input_ids,
             target_ids=None, 
             training=None):

        
        return self.fit(input_ids, target_ids, training)
        
Model = AbstractiveSummarization(
                                num_layers=config.num_layers, 
                                d_model=config.d_model, 
                                num_heads=config.num_heads, 
                                dff=config.dff, 
                                input_vocab_size=config.input_vocab_size,
                                target_vocab_size=config.target_vocab_size,
                                output_seq_len=config.target_seq_length, 
                                rate=config.dropout_rate
                                )

if config.use_BERT:
    source_tokenizer = BertTokenizer.from_pretrained(config.input_pretrained_bert_model)
    target_tokenizer = BertTokenizer.from_pretrained(config.target_pretrained_bert_model)
else:
    source_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.input_seq_vocab_path)
    target_tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.output_seq_vocab_path)
