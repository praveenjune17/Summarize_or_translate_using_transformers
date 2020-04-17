def query_decoder(self, enc_output, input_ids, dec_input, training=False):
    embeddings = self.decoder_embedding(dec_input) if config.model_architecture == 'bertified_transformer' else dec_input
    _, combined_mask, dec_padding_mask = create_masks(input_ids, embeddings)
    # (batch_size, i+1, vocab), (_)            
    dec_output, attention_dist = self.decoder(input_ids,
                                               embeddings, 
                                               enc_output, 
                                               training, 
                                               combined_mask, 
                                               dec_padding_mask
                                               )        

    # (batch_size, 1, vocab)
    return dec_output[:, -1: ,:]