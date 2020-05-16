'''
Rl objective:-
	a) Train the model using log likelihood:- as usual
	b) Perform training using RL objective:
		*) From the trained model, 
		  y^s <-- generate random sample (sampling distribution for inference)
		  y^  <-- generated sequence using argmax(draft or refine predictions)
		X) <- Take BERT score(x) 
		Y) <- BERT score(y)

		Z) use y^s in the categorical cross entropy loss. like below
			Instead of the existing
				*) loss_object(true_ids_3D[:, 1:, :], draft_predictions)
			use
				*) loss_object(one_hot(y^s), y^)

Loss = (Y - X)*(Z)

a) each refine step should carry its own loss function..unify the total loss when applying gradients
	*) create draft_loss function:- nll 
	*) mlm loss
'''

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
        dec_input = [config.CLS_ID]#draft_output_sequence
        sample_targets = [config.CLS_ID]
        for i in (range(1, config.target_seq_length)):    

            # (batch_size, seq_len)
            #dec_input = mask_timestamp(dec_input, i, config.MASK_ID)
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
            dec_output_i = dec_output[:, -1: ,:]
            predicted_id = tf.expand_dims(tf.cast(tf.argmax(dec_output_i, axis=-1), tf.int32), 1)
            sample_return = tf.random.categorical(dec_output_i, num_samples=1, dtype=tf.int32, seed=1)
            sample_targets = tf.concat([sample_targets, sample_return], axis=-1)

            # return the result if the predicted_id is equal to the end token
		    if predicted_id == config.SEP_ID:
		      return (dec_input, sample_targets)
            # if decoder_type == 'greedy':
            #     predictions = tf.expand_dims(tf.math.argmax(truncated_logits, axis=-1, output_type=tf.int32), 1)
            # else:
            #     predictions = tf.random.categorical(truncated_logits, num_samples=1, dtype=tf.int32, seed=1)
            dec_input = tf.concat([dec_input, predicted_id], axis=-1)
        # (batch_size, seq_len, vocab_len), (batch_size, seq_len, vocab_len)
        return dec_input, sample_targets

#use tokenizer.decode_batch to decode the above two and find the bert score
#pass the above two to loss
#RL loss = (b2-b1)*loss