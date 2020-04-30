def task_check(source_tokenizer, special_tokens):

	if model_parms['task'] == 'summarize':
		special_tokens['target_CLS_ID'] = special_tokens['input_CLS_ID']
    	special_tokens['target_SEP_ID'] = special_tokens['input_SEP_ID']
		target_tokenizer = source_tokenizer
	else:
		if model_parms['model_architecture'] == 'transformer':
			target_tokenizer = create_vocab(file_path['output_seq_vocab_path'], 'target')
		elif model_parms['model_architecture'] == 'bertified_transformer':
			target_tokenizer = BertTokenizer.from_pretrained(model_parms['target_pretrained_bert_model'])
		special_tokens['target_CLS_ID'] = target_tokenizer.vocab_size
        special_tokens['target_SEP_ID'] = target_tokenizer.vocab_size+1
		
	return (target_tokenizer, special_tokens)

def set_bertified_transformer_rules():
	
	source_tokenizer = BertTokenizer.from_pretrained(model_parms['input_pretrained_bert_model'])
	model_parms['input_vocab_size'] = source_tokenizer.vocab_size 
    special_tokens['input_CLS_ID'] = special_tokens['CLS_ID']
    special_tokens['input_SEP_ID'] = special_tokens['SEP_ID']
    (target_tokenizer, special_tokens) = task_check(source_tokenizer, special_tokens)
    model_parms['target_vocab_size'] = target_tokenizer.vocab_size
    model_parms['num_of_decoders'] = 2

def set_transformer_rules():

	source_tokenizer = create_vocab(file_path['input_seq_vocab_path'], 'source')
    model_parms['input_vocab_size'] = source_tokenizer.vocab_size + 2
    special_tokens['input_CLS_ID'] = source_tokenizer.vocab_size
    special_tokens['input_SEP_ID'] = source_tokenizer.vocab_size+1
    (target_tokenizer, special_tokens) = task_check(source_tokenizer, special_tokens)
    model_parms['target_vocab_size'] = target_tokenizer.vocab_size + 2
    model_parms['num_of_decoders'] = 1

