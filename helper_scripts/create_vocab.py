# -*- coding: utf-8 -*-
import os
import tensorflow_datasets as tfds
from configuration import config
from creates import log



def create_vocab(tokenizer_path, tok_type):

	try:
		tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(config.tokenizer_path)
	except FileNotFoundError:
		log.warning(f'Vocab files not available in {config.tokenizer_path} so building it from the training set')
	    if config.use_tfds:
			examples, metadata = tfds.load(config.tfds_name, with_info=True, as_supervised=True)
			train_examples = examples['train']
			if tok_type=='source':
			  tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
			          (ip_seq.numpy() for ip_seq, _ in train_examples), target_vocab_size=2**13)
			else:
			  tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
			          (op_seq.numpy() for _, op_seq in train_examples), target_vocab_size=2**13)
	    tokenizer.save_to_file(config.tokenizer_path)
	if tok_type=='source':
		assert(tokenizer.vocab_size+2 == config.input_vocab_size),f'{tok_type}vocab size in configuration script should be {tokenizer.vocab_size+2}'
	else:
		assert(tokenizer.vocab_size+2 == config.output_vocab_size),f'{tok_type}vocab size in configuration script should be {tokenizer.vocab_size+2}'
	log.info(f'{tok_type} vocab file created and saved to {config.tokenizer_path}')
	return tokenizer


if not (config.model_architecture == 'bertified_transformer'):
	source_tokenizer = create_vocab(config.input_seq_vocab_path, 'source')
    target_tokenizer = create_vocab(config.output_seq_vocab_path, 'target')