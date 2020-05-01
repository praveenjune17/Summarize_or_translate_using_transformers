import tensorflow as tf
import tensorflow_datasets as tfds
from transformers import BertTokenizer

def create_vocab(tokenizer_path, tok_type, log=None):

    try:
        tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file(
                                                            tokenizer_path
                                                            )
    except FileNotFoundError:
        if log is None:
            print(f'Vocab files not available in {tokenizer_path} so\
                            building it from the training set')
        else:
            log.warning(f'Vocab files not available in {tokenizer_path} so \
                             building it from the training set')
        if config['use_tfds']:
            examples, metadata = tfds.load(config['tfds_name'], 
                                            with_info=True, 
                                            as_supervised=True)
            train_examples = examples['train']
            if tok_type=='source':
              tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                      (ip_seq.numpy() for ip_seq, _ in train_examples), 
                      target_vocab_size=2**13)
            else:
              tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
                      (op_seq.numpy() for _, op_seq in train_examples), 
                      target_vocab_size=2**13)
        tokenizer.save_to_file(tokenizer_path)

    if log is None:
        print(f'{tok_type} vocab file created and saved to {tokenizer_path}')
    else:
        log.info(f'{tok_type} vocab file created and saved to {tokenizer_path}')

    return tokenizer

def task_check(source_tokenizer, config):

    if config['task'] == 'summarize':
        config['bert_score_model'] = 'bert-base-uncased'
        config['target_CLS_ID'] = config['input_CLS_ID']
        config['target_SEP_ID'] = config['input_SEP_ID']
        target_tokenizer = source_tokenizer
    else:
        config['bert_score_model'] = 'bert-base-multilingual-cased'
        if config['model_architecture'] == 'transformer':
            target_tokenizer = create_vocab(config['output_seq_vocab_path'], 'target')
            config['target_CLS_ID'] = target_tokenizer.vocab_size
            config['target_SEP_ID'] = target_tokenizer.vocab_size+1
        elif config['model_architecture'] == 'bertified_transformer':
            target_tokenizer = BertTokenizer.from_pretrained(
                                            config['target_pretrained_bert_model']
                                                            )
            config['target_CLS_ID'] = config['input_CLS_ID']
            config['target_SEP_ID'] = config['input_SEP_ID']

    return (target_tokenizer, config)

def set_bertified_transformer_rules(config):
    
    source_tokenizer = BertTokenizer.from_pretrained(config['input_pretrained_bert_model'])
    config['input_vocab_size'] = source_tokenizer.vocab_size 
    config['input_CLS_ID'] = config['CLS_ID']
    config['input_SEP_ID'] = config['SEP_ID']
    (target_tokenizer, config) = task_check(source_tokenizer, config)
    config['target_vocab_size'] = target_tokenizer.vocab_size
    config['num_of_decoders'] = 2

    return(config, source_tokenizer, target_tokenizer)

def set_transformer_rules(config):

    source_tokenizer = create_vocab(config['input_seq_vocab_path'], 'source')
    config['input_vocab_size'] = source_tokenizer.vocab_size + 2
    config['input_CLS_ID'] = source_tokenizer.vocab_size
    config['input_SEP_ID'] = source_tokenizer.vocab_size+1
    (target_tokenizer, config) = task_check(source_tokenizer, config)
    config['target_vocab_size'] = target_tokenizer.vocab_size + 2
    config['num_of_decoders'] = 1

    return(config, source_tokenizer, target_tokenizer)

def set_inference_rules(config):

    if config['draft_decoder_type'] == 'greedy':
        config['draft_decoder_type'] = 'only_beam_search'
        config['beam_size'] = 1
        config['topp'] = 1 
        config['topk'] = 0

    elif config['draft_decoder_type'] == 'only_beam_search':
        config['topp'] = 1 
        config['topk'] = 0 

    return config

def load_and_set_bias(path):

    read_tensor = tf.io.read_file(path, name=None)
    output_bias_tensor = tf.io.parse_tensor(read_tensor, tf.float32, name=None)

    return tf.keras.initializers.Constant(output_bias_tensor.numpy())

def set_testing_rules(config):

    if config['unit_test_dataset_batch_size'] < config['gradient_accumulation_steps']:
        config['gradient_accumulation_steps'] =  config['unit_test_dataset_batch_size']
    if config['unit_test_dataset_batch_size'] > config['samples_to_test']:
        config['unit_test_dataset_batch_size'] = config['samples_to_test']
    if config['steps_to_print_training_info'] > config['unit_test_dataset_batch_size']:
        config['steps_to_print_training_info'] = config['unit_test_dataset_batch_size']
    config['grad_clipnorm'] = None
    config['run_tensorboard'] = False
    config['dropout_rate'] = config['epsilon_ls'] = config['l2_norm'] = 0
    config['batches_to_train'] = config['batches_to_test']

    return config

def set_training_rules(config):

    config['save_initial_weights'] = False
    config['run_init_eval'] = False
    config['init_loss_check'] = False
    config['input_independent_baseline_check'] = False
    config['check_model_capacity'] = False
    config['random_results_check'] = False
    config['print_config'] = True
    config['clear_log'] = False

    return config

def adhere_task_rules(config):

    if config['model_architecture'] == 'transformer':
        (config, source_tokenizer, 
            target_tokenizer) = set_transformer_rules(config)
    elif config['model_architecture'] == 'bertified_transformer':
        (config, source_tokenizer, 
            target_tokenizer) = set_bertified_transformer_rules(config)
        config['d_model'] = 768
        config['dff']: 2048
        config['num_heads'] = 8
        config['num_layers'] = 8

    if config['accumulate_gradients'] == False:
        config['gradient_accumulation_steps'] = 1

    if config.add_bias is not None:
        if (config.target_pretrained_bert_model == 'bert-base-multilingual-cased'
        and config.task == 'translation'):
            assert config.target_language in config.serialized_tensor_path, (
            'serialized Bias file not found,\
            please create it using helper scripts/create_bias script')
            config['add_bias'] = load_and_set_bias(config['serialized_tensor_path'])
        else:
            assert False,(
            'add_bias is only available for bert-base-multilingual-cased\
             and translation combo')

    if config['test_script']:
        config = set_testing_rules(config)
    else:
        config = set_training_rules(config)

    config = set_inference_rules(config)

    return (config, source_tokenizer, target_tokenizer)

# create metrics dict
def assert_config_values(config):

    allowed_decoder_types = ['topktopp','greedy', 'only_beam_search']
    allowed_model_architectures = ['transformer', 'bertified_transformer']
    summarization_datasets = ['cnn_dailymail']
    translation_datasets = ['en_tam_parallel_text']
    implemented_tasks = ['summarize', 'translate']
    assert config.task in implemented_tasks, 'summarize and translate are implemented currently'
    assert config.d_model % config.num_heads == 0, 'd_model should be a multiple of num_heads'
    assert config.eval_after_steps % config.steps_to_print_training_info == 0, (
    'For printing the training results "steps_to_print_training_info"\
     must be a factor of eval_after_steps')
    assert config.draft_decoder_type  in allowed_decoder_types, (
            f'available decoding types are {allowed_decoder_types}')
    assert config.model_architecture  in allowed_model_architectures, (
            f'available model_architectures are {allowed_model_architectures}')
    assert len(config.metric_weights) == 2,'Only two metrics are allowed'
    assert sum(config.metric_weights.values()) == 1, 'weights should sum to 1'
    if config.task == 'summarize':
        assert config.tfds_name in summarization_datasets, (
                f'{config.tfds_name} is not a summarization dataset')
        assert config.input_seq_length > config.target_seq_length, (
            'input_seq_length must be greater than target_seq_length for summarize')
        assert config.input_pretrained_bert_model == config.target_pretrained_bert_model, (
                f'For {config.task} the input and target models must be same  for {config.task}')
    elif config.task == 'translate':
        assert config.tfds_name in translation_datasets , (
                f'{config.tfds_name} is not a  translation dataset')
        if config.model_architecture == 'bertified_transformer':
            assert config.input_pretrained_bert_model != config.target_pretrained_bert_model, (
                f'For translate the input and target pre-trained BERT must not be same')

    return config