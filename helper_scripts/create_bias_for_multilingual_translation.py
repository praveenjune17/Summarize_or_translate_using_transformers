import requests
import numpy as np
import tensorflow as tf
from langdetect import detect
from configuration import config

url = f'https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-vocab.txt'
r = requests.get(url)
bert_multilingual_vocab_size = 119547

def count_language_vocab(assign_bias_to_indices=None):
    
    target_language_count=0
    for i, ch in enumerate(r.content.decode('utf-8').split('\n')):
        try:
            if detect(ch) == config.target_language:
                target_language_count+=1
                if assign_bias_to_indices is not None:
                    assign_bias_to_indices[i] = 0
        except:
            continue
    minority_ratio = np.log(target_language_count/(bert_multilingual_vocab_size-target_language_count))
    if assign_bias_to_indices is None:
        return minority_ratio
    else:
        return assign_bias_to_indices

assert config.task == 'translate' , 'Do this only for translation task'
assert config.target_pretrained_bert_model =='bert-base-multilingual-cased', 'bias works only for multilingual model'
minority_ratio = count_language_vocab()
bias = [minority_ratio]*bert_multilingual_vocab_size
bias = count_language_vocab(bias)

serialized_tensor = tf.io.serialize_tensor(bias, name='output_bias')
tf.io.write_file(config.serialized_tensor_path, 
                 serialized_tensor, 
                 name=None
                )

# read_tensor = tf.io.read_file(
#     config.serialized_tensor_path, name=None
# )