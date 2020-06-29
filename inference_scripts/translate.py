import sys
sys.path.insert(0, '/content/Summarize_or_translate_using_transformers/scripts')
import time
import re
import pickle
import tensorflow as tf
import tensorflow_datasets as tfds
from profanity_check import predict_prob as vulgar_check
from create_model import  Model
from model_utils import create_padding_mask
from configuration import config, source_tokenizer, target_tokenizer

def preprocess(sentence):
    en_blacklist = '"#$%&\()*+-./:;<=>@[\\]^_`♪{|}~='
    cleantxt = re.compile('<.*?>')
    # Lower case english lines
    sentence_lower = sentence.lower()
    # Remove html tags from text
    cleaned_sentence = re.sub(cleantxt, '', sentence_lower)
    # Remove english text in tamil sentence and tamil text in english sentence
    cleaned_sentence = ''.join([ch for ch in cleaned_sentence if ch not in en_blacklist])
    # Remove duplicate empty spaces
    preprocessed_sentence = " ".join(cleaned_sentence.split())
    vulgar_prob = vulgar_check([preprocessed_sentence])[0]
    if vulgar_prob > 0.6:
        raise ValueError("No vulgar words please :) ")
    else:
        return preprocessed_sentence

def postprocess(input_sentence, translated_sequence, input_word_to_be_corrected,
                incorrect_target_word, correct_target_word):
    correction_available = correction_dictonary.get(input_word_to_be_corrected, None)
    if correction_available is None:
        if input_word_to_be_corrected in input_sentence:
            translated_sequence = translated_sequence.replace(incorrect_target_word, correct_target_word)
            correction_dictonary[input_word_to_be_corrected] = (incorrect_target_word, correct_target_word)
    else:
        if input_word_to_be_corrected in input_sentence:
            translated_sequence = translated_sequence.replace(correction_available[0], correction_available[1])    
    return translated_sequence

def translate():
    
    en_input = input('Enter the sentence to be translated-> ')
    en_input = preprocess(en_input)
    input_CLS_ID = source_tokenizer.vocab_size
    input_SEP_ID = source_tokenizer.vocab_size+1
    target_CLS_ID = target_tokenizer.vocab_size
    target_SEP_ID = target_tokenizer.vocab_size+1

    input_ids = tf.convert_to_tensor([[input_CLS_ID] + source_tokenizer.encode(en_input) + [input_SEP_ID]])
    dec_padding_mask = create_padding_mask(input_ids)
    start = time.time()
    preds_draft_summary, _, _, _ = Model.predict(input_ids,
                                                dec_padding_mask
                                                )
                                                
    translated_sequence = target_tokenizer.decode([i for i in tf.squeeze(preds_draft_summary) if i not in [target_CLS_ID, 
                                                                                                           target_SEP_ID,
                                                                                                           config.PAD_ID]])
    print(f'the summarized output is --> {translated_sequence if translated_sequence else "EMPTY"}')
    print(f'Time to process --> {round(time.time()-start)} seconds')

if __name__ == '__main__':
    ckpt = tf.train.Checkpoint(
                               Model=Model
                              )
    ckpt.restore('/content/content/drive/My Drive/best_checkpoints/en_tam_parallel_text/ckpt-232').expect_partial()
    translate()
