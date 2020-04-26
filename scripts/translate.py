import time
import re
import pickle
import tensorflow as tf
from profanity_check import predict_prob as vulgar_check
from create_model import  source_tokenizer,target_tokenizer, Model
from creates import detokenize
from model_utils import create_padding_mask
from configuration import config


en_blacklist = '"#$%&\()*+-./:;<=>@[\\]^_`♪{|}~='
cleantxt = re.compile('<.*?>')


with open('correction_dictonary.pickle', 'rb') as handle:
    correction_dictonary = pickle.load(handle)

def preprocess(sentence):
    # Lower case english lines
    sentence_lower = sentence.lower()
    # Remove unwanted html tags from text
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
    with open('correction_dictonary.pickle', 'wb') as handle:
        pickle.dump(correction_dictonary, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return translated_sequence
def translate():
    
    en_input = input('Enter the sentence to be translated-> ')
    en_input = preprocess(en_input)
    input_ids = tf.convert_to_tensor([[config.input_CLS_ID] + source_tokenizer.encode(en_input) + [config.input_SEP_ID]])
    dec_padding_mask = create_padding_mask(input_ids)
    start = time.time()
    preds_draft_summary, _, _, _ = Model.predict(input_ids,
                                                dec_padding_mask
                                                )
                                                
    translated_sequence = target_tokenizer.decode([i for i in tf.squeeze(preds_draft_summary) if i not in [config.target_CLS_ID, 
                                                                                                           config.target_SEP_ID, 
                                                                                                           config.PAD_ID]])
    translated_sequence = postprocess(en_input, 
                                      translated_sequence, 
                                      input_word_to_be_corrected='hemalatha',
                                      incorrect_target_word='ஹேமாலயா', 
                                      correct_target_word='ஹேமலதா'
                                     )
    print(f'the summarized output is --> {translated_sequence if translated_sequence else "EMPTY"}')
    print(f'Time to process --> {round(time.time()-start)} seconds')

if __name__ == '__main__':
    ckpt = tf.train.Checkpoint(
                               Model=Model
                              )
    ckpt.restore(config.infer_ckpt_path).expect_partial()
    translate()

# import pickle
# correction_dictonary = {}
# with open('correction_dictonary.pickle', 'wb') as handle:
#     pickle.dump(correction_dictonary, handle, protocol=pickle.HIGHEST_PROTOCOL)