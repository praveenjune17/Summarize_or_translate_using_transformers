import time
import re
import tensorflow as tf
from create_model import  source_tokenizer,target_tokenizer, Model
from creates import detokenize
from model_utils import create_padding_mask
from configuration import config


en_blacklist = '"#$%&\()*+-./:;<=>@[\\]^_`♪{|}~='
cleantxt = re.compile('<.*?>')
def translate():
    start = time.time()
    en_input = input('Enter the sentence to be translated ')
    # Lower case english lines
    l1 = en_input.lower()
    # Remove unwanted html tags from text
    l1 = re.sub(cleantxt, '', l1)
    # Remove english text in tamil sentence and tamil text in english sentence
    cleaned_l1 = ''.join([ch for ch in l1 if ch not in en_blacklist])
    # Remove duplicate empty spaces
    cleaned_l1 = " ".join(cleaned_l1.split())
    print(cleaned_l1)
    input_ids = tf.convert_to_tensor([[config.input_CLS_ID] + source_tokenizer.encode() + [config.input_SEP_ID]])
    dec_padding_mask = create_padding_mask(input_ids)
    preds_draft_summary, _, _, _ = Model.predict(input_ids,
                                                dec_padding_mask
                                                )
                                                
    translated_sequence = target_tokenizer.decode([i for i in tf.squeeze(preds_draft_summary) if i not in [config.target_CLS_ID, 
                                                                                                           config.target_SEP_ID, 
                                                                                                           config.PAD_ID]])

    print(f'the summarized output is --> {translated_sequence if translated_sequence else "EMPTY"}')
    print(f'Time to process {round(start-time.time())} seconds')

if __name__ == '__main__':
    ckpt = tf.train.Checkpoint(
                               Model=Model
                              )
    ckpt.restore(config.infer_ckpt_path).expect_partial()
    translate()    
    
    