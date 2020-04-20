import time
import tensorflow as tf
from create_model import  target_tokenizer, Model
from creates import detokenize
from model_utils import create_padding_mask
from configuration import config

def translate():
    ckpt = tf.train.Checkpoint(
                               Model=Model
                              )
    ckpt.restore('D:\\Local_run\\best_checkpoints\\en_tam_parallel_text\\ckpt-87').expect_partial()

    start = time.time()
    input_ids = tf.convert_to_tensor([[config.input_CLS_ID] + target_tokenizer.encode(input('Enter the sentence to be translated ')) + [config.input_SEP_ID]])
    dec_padding_mask = create_padding_mask(input_ids)
    preds_draft_summary, _, _, _ = Model(input_ids,
                                          dec_padding_mask
                                          )
                                                
    translated_sequence = target_tokenizer.decode([i for i in tf.squeeze(preds_draft_summary) if i not in [config.target_CLS_ID, 
                                                                                                           config.target_SEP_ID, 
                                                                                                           config.PAD_ID]])

    print(f'the summarized output is --> {translated_sequence if translated_sequence else "EMPTY"}')
    print(f'Time to process {round(start-time.time())} seconds')

if __name__ == '__main__':
    translate()
