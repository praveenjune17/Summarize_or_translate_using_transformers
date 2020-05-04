import sys
sys.path.insert(0, 'D:\\Local_run\\Summarize_or_translate_using_transformers\\scripts')
sys.path.insert(0, 'D:\\Local_run\\models')
import tensorflow as tf
tf.random.set_seed(100)
import tensorflow_datasets as tfds
import numpy as np
import os
from configuration import config, source_tokenizer, target_tokenizer
from utilities import log
from preprocess import create_dataset
from model_training_helper import evaluate_validation_set
from create_model import Model

infer_template = '''BLEU  <--- {}\nBERT-f1   <--- {}\nCombined_metric   <--- {}\nbeam-size   <--- {}\nstep   <--- {}'''
best_combo = '''Beam_size <--- {}\nLength_penalty <--- {}\ntemperature <--- {}\ntop_p <--- {}\ntop_k <--- {}\nCombined_metric <---{}'''

def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               Model=Model
                               )
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'Incorrect checkpoint directory'
    ckpt.restore(checkpoint_path).expect_partial()
    print(f'{checkpoint_path} restored')

restore_chkpt(config.infer_ckpt_path)

test_dataset = create_dataset(
                             split='test', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size,
                             drop_remainder=True
                             )
max_combined_metric = 0
decoder_type = 'topktopp'
temperatures = [1]
length_penalties = [0.8]
beams =  [12, 13, 14, 15]
top_ps = [1]
top_ks = [10]

for beam_size in beams:
    for length_penalty in length_penalties:
        for top_p in top_ps:
            for top_k in top_ks:
                for temperature in temperatures:

                    step='_ '.join([str(i) for i in (decoder_type,
                                                   beam_size,
                                                   length_penalty,
                                                   temperature, 
                                                   top_p,
                                                   top_k)])
                    (task_score, bert_score) = evaluate_validation_set(test_dataset,
                                                                       step,
                                                                       decoder_type,
                                                                       beam_size,
                                                                       length_penalty,
                                                                       temperature, 
                                                                       top_p,
                                                                       top_k)
                    combined_metric = (0.8*bert_score) +  (0.2*task_score)
                    if max_combined_metric < combined_metric:
                        max_combined_metric = combined_metric
                        best_beam_size = beam_size
                        best_length_penalty = length_penalty
                        best_temperature = temperature 
                        best_top_p = top_p
                        best_top_k = top_k
                    log.info(infer_template.format(task_score, bert_score, combined_metric, beam_size, step))
log.info(best_combo.format(
                        best_beam_size,
                        best_length_penalty,
                        best_temperature,
                        best_top_p,
                        best_top_k, 
                        max_combined_metric
                        )
        )

