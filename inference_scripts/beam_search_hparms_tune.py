import tensorflow as tf
tf.random.set_seed(100)
import tensorflow_datasets as tfds
import numpy as np
import os
from configuration import config
from creates import log
from preprocess import create_dataset
from local_tf_ops import evaluate_validation_set
from create_model import source_tokenizer, target_tokenizer, Model




infer_template = '''ROUGE-f1  <--- {}\nBERT-f1   <--- {}\nCombined_metric   <--- {}\nbeam-size   <--- {}\nlength_penalty   <--- {}'''
best_combo = '''Beam_size <--- {}\nLength_penalty <--- {}\nCombined_metric <---{}'''
def restore_chkpt(checkpoint_path):
    ckpt = tf.train.Checkpoint(
                               Model=Model
                               )
    assert tf.train.latest_checkpoint(os.path.split(checkpoint_path)[0]), 'Incorrect checkpoint directory'
    ckpt.restore(checkpoint_path).expect_partial()
    print(f'{checkpoint_path} restored')

restore_chkpt(config.infer_ckpt_path)

val_dataset = create_dataset(
                             split='validation', 
                             source_tokenizer=source_tokenizer, 
                             target_tokenizer=target_tokenizer, 
                             from_=0, 
                             to=100, 
                             batch_size=config.validation_batch_size,
                             drop_remainder=True
                             )
step=0
max_combined_metric = 0
for beam_size in [7]:
  for length_penalty in [0.4, 0.45, 0.5, 0.55, 0.6, 0.65]:
    step+=1
    (rouge_score, bert_score) = evaluate_validation_set(val_dataset, beam_size, length_penalty, step)
    combined_metric = (0.8*bert_score) +  (0.2*rouge_score)
    if max_combined_metric < combined_metric:
        max_combined_metric = combined_metric
        best_beam_size = beam_size
        best_length_penalty = length_penalty
    log.info(infer_template.format(rouge_score, bert_score, combined_metric, beam_size, length_penalty))
log.info(best_combo.format(best_beam_size,best_length_penalty, max_combined_metric))