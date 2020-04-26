# -*- coding: utf-8 -*-
import tempfile
import shutil
import os
import tensorflow as tf
import numpy as np
from rouge import Rouge
from bert_score import score as b_score
from official.nlp.transformer import compute_bleu
from configuration import config
from creates import log, valid_output_sequence_writer, detokenize
from create_model import source_tokenizer, target_tokenizer 

calculate_combined_metric = tf.keras.metrics.Mean(name='mean', dtype=None)

class evaluation_metrics:

    def __init__(self, true_output_sequences, predicted_output_sequences):
        self.ref_sents = true_output_sequences
        self.hyp_sents = predicted_output_sequences
        self.calculate_rouge = Rouge()

    def evaluate_rouge(self):
        
        try:
            all_rouge_scores = self.calculate_rouge.get_scores(self.ref_sents , self.hyp_sents)
            avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], 
                              rouge_scores['rouge-2']["f"], 
                              rouge_scores['rouge-l']["f"]]) for rouge_scores in all_rouge_scores])
        except:
            log.warning('Some problem while calculating ROUGE so setting it to zero')
            avg_rouge_f1 = 0

        return avg_rouge_f1

    def evaluate_bert_score(self):
        
        try:
            _, _, bert_f1 = b_score(self.ref_sents, self.hyp_sents, 
                                  model_type=config.bert_score_model)
            avg_bert_f1 = np.mean(bert_f1.numpy())
        except:
            log.warning('Some problem while calculating BERT score so setting it to zero')
            avg_bert_f1 = 0
            
        return avg_bert_f1

    def evaluate_bleu_score(self, case_sensitive=False):

        ref_filename = tempfile.NamedTemporaryFile(delete=False)
        hyp_filename = tempfile.NamedTemporaryFile(delete=False)

        with tf.io.gfile.GFile(ref_filename.name, 'w') as f_ref:
            with tf.io.gfile.GFile(hyp_filename.name, 'w') as f_hyp:
                for references, hypothesis_output in zip(self.ref_sents , self.hyp_sents):
                    f_hyp.write(hypothesis_output+'\n')
                    f_ref.write(references+'\n')
        try:
            bleu_score = compute_bleu.bleu_wrapper(ref_filename = ref_filename.name, 
                                                   hyp_filename = hyp_filename.name,
                                                   case_sensitive = False)
        except:
            log.warning('Some problem while calculating BLEU score so setting it to zero')
            bleu_score = 0

        return bleu_score

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):

    def __init__(self, d_model, warmup_steps=4000):
      super(CustomSchedule, self).__init__()
      
      self.d_model = d_model
      self.d_model = tf.cast(self.d_model, tf.float32)
      self.warmup_steps = warmup_steps
      
    def __call__(self, step):

      arg1 = tf.math.rsqrt(step)
      arg2 = step * (self.warmup_steps ** -1.5)

      return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def label_smoothing(inputs, epsilon=config.epsilon_ls):
    # V <- number of channels
    V = inputs.get_shape().as_list()[-1] 
    epsilon = tf.cast(epsilon, dtype=inputs.dtype)
    V = tf.cast(V, dtype=inputs.dtype)

    return ((1-epsilon) * inputs) + (epsilon / V)


def convert_wordpiece_to_words(w_piece):

    new=[]
    for i in w_piece:
        if '##' in i:
            m = i.replace('##', '')
        else:
            if w_piece.index(i) == 0:
                m = i
            else:
                m = ' '+i
        new.append(m)

    return (''.join(new))

def mask_and_calculate_loss(mask, loss):

    mask   = tf.cast(mask, dtype=loss.dtype)
    loss = loss * mask
    loss = tf.reduce_sum(loss)/tf.reduce_sum(mask)

    return loss


def loss_function(target_ids, draft_predictions, refine_predictions, model):
    # pred shape == real shape = (batch_size, tar_seq_len, target_vocab_size)
    true_ids_3D = label_smoothing(tf.one_hot(target_ids, depth=config.target_vocab_size))
    loss_object = tf.keras.losses.CategoricalCrossentropy(
                                                      from_logits=True, 
                                                      reduction='none'
                                                      )
    draft_loss  = loss_object(true_ids_3D[:, 1:, :], draft_predictions)
    draft_mask = tf.math.logical_not(tf.math.equal(target_ids[:, 1:], config.PAD_ID))
    draft_loss = mask_and_calculate_loss(draft_mask, draft_loss)
    target = true_ids_3D[:, 1:, :]
    if refine_predictions:
        refine_loss  = loss_object(true_ids_3D[:, :-1, :], refine_predictions)
        refine_mask = tf.math.logical_not(tf.math.logical_or(tf.math.equal(
                                                                target_ids[:, :-1], 
                                                                config.target_CLS_ID
                                                                          ), 
                                                             tf.math.equal(
                                                                target_ids[:, :-1], 
                                                                config.PAD_ID
                                                                          )
                                                             )
                                          )
        refine_loss = mask_and_calculate_loss(refine_mask, refine_loss)
        target = true_ids_3D[:, :-1, :]
    else:
        refine_loss = 0.0
    regularization_loss = tf.add_n(model.losses)
    total_loss = tf.reduce_sum([draft_loss, refine_loss, regularization_loss])

    return (total_loss, target)
    
def get_loss_and_accuracy():

    loss = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.CategoricalAccuracy(name='Accuracy')

    return(loss, accuracy)
    
def write_output_sequence(input_ids, true_target_ids, predictions, step, write_output_seq):

    ref_sents = []
    hyp_sents = []
    inp_sents = []
    rouge_all = Rouge()
    for input_id, true_target_id, ref_hyp in zip(input_ids, true_target_ids, predictions):
        detokenized_refs, detokenized_hyp_sents = detokenize(target_tokenizer, 
                                                           tf.squeeze(true_target_id), 
                                                           tf.squeeze(ref_hyp) 
                                                           )
        detokenized_input_sequence,_ = detokenize(None, 
                                               tf.squeeze(input_id), 
                                               None,
                                               source_tokenizer
                                               )

        ref_sents.append(detokenized_refs)
        hyp_sents.append(detokenized_hyp_sents)
        inp_sents.append(detokenized_input_sequence)
    evaluate = evaluation_metrics(ref_sents, hyp_sents)
    rouge_f1 = evaluate.evaluate_rouge()
    bert_f1  = evaluate.evaluate_bert_score()
    bleu     = evaluate.evaluate_bleu_score()
    if write_output_seq:
        with tf.io.gfile.GFile(config.output_sequence_write_path+str(step.numpy()), 'w') as f:
            for source, ref, hyp in zip(inp_sents, ref_sents, hyp_sents):
                f.write(source+'\t'+ref+'\t'+hyp+'\n')

    return (rouge_f1, bert_f1, bleu)
  
  
def tf_write_output_sequence(input_ids, tar_real, predictions, step, write_output_seq):

    return tf.py_function(write_output_sequence, 
                          [input_ids, tar_real, predictions, step, write_output_seq], 
                          Tout=[tf.float32, tf.float32, tf.float32]
                          )
    

def monitor_run(ckpt_save_path, 
                bert_score, 
                rouge_score, 
                bleu,
                train_loss,
                step,
                copy_best_ckpt=True,
                to_monitor=config.monitor_metric):
  
    
    if config.run_tensorboard:
        with valid_output_sequence_writer.as_default():
            tf.summary.scalar('ROUGE_f1', rouge_score, step=step)
            tf.summary.scalar('BERT_f1', bert_score, step=step)
            tf.summary.scalar('BLEU', bleu, step=step)
    monitor_metrics = dict()
    monitor_metrics['BERT_f1'] = bert_score
    monitor_metrics['ROUGE_f1'] = rouge_score
    monitor_metrics['bleu'] = bleu
    monitor_metrics['combined_metric'] = [
                                          monitor_metrics['BERT_f1'], 
                                          monitor_metrics['ROUGE_f1'],
                                          monitor_metrics['bleu']
                                          ]
    assert config.monitor_metric in monitor_metrics.keys(), f'Available metrics to monitor are {monitor_metrics.keys()}'
    assert sum(config.combined_metric_weights) == 1, 'weights should sum to 1'
    monitor_metrics['combined_metric'] = calculate_combined_metric(monitor_metrics['combined_metric'], 
                                                                   sample_weight=config.combined_metric_weights)
    log.info(f"combined_metric {monitor_metrics['combined_metric'].numpy()}")
    if config.last_recorded_value <= monitor_metrics[to_monitor]:
        if copy_best_ckpt:
            # reset tolerance to zero if the monitor_metric decreases before the tolerance threshold
            ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
            config.tolerance=0
            config.last_recorded_value =  monitor_metrics[to_monitor]
            ckpt_files_tocopy = [files for files in os.listdir(os.path.split(ckpt_save_path)[0]) \
                                 if ckpt_string in files]
            log.info(f'{to_monitor} is {monitor_metrics[to_monitor]:4f} so checkpoint files {ckpt_string} \
                     will be copied to best checkpoint directory')
            # copy the best checkpoints
            shutil.copy2(os.path.join(ckpt_fold, 'checkpoint'), config.best_ckpt_path)
            for files in ckpt_files_tocopy:
                shutil.copy2(os.path.join(ckpt_fold, files), config.best_ckpt_path)
        else:
            pass
    else:
        config.tolerance+=1
    
    # stop if minimum training loss is reached
    if train_loss < config.min_train_loss:
        log.warning(f'Minimum training loss reached')
        config.tolerance+=1
    # Warn and early stop
    if config.tolerance > config.tolerance_threshold:
        log.warning('Tolerance exceeded')
        if config.early_stop:
            log.info(f'Early stopping since the {to_monitor} reached the tolerance threshold')
            return True
        else:
            return False
    else:
        return False

def get_optimizer():

    learning_rate = config.learning_rate if config.learning_rate else CustomSchedule(config.d_model)    
    if config.grad_clipnorm:
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=learning_rate, 
                                 beta_1=0.9, 
                                 beta_2=0.98, 
                                 clipnorm=config.grad_clipnorm,
                                 epsilon=1e-9
                                 )
    else:
        optimizer = tf.keras.optimizers.Adam(
                                 learning_rate=learning_rate, 
                                 beta_1=0.9, 
                                 beta_2=0.98, 
                                 epsilon=1e-9
                                 )

    return optimizer
