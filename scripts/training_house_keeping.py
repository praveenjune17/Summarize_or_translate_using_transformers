import shutil
import os
import tensorflow as tf
from utilities import create_tensorboard_parms, detokenize

model_metrics = 'Step {},\n\
                 Train Loss {:.4f},\n\
                 Train_Accuracy {:.4f},\n\
                 {} {:4f},\n\
                 BERT_f1 {:4f}\n'
evaluation_step  = 'Time taken for {} step : {} secs' 
checkpoint_details = 'Saving checkpoint at step {} on {}'
(_, valid_output_sequence_writer, _) = create_tensorboard_parms()
avg_unified_metric = tf.keras.metrics.Mean(
                        name='weighted_and_unified_metric_mean', 
                        dtype=None)

def train_sanity_check(tokenizer, predictions, target_id, log):
    # use the last sample in the batch
    predicted, target = detokenize(tokenizer, 
                                   tf.squeeze(tf.argmax(predictions,axis=-1)[-1:]), 
                                   tf.squeeze(target_id[:, :-1][-1:])
                                   )
    log.info(f'target -> {target}')
    log.info(f'predicted by teacher forcing ->\
              {predicted if predicted else "empty hence evaluation will be skipped"}')

    return predicted

def training_results(
                    step,
                    train_loss,
                    train_accuracy, 
                    task_score, 
                    bert_score,
                    timing_info,
                    ckpt_save_path,
                    log,
                    config
                    ):

      log.info(
                model_metrics.format(
                        step, 
                        train_loss,
                        train_accuracy,
                        'ROUGE_f1' if config.task == 'summarize' else 'BLEU',
                        task_score*100,
                        bert_score*100
                        )
              )
      log.info(evaluation_step.format(step, timing_info))
      log.info(checkpoint_details.format(step, ckpt_save_path))
      
def copy_checkpoint(copy_best_ckpt, ckpt_save_path, all_metrics, 
                    to_monitor, log, config):

    ckpt_fold, ckpt_string = os.path.split(ckpt_save_path)
    config.tolerance=0
    config.last_recorded_value =  all_metrics[to_monitor]
    ckpt_files_tocopy = [files for files in os.listdir(ckpt_fold) if ckpt_string in files]
    ckpt_files_tocopy = ['checkpoint'] + ckpt_files_tocopy
    for files in ckpt_files_tocopy:
        shutil.copy2(os.path.join(ckpt_fold, files), config.best_ckpt_path)
    log.info(f'{to_monitor} is {all_metrics[to_monitor]:4f} so checkpoint \
            {ckpt_string} are copied to best checkpoints directory')

def pass_eval_results_to_tensorboard(task_score, bert_score, unified_score, step, config):
    
    with valid_output_sequence_writer.as_default():
        tf.summary.scalar(f'ROUGE_f1' if config.task == 'summarize' else 'BLEU', 
                         task_score, 
                         step=step)
        tf.summary.scalar('BERT_f1', bert_score, step=step)
        tf.summary.scalar('weighted_and_unified_metric', 
                          unified_score,
                          step=step)

def early_stop(train_loss, log, config):
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
    # stop if minimum training loss is reached
    if train_loss < config.min_train_loss:
        log.warning(f'Minimum training loss reached')
        return True
    else:
        return False

def monitor_eval_metrics(ckpt_save_path, 
                bert_f1_score, 
                task_score, 
                train_loss,
                step,
                log,
                config,
                copy_best_ckpt=True):

    to_monitor=config.monitor_metric  
    all_eval_metrics = {'unified_metric' : None, 
                        'bert_f1_score' : bert_f1_score, 
                        'task_score' : task_score
                        }
    assert to_monitor in all_eval_metrics, (
                  f'Available metrics to monitor are {all_eval_metrics}')
    all_eval_metrics['unified_metric'] = avg_unified_metric([
                                        all_eval_metrics['bert_f1_score'], 
                                        all_eval_metrics['task_score']], 
                                          sample_weight=[
                                          config.metric_weights['bert_f1_score'], 
                                          config.metric_weights['task_score']
                                                    ]
                                                        ).numpy()
    log.info(f"weighted_and_unified_metric {all_eval_metrics['unified_metric']}")
    if config.run_tensorboard:
        pass_eval_results_to_tensorboard(all_eval_metrics['bert_f1_score'], 
                                        all_eval_metrics['task_score'],
                                        all_eval_metrics['unified_metric'], 
                                        step,
                                        config)
    if config.last_recorded_value <= all_eval_metrics[to_monitor]:
        if copy_best_ckpt:
            copy_checkpoint(copy_best_ckpt, ckpt_save_path, all_eval_metrics, to_monitor, log, config)
        else:
            pass
    else:
        config.tolerance+=1

    return early_stop(train_loss, log)
