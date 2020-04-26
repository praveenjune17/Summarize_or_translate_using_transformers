import tempfile
import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, 'D:\\Local_run\\models')
from official.nlp.transformer import compute_bleu
from rouge import Rouge
from bert_score import score as b_score
from configuration import config

class evaluation_metrics:
    def __init__(self, true_output_sequences, predicted_output_sequences):
        self.ref_sents = true_output_sequences
        self.hyp_sents = predicted_output_sequences
        self.calculate_rouge = Rouge()
        _ = b_score(["I'm Batman"], ["I'm Spiderman"], lang='en', model_type=config.target_pretrained_bert_model)
        log.info('Loaded Pre-trained BERT for BERT-SCORE calculation')

    def evaluate_rouge(self):
        
        try:
            all_rouge_scores = self.calculate_rouge.get_scores(self.ref_sents , self.hyp_sents)
            avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], 
                                            rouge_scores['rouge-2']["f"], 
                                            rouge_scores['rouge-l']["f"]]) for rouge_scores in all_rouge_scores])
        except:
            log.warning('Some problem while calculating ROUGE so setting it to zero')
            print('problem calculating rouge')
            avg_rouge_f1 = 0

        return avg_rouge_f1

    def evaluate_bert_score(self):
        
        try:
            _, _, bert_f1 = b_score(self.ref_sents, self.hyp_sents, model_type=config.bert_score_model)
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
                for refs, hyps in zip(self.ref_sents , self.hyp_sents):
                    f_hyp.write(hyps+'\n')
                    f_ref.write(refs+'\n')
        try:
            bleu_score = compute_bleu.bleu_wrapper(ref_filename = ref_filename.name, 
                                                   hyp_filename = hyp_filename.name,
                                                   case_sensitive = False)
        except:
            log.warning('Some problem while calculating BLEU score so setting it to zero')
            bleu_score = 0

        return bleu_score

true_output_sequences = ['அதனை நிரூபிப்பதுபோல் இருக்குமாம் படம்']
predicted_output_sequences = ['படத்தின் நிரூபிக்கும் படம் இதுதான் நிரூபிக்கப்படும்']
evaluate = evaluation_metrics(true_output_sequences, predicted_output_sequences)

print(f' ROUGE {evaluate.evaluate_rouge()}')
print(f' BERT {evaluate.evaluate_bert_score()}')
print(f' BLEU {evaluate.evaluate_bleu_score()}')
