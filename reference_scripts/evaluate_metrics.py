from bert_score import score as b_score
from rouge import Rouge
import sacrebleu

class eval_metrics(object):
	def __init__(self, hypothesis_output, references):
		super(eval_metrics, self).__init__()

		self.hypothesis_output = hypothesis_output
		self.references = references

	def calculate_rouge():
		rouge_all = Rouge()
		try:
			rouges_scores = rouge_all.get_scores(self.references , self.hypothesis_output)
			avg_rouge_f1 = np.mean([np.mean([rscore['rouge-1']["f"], 
	                                        rscore['rouge-2']["f"], 
	                                        rscore['rouge-l']["f"]]) for rscore in rouges_scores])
		except:
			log.warning('Some problem while calculating ROUGE so setting it to zero')
			avg_rouge_f1 = 0
		return avg_rouge_f1

	def calculate_bert_score():
		try:
			_, _, bert_f1 = b_score(self.references, self.hypothesis_output, model_type=config.bert_score_model)
			avg_bert_f1 = np.mean(bert_f1.numpy())
		except:
			log.warning('Some problem while calculating BERT_F1 score so setting it to zero')
			avg_bert_f1 = 0
		return avg_bert_f1

	def calculate_bleu_score():
		try:
			bleu = sacrebleu.corpus_bleu(self.references, self.hypothesis_output, lowercase=True)
		except:
			log.warning('Some problem while calculating BLEU score so setting it to zero')
			bleu = 0
		return bleu

def write_output_sequence(tar_real, predictions, step, write_output_seq):
    ref_sents = []
    hyp_sents = []
    rouge_all = Rouge()
    for tar, ref_hyp in zip(tar_real, predictions):
        detokenized_refs, detokenized_hyp_sents = detokenize(target_tokenizer, 
                                                           tf.squeeze(tar), 
                                                           tf.squeeze(ref_hyp) 
                                                           )
        ref_sents.append(detokenized_refs)
        hyp_sents.append(detokenized_hyp_sents)
    try:
        rouges = rouge_all.get_scores(ref_sents , hyp_sents)
        avg_rouge_f1 = np.mean([np.mean([rouge_scores['rouge-1']["f"], 
                                        rouge_scores['rouge-2']["f"], 
                                        rouge_scores['rouge-l']["f"]]) for rouge_scores in rouges])
        _, _, bert_f1 = b_score(ref_sents, hyp_sents, model_type=config.bert_score_model)
        avg_bert_f1 = np.mean(bert_f1.numpy())
    except:
        log.warning('Some problem while calculating ROUGE so setting ROUGE score to zero')
        avg_rouge_f1 = 0
        avg_bert_f1 = 0
    
    if write_output_seq:
        with tf.io.gfile.GFile(config.output_sequence_write_path+str(step.numpy()), 'w') as f:
            for ref, hyp in zip(ref_sents, hyp_sents):
                f.write(ref+'\t'+hyp+'\n')
    return (avg_rouge_f1, avg_bert_f1)