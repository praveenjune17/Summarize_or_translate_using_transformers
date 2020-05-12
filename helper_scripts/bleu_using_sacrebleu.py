bleu_score = sacrebleu.corpus_bleu(self.hyp_sents, [self.ref_sents], 
                              lowercase=True, tokenize='intl', 
                              use_effective_order=True)
bleu_score = bleu_score.score