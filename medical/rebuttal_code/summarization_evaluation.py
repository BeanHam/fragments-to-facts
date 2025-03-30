#!pip install evaluate
#!pip install absl-py
#!pip install rouge-score
#!pip install bert-score

import evaluate
import numpy as np

def compute_summarization_metrics(predictions, references) -> dict:
    """
    Compute ROUGE, BLEU, and BERTscore metrics for a set of predictions and references.
    """

    ## load metrics
    metric_results = {}
    rouge = evaluate.load('rouge')
    bleu = evaluate.load('bleu')
    bertscore = evaluate.load('bertscore')

    ## calculate metrics
    rouge_results = rouge.compute(predictions=predictions, 
                                  references=references, 
                                  use_aggregator=True)
    bleu_results = bleu.compute(predictions=predictions, 
                                references=references)
    bertscore_results = bertscore.compute(predictions=predictions, 
                                          references=references, 
                                          lang='en', 
                                          model_type="distilbert-base-uncased")

    ## save metrics
    metric_results['rouge'] = rouge_results  
    metric_results['bleu'] = bleu_results
    metric_results['bertscore'] = {k: np.mean(v) for k, v in bertscore_results.items() if k in ['precision', 'recall', 'f1']}
    
    return metric_results