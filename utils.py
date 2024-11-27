import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

MODEL_DICT = {
    'llama_3_1': 'meta-llama/Meta-Llama-3.1-8B-Reference',
    'llama_3_1_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
    'llama_3': 'meta-llama/Meta-Llama-3-8B',
    'llama_3_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama_2': 'togethercomputer/llama-2-7b',
    'llama_2_chat': 'togethercomputer/llama-2-7b-chat',
    'codellama': 'codellama/CodeLlama-7b-hf',
    'codellama_python': 'codellama/CodeLlama-7b-Python-hf',
    'codellama_instruct': 'codellama/CodeLlama-7b-Instruct-hf',
    'mixtral_8x7b': 'mistralai/Mixtral-8x7B-v0.1',
    'mixtral_8x7b_instruct': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'nous_hermes_2_mixtral_8x7b_dpo': 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
    'nous_hermes_2_mixtral_8x7b_sft': 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT',
    'mistral_7b_instruct': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral_7b': 'mistralai/Mistral-7B-v0.1',
    'qwen2': 'Qwen/Qwen2-1.5B',
    'qwen2_instruct': 'Qwen/Qwen2-1.5B-Instruct',
    'openhermes_2_5_mistral_7b': 'teknium/OpenHermes-2p5-Mistral-7B',
    'zephyr_7b': 'HuggingFaceH4/zephyr-7b-beta',
    'solar_instruct_v1': 'upstage/SOLAR-10.7B-Instruct-v1.0',
}


def likelihood_ratio_to_probability(likelihood_ratio, prior_in=0.5):
    
    """
    Function to convert from likelihood ratio to probability.
    
    Input:
        - likelihood ratio = p_in/p_out. 
            p_in is the target sequence probability from the target model.
            p_out is the target sequence probability from the shadow model.
                  
    Output:
        - probability
    
    """
    prior_out = 1 - prior_in

    # compute the marginal likelihood
    marginal_likelihood = (likelihood_ratio * prior_in) + (1 / likelihood_ratio * prior_out)

    # bayes to get the posterior probability of membership
    posterior_in = (likelihood_ratio * prior_in) / marginal_likelihood

    return posterior_in


def auc_calculation(labels, probs):
    fpr,tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr,tpr)
    return roc_auc