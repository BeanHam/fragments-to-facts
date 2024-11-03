import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc


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