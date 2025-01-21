import json
import os
import pandas as pd
import numpy as np
from typing import Any, Dict, List
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_auc_score, auc, roc_curve, accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import scienceplots
from transformers import pipeline
import lightgbm
import matplotlib.pyplot as plt
import hashlib

def likelihood_ratio_to_probability(likelihood_ratio, prior_in=0.9):
    # NOTE: a default prior of 0.5 means essentially no prior knowledge
    # on the probability of membership in the target class
    # NOTE: we get good performance with prior of 0.9, and bad 
    # performance with prior of 0.1 - why? probably because a high prior
    # on token is a member of the target class is a good assumption
    # if the likelihood ratio is high, then the token is likely to be a member
    
    prior_out = 1 - prior_in

    epsilon = 1e-12
    likelihood_ratio = np.clip(likelihood_ratio, epsilon, None)
    inverse_lr = 1 / likelihood_ratio

    # marginal likelihood
    marginal_likelihood = (likelihood_ratio * prior_in) + (inverse_lr * prior_out)

    # get the posterior probability of membership
    posterior_in = (likelihood_ratio * prior_in) / marginal_likelihood

    return posterior_in

def json_to_dataframe(data):
    rows = []

    for top_key, sample in data.items():
        is_train = top_key.startswith("train")

        def process_category(category, label):
            if category in sample:
                for condition, metrics in sample[category].items():
                    target = metrics.get('target_prob', None)

                    world_models = metrics.get('world_models', [])
                    if world_models:
                        avg_world = sum(world_models) / len(world_models)
                    else:
                        avg_world = None 

                    shadow_models = metrics.get('shadow_models', [])
                    if shadow_models:
                        avg_shadow = sum(shadow_models) / len(shadow_models)
                    else:
                        avg_shadow = None  

                    rows.append([condition, target, avg_world, avg_shadow, label])

        if is_train:
            process_category('y_stars', 1)
        else:
            process_category('y_stars', 0)

        process_category('y_NON_stars', 0)

    df = pd.DataFrame(rows, columns=['condition', 'target', 'world_model', 'shadow_model', 'label'])
    return df

def _compute_df_hash(df):
    # stable hash
    df_sorted = df.sort_index(axis=1)
    row_hashes = pd.util.hash_pandas_object(df_sorted, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()

def assign_ner_categories(df, model_name, ner_name="d4data/biomedical-ner-all"):
    # assigns NER categories to each row in the DataFrame based on the 'condition' field
    
    # check to see if this has been run for that same model, same dataset, before
    # do so by checking the "cache" folder, which contains a file of "hashes.json"
    # that map from model name to a hash of the data used to train the NER model
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    hashes_file = os.path.join(cache_dir, "hashes.json")
    if os.path.exists(hashes_file):
        with open(hashes_file, "r") as f:
            hashes = json.load(f)
    else:
        hashes = {}

    current_data_hash = _compute_df_hash(df)

    if model_name in hashes and hashes[model_name] == current_data_hash:
        cached_file = os.path.join(cache_dir, f"{model_name}_results.csv")
        if os.path.exists(cached_file):
            print(f"Loading cached results for model: {model_name}")
            return pd.read_csv(cached_file)
    
    ner_pipeline = pipeline("ner", model=ner_name)
    
    categories = []

    assert 'condition' in df.columns, "df must have a 'condition' column"

    for condition in df['condition']:
        ner_results = ner_pipeline(condition)
        if ner_results:
            entity = ner_results[0]['entity']
            categories.append(entity)
        else:
            categories.append('other')
    
    df['category'] = categories

    hashes[model_name] = current_data_hash
    with open(hashes_file, "w") as f:
        json.dump(hashes, f)

    cached_file = os.path.join(cache_dir, f"{model_name}_results.csv")
    df.to_csv(cached_file)

    return df

def plot_overall_roc(fpr_model, tpr_model, roc_auc_model,
                     fpr_lr_attack, tpr_lr_attack, roc_auc_lr_attack,
                     fpr_prism, tpr_prism, roc_auc_prism,
                     title, filename, figsize=(4, 4)):
    
    plt.style.use(['science'])
    plt.rc('text', usetex=False)
    plt.figure(figsize=figsize)

    plt.plot(fpr_model, tpr_model, color='blue', lw=3, 
             label=f'Classifier (auc = {roc_auc_model:.2f})')
    plt.plot(fpr_lr_attack, tpr_lr_attack, color='green', lw=3, linestyle='--', 
             label=f'LR-Attack (auc = {roc_auc_lr_attack:.2f})')
    plt.plot(fpr_prism, tpr_prism, color='red', lw=3, linestyle='-.', 
             label=f'PRISM (auc = {roc_auc_prism:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')

def plot_feature_importance(model, title='Feature Importance', figsize=(8, 6), savefig=None):
    plt.figure(figsize=figsize)
    lightgbm.plot_importance(model)
    plt.title(title)
    if savefig:
        plt.savefig(savefig, bbox_inches='tight')

def compute_category_metrics(df_results, categories_to_analyze, y_pred_model,
                             y_pred_proba_model, lr_attack_scores, prism_attack_scores):

    category_metrics = []

    for category in categories_to_analyze:
        for scores, score_name in zip([y_pred_proba_model, lr_attack_scores, prism_attack_scores],
                          ['model', 'lr_attack', 'prism']):
            print(f"category: {category}, score: {score_name}")
            df_cat = df_results[df_results['category'] == category]

            if df_cat['label'].nunique() < 2:
                print(f"skip category '{category}' - needs both classes represented")
                continue

            y_true_cat = df_cat['label']

            y_pred_proba_cat = scores[df_cat.index]

            roc_auc = roc_auc_score(y_true_cat, y_pred_proba_cat)

            category_metrics.append({
                'category': category,
                'roc auc': roc_auc
            })

            print(f"auc for category: {category}")

    return category_metrics

def plot_category_roc(df_results, category, y_pred_proba_model, 
                      lr_attack_scores, prism_attack_scores, model_name, save_path,
                      figsize=(4,4)):
    
    df_cat = df_results[df_results['category'] == category]
    if df_cat['label'].nunique() < 2:
        print(f"skip roc plot for category '{category}'")
        return

    y_true = df_cat['label']
    n_cat = df_cat.shape[0]
    num_pos = df_cat['label'].sum()

    idx = df_cat.index
    y_pred_proba_model_cat = y_pred_proba_model[idx]
    lr_attack_scores_cat = lr_attack_scores[idx]
    prism_attack_scores_cat = prism_attack_scores[idx]

    fpr_model, tpr_model, _ = roc_curve(y_true, y_pred_proba_model_cat)
    fpr_lr_attack, tpr_lr_attack, _ = roc_curve(y_true, lr_attack_scores_cat)
    fpr_prism, tpr_prism, _ = roc_curve(y_true, prism_attack_scores_cat)

    roc_auc_model = roc_auc_score(y_true, y_pred_proba_model_cat)
    roc_auc_lr_attack = roc_auc_score(y_true, lr_attack_scores_cat)
    roc_auc_prism = roc_auc_score(y_true, prism_attack_scores_cat)

    plt.figure(figsize=figsize)
    plt.plot(fpr_model, tpr_model, color='blue', lw=3, label=f'Classifier (auc = {roc_auc_model:.2f})')
    plt.plot(fpr_lr_attack, tpr_lr_attack, color='green', lw=3, linestyle='--', label=f'LR-Attack (auc = {roc_auc_lr_attack:.2f})')
    plt.plot(fpr_prism, tpr_prism, color='red', lw=3, linestyle='-.', label=f'PRISM (auc = {roc_auc_prism:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.title(f'{category}, P={num_pos}/{n_cat}', fontsize=13)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.tight_layout()

    filename = save_path + f"/figures/{category}_attack_{model_name}.pdf"
    print(f"saving roc plot for category '{category}' to: {filename}")
    plt.savefig(filename, bbox_inches='tight')
