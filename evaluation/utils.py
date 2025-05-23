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
import openai
import re
from tqdm import tqdm

legal_category_dict = {
    "Criminal_Offenses": "Named criminal charges, offenses, or specific crime classifications.",
    "Weapons_Firearms": "Mentions of firearms, stun guns, ammunition, or other weapon references.",
    "Clothing_Appearance": "References to apparel, color descriptors, or identifying attire.",
    "Courts_Corrections": "Court names, correctional facilities, or related legal venues.",
    "Geography": "Locations, place names, or geographic regions (not specifically courts/facilities).",
    "People_Roles": "References to individuals, their titles, roles, or relationships (e.g. judges, defendants).",
    "Dates_Times": "Dates, years, or time spans relevant to the document.",
    "Sentences_Penalties": "Terms of imprisonment, probation, postrelease supervision, or other punishments.",
    "Legal_References": "Statutes, acts, or procedural/legal terms (not direct criminal charges).",
    "Miscellaneous": "Miscellaneous items that do not fit into the other categories."
}

model_mapping = {
    "25_llama_1_epoch": "Llama 3 8B\n(75% Permuted, 1 Epoch, ",
    "25_llama_10_epoch": "Llama 3 8B\n(75% Permuted, Conv., ",
    "50_llama_1_epoch": "Llama 3 8B\n(50% Permuted, 1 Epoch, ",
    "50_llama_10_epoch": "Llama 3 8B\n(50% Permuted, Conv., ",
    "75_llama_1_epoch": "Llama 3 8B\n(25% Permuted, 1 Epoch, ",
    "75_llama_10_epoch": "Llama 3 8B\n(25% Permuted, Conv., ",
    "llama_1_epoch": "Llama 3 8B\n(1 Epoch, ",
    "llama_10_epoch": "Llama 3 8B\n(Conv., ",
    "lora_llama_3b_10_epoch": "Llama 3 3B LoRA Finetuned\n(Conv., ",
    "lora_llama_10_epoch": "Llama 3 8B LoRA Finetuned\n(Conv., ",
    "mistral_10_epoch": "Mistral 7B\n(Conv., ",
    "qwen_1_epoch": "Qwen2 7B\n(1 Epoch, ",
    "qwen_10_epoch": "Qwen2 7B\n(Conv., "
}



openai.api_key = "{your_openai_key}"
MODEL_NAME = "gpt-4o-mini"  # Or e.g. "gpt-4"

def openai_prompt(message_history, text_only=True):
    try:
        response = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=message_history,
            temperature=0.7,
            max_tokens=2000,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )
        if text_only:
            return response.choices[0].message.content.strip()
        else:
            return response
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def analyze_legal_text(legal_text):
    # use llm to decide which categories apply to legal text (from legal_category_dict)
    # output in <Answer>...</Answer> as a comma-separated list

    category_list_str = "\n".join([f"{key}: {desc}" for key, desc in legal_category_dict.items()])
    system_instruction = (
        "You are an assistant that categorizes legally related text "
        "(i.e. text extracted from legal documents). you will receive "
        "a list of text snippets to analyze.\n\n"
        "You will then categorize that text, IN ORDER, "
        "into one following categories:\n\n"
        f"{category_list_str}\n\n"
        "Output your chain-of-thought reasoning normally. Then at the very end, enclose the "
        "final, comma-separated categories for the text snippets IN ORDER in <Answer>...</Answer>.\n"
        "Note that your list of categories should contain EXACTLY one category for each text snippet.\n"
        "Do NOT include additional commentary after the <Answer> block.\n"
    )

    user_message = f"List of legally related text snippets to analyze:\n\n{legal_text}\n"

    message_history = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ]

    model_response = openai_prompt(message_history, text_only=True)
    return model_response


def parse_categories(model_response: str):
    # categories from <Answer>...</Answer>
    answer_match = re.search(r"<Answer>(.*?)</Answer>", model_response, re.DOTALL)
    categories = []
    if answer_match:
        raw_categories = answer_match.group(1).strip()
        if raw_categories:
            # split on commas and strip whitespace
            categories = [c.strip() for c in raw_categories.split(",") if c.strip()]
    return categories

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

def json_to_dataframe(data, data_type='medical'):
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
    if data_type == 'law':
        df = pd.DataFrame(rows, columns=['legal_text', 'target', 'world_model', 'shadow_model', 'label'])
    elif data_type == 'medical':    
        df = pd.DataFrame(rows, columns=['condition', 'target', 'world_model', 'shadow_model', 'label'])
    return df

def _compute_df_hash(df):
    # stable hash
    df_sorted = df.sort_index(axis=1)
    row_hashes = pd.util.hash_pandas_object(df_sorted, index=True).values
    return hashlib.sha256(row_hashes.tobytes()).hexdigest()

def assign_ner_categories(df, model_name, data_type='medical'):
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
    
    hash_name = f"{data_type}_{model_name}"
    if hash_name in hashes and hashes[hash_name] == current_data_hash:
        cached_file = os.path.join(cache_dir, f"{data_type}_{model_name}_results.csv")
        if os.path.exists(cached_file):
            print(f"Loading cached results for model: {hash_name}")
            return pd.read_csv(cached_file)
    
    if data_type == 'medical':
        ner_name="d4data/biomedical-ner-all"
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

    elif data_type == 'law':
        assert 'legal_text' in df.columns, "df must have a 'legal_text' column"

        categories_col = []

        batch_size = 4
        for i in tqdm(range(0, len(df), batch_size), total=len(df) // batch_size):
            batch = df['legal_text'].iloc[i:i + batch_size].tolist()

            model_response = analyze_legal_text(batch)

            found_categories = parse_categories(model_response)

            if len(found_categories) != len(batch):
                print(f"Warning: Mismatch in batch size and found categories. Retrying the batch.")
                print(f"Batch: {batch}")
                print(f"Categories: {found_categories}")

                # retry 
                model_response = analyze_legal_text(batch)
                found_categories = parse_categories(model_response)

                # still mismatched, so "Missing" for entire batch, fix later
                if len(found_categories) != len(batch):
                    print(f"retry failed - assign 'Missing' to the entire batch")
                    found_categories = ["Missing"] * len(batch)

            categories_col.extend(found_categories)

        df['category'] = categories_col
    
    hashes[hash_name] = current_data_hash
    with open(hashes_file, "w") as f:
        json.dump(hashes, f)

    cached_file = os.path.join(cache_dir, f"{data_type}_{model_name}_results.csv")
    df.to_csv(cached_file)

    return df

def plot_overall_roc_old(fpr_model, tpr_model, roc_auc_model,
                     fpr_lr_attack, tpr_lr_attack, roc_auc_lr_attack,
                     fpr_prism, tpr_prism, roc_auc_prism,
                     title, filename, data_type='medical', figsize=(4, 4)):
    
    plt.style.use(['science'])
    plt.rc('text', usetex=False)
    plt.figure(figsize=figsize)

    plt.plot(fpr_model, tpr_model, color='blue', lw=1, 
             label=f'Classifier (auc = {roc_auc_model:.2f})')
    plt.plot(fpr_lr_attack, tpr_lr_attack, color='green', lw=1, linestyle='--', 
             label=f'LR-Attack (auc = {roc_auc_lr_attack:.2f})')
    plt.plot(fpr_prism, tpr_prism, color='red', lw=1, linestyle='-.', 
             label=f'PRISM (auc = {roc_auc_prism:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=0.5, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.title(f"{title}{data_type})", fontsize=12)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.savefig(filename, bbox_inches='tight')

def plot_overall_roc(fpr_model, tpr_model, roc_auc_model,
                     fpr_lr_attack, tpr_lr_attack, roc_auc_lr_attack,
                     fpr_prism, tpr_prism, roc_auc_prism,
                     title, filename, data_type='medical', figsize=(4, 4)):
    base_font = 12.5
    plt.style.use(['science'])
    plt.rc('text', usetex=False)
    plt.rc('font', size=base_font)
    plt.rc('axes', titlesize=base_font - 1)
    plt.rc('legend', fontsize=base_font - 2)

    plt.figure(figsize=figsize)
    epsilon = 1e-2

    fpr_model_plot = np.clip(fpr_model, epsilon, 1.0)
    fpr_lr_attack_plot = np.clip(fpr_lr_attack, epsilon, 1.0)
    fpr_prism_plot = np.clip(fpr_prism, epsilon, 1.0)

    plt.plot(fpr_model_plot, tpr_model, 'b-', lw=1.5, label=f'Classifier (AUC = {roc_auc_model:.2f})')
    plt.plot(fpr_lr_attack_plot, tpr_lr_attack, 'g--', lw=1.5, label=f'LR-Attack (AUC = {roc_auc_lr_attack:.2f})')
    plt.plot(fpr_prism_plot, tpr_prism, 'r-.', lw=1.5, label=f'PRISM (AUC = {roc_auc_prism:.2f})')

    plt.plot([epsilon, 1], [epsilon, 1], 'navy', lw=0.5, linestyle='--')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(epsilon, 1.0)
    plt.ylim(epsilon, 1.0)

    plt.xlabel('FPR', fontsize=base_font, labelpad=3)
    plt.ylabel('TPR', fontsize=base_font, labelpad=3)

    plt.title(f"{model_mapping[title]}{data_type})", fontsize=base_font + 1)

    plt.xticks([1e-2, 2e-2, 5e-2, 1e-1, 5e-1, 1.0], ['0.01', '0.02', '0.05', '0.1', '0.5', '1'])
    plt.yticks([1e-2, 2e-2, 5e-2, 1e-1, 5e-1, 1.0], ['', '0.02', '0.05', '0.1', '0.5', '1'], rotation=90, va='center')

    plt.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)

    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


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
                      data_type='medical',
                      figsize=(4, 4)):
    base_font = 12.5
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

    plt.style.use(['science'])
    plt.rc('text', usetex=False)
    plt.figure(figsize=figsize)

    epsilon = 1e-2  

    fpr_model_plot = np.clip(fpr_model, epsilon, 1.0)
    fpr_lr_attack_plot = np.clip(fpr_lr_attack, epsilon, 1.0)
    fpr_prism_plot = np.clip(fpr_prism, epsilon, 1.0)

    plt.plot(fpr_model_plot, tpr_model, color='blue', lw=1.5, 
             label=f'Classifier (AUC = {roc_auc_model:.2f})')
    plt.plot(fpr_lr_attack_plot, tpr_lr_attack, color='green', lw=1.5, linestyle='--', 
             label=f'LR-Attack (AUC = {roc_auc_lr_attack:.2f})')
    plt.plot(fpr_prism_plot, tpr_prism, color='red', lw=1.5, linestyle='-.', 
             label=f'PRISM (AUC = {roc_auc_prism:.2f})')

    plt.plot([epsilon, 1], [epsilon, 1], color='navy', lw=0.5, linestyle='--')

    plt.xscale('log')
    plt.yscale('log')

    plt.xlim(epsilon, 1.0)
    plt.ylim(epsilon, 1.0)

    plt.xlabel('FPR', fontsize=base_font, labelpad=3)
    plt.ylabel('TPR', fontsize=base_font, labelpad=3)
    plt.title(f"{category}\n#P={num_pos}/{n_cat} ({data_type})", fontsize=13)

    plt.xticks([1e-2, 2e-2, 5e-2, 1e-1, 5e-1, 1.0], ['0.01', '0.02', '0.05', '0.1', '0.5', '1'])
    plt.yticks([1e-2, 2e-2, 5e-2, 1e-1, 5e-1, 1.0], ['', '0.02', '0.05', '0.1', '0.5', '1'], rotation=90, va='center')

    plt.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.5)

    plt.legend(loc="lower right", fontsize=10.5)

    plt.tight_layout()
    
    filename = f"{save_path}/figures/{data_type}_{category}_attack_{model_name}.pdf"
    print(f"saving ROC plot for category '{category}' to: {filename}")
    plt.savefig(filename, bbox_inches='tight')
    plt.close()

score_mapping = {
    'y_pred_proba_model': {
        'label': 'Classifier',
        'color': 'blue',
        'linestyle': '-',
        'linewidth': 3
    },
    'lr_attack_scores': {
        'label': 'LR-Attack',
        'color': 'green',
        'linestyle': '--',
        'linewidth': 3
    },
    'prism_attack_scores': {
        'label': 'PRISM',
        'color': 'red',
        'linestyle': '-.',
        'linewidth': 3
    }
}

def plot_with_ents(filtered_results, num_ents):
    plt.style.use(['science'])
    plt.rc('text', usetex=False)  

    figsize = (6, 6)
    plt.figure(figsize=figsize)

    for score_col, style in score_mapping.items():
        fpr, tpr, _ = roc_curve(filtered_results['label'], filtered_results[score_col])
        auc_score = roc_auc_score(filtered_results['label'], filtered_results[score_col])
        
        plt.plot(
            fpr,
            tpr,
            color=style['color'],
            linestyle=style['linestyle'],
            linewidth=style['linewidth'],
            label=f"{style['label']} (AUC = {auc_score:.2f})"
        )

    plt.plot( [0, 1], [0, 1], color='navy', linestyle='--', linewidth=2)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    title = "roc curve"
    data_type = f"num_ents range = {num_ents}"
    percent_positive = filtered_results['label'].sum() / len(filtered_results)
    plt.title(f"{title} ({data_type}, {percent_positive:.2f} positive)", fontsize=11)
    plt.xlabel('FPR', fontsize=14)
    plt.ylabel('TPR', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True)
    plt.show()

    import numpy as np
import pandas as pd

def generate_main_results_table(
    # classifier
    classifier_tpr2,   # tuple of (llama, qwen, mistral)
    classifier_tpr10,  # tuple of (llama, qwen, mistral)
    classifier_roc,    # tuple of (llama, qwen, mistral)
    # lr-Attack
    lrattack_tpr2,
    lrattack_tpr10,
    lrattack_roc,
    # prism
    prism_tpr2,
    prism_tpr10,
    prism_roc,
    caption="Main Results",
    label="tab:main_results",
    # float_fmt=".2f",
    largest_is_better=True
):
    methods = ["Classifier", "LR-Attack", "PRISM"]
    cols = [
        "TPR2_Llama", "TPR2_Qwen", "TPR2_Mistral",
        "TPR10_Llama", "TPR10_Qwen", "TPR10_Mistral",
        "ROC_Llama", "ROC_Qwen", "ROC_Mistral"
    ]

    # convert tprs to percentages
    classifier_tpr2 = [x * 100 for x in classifier_tpr2]
    classifier_tpr10 = [x * 100 for x in classifier_tpr10]
    lrattack_tpr2 = [x * 100 for x in lrattack_tpr2]
    lrattack_tpr10 = [x * 100 for x in lrattack_tpr10]
    prism_tpr2 = [x * 100 for x in prism_tpr2]
    prism_tpr10 = [x * 100 for x in prism_tpr10]

    # clip each decimal in the tprs to 1 digit
    classifier_tpr2 = [round(x, 1) for x in classifier_tpr2]
    classifier_tpr10 = [round(x, 1) for x in classifier_tpr10]
    lrattack_tpr2 = [round(x, 1) for x in lrattack_tpr2]
    lrattack_tpr10 = [round(x, 1) for x in lrattack_tpr10]
    prism_tpr2 = [round(x, 1) for x in prism_tpr2]
    prism_tpr10 = [round(x, 1) for x in prism_tpr10]

    # round the roc aucs to 2 digits
    classifier_roc = [round(x, 2) for x in classifier_roc]
    lrattack_roc = [round(x, 2) for x in lrattack_roc]
    prism_roc = [round(x, 2) for x in prism_roc]

    print("classifier_tpr2", classifier_tpr2)

    data = [
        list(classifier_tpr2) + list(classifier_tpr10) + list(classifier_roc),
        list(lrattack_tpr2)   + list(lrattack_tpr10)   + list(lrattack_roc),
        list(prism_tpr2)      + list(prism_tpr10)      + list(prism_roc),
    ]

    df = pd.DataFrame(data, index=methods, columns=cols)

    ascending = not largest_is_better
    ranks = df.rank(method='dense', ascending=ascending)

    df_colored = df.copy()

    for c in cols:
        for row_idx in df.index:
            val = df.loc[row_idx, c]
            rank_val = ranks.loc[row_idx, c]
            val_str = f"{val}"

            if rank_val == 1:
                val_str = r"\cellcolor{gold!30} \textbf{" + val_str + r"}"
            elif rank_val == 2:
                val_str = r"\cellcolor{silver!30}" + val_str
            elif rank_val == 3:
                val_str = r"\cellcolor{bronze!30}" + val_str

            df_colored.loc[row_idx, c] = val_str

    prob_access = {
        "Classifier": [
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
        ],
        "LR-Attack": [
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
            r"\tikz\draw[black,fill=white] (0,0) circle (.5ex);",
        ],
        "PRISM": [
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
            r"\tikz\draw[black,fill=black] (0,0) circle (.5ex);",
        ],
    }

    lines = []
    lines.append(r"\begin{table}[ht!]")
    lines.append(r"    \centering")
    lines.append(r"    \small")
    lines.append(r"    \renewcommand*{\arraystretch}{1.15}")
    lines.append(r"    \begin{tabular}{ccccccccccccc}")
    lines.append(
        r"        Method &  \multicolumn{3}{c}{Prob. Access} & \multicolumn{3}{c}{TPR @ 2\% FPR} & \multicolumn{3}{c}{TPR @ 5\% FPR} & \multicolumn{3}{c}{ROC-AUC} \\ \cline{5-13}"
    )
    lines.append(
        r"        & {\normalsize{$\pdata$}} & {\normalsize{$\pshadow$}} & {\normalsize{$\pworld$}} & \scriptsize{Llama 3 8B} & \scriptsize{Qwen 2 7B} & \scriptsize{Mistral 7B} & \scriptsize{Llama 3 8B} & \scriptsize{Qwen 2 7B} & \scriptsize{Mistral 7B} & \scriptsize{Llama 3 8B} & \scriptsize{Qwen 2 7B} & \scriptsize{Mistral 7B} \\ \hline"
    )

    for method in methods:
        # circles
        c1, c2, c3 = prob_access[method]

        # numeric columns from df_colored
        #   TPR2_Llama, TPR2_Qwen, TPR2_Mistral,
        #   TPR10_Llama, TPR10_Qwen, TPR10_Mistral,
        #   ROC_Llama,  ROC_Qwen,   ROC_Mistral
        row_vals = [
            df_colored.loc[method, "TPR2_Llama"],
            df_colored.loc[method, "TPR2_Qwen"],
            df_colored.loc[method, "TPR2_Mistral"],
            df_colored.loc[method, "TPR10_Llama"],
            df_colored.loc[method, "TPR10_Qwen"],
            df_colored.loc[method, "TPR10_Mistral"],
            df_colored.loc[method, "ROC_Llama"],
            df_colored.loc[method, "ROC_Qwen"],
            df_colored.loc[method, "ROC_Mistral"],
        ]

        line = (
            f"        {method} & {c1} & {c2} & {c3} & "
            f"{row_vals[0]}\% & {row_vals[1]}\% & {row_vals[2]}\% & "
            f"{row_vals[3]}\% & {row_vals[4]}\% & {row_vals[5]}\% & "
            f"{row_vals[6]} & {row_vals[7]} & {row_vals[8]} \\\\"
        )
        lines.append(line)

    lines.append(r"        \hline")
    lines.append(r"    \end{tabular}")
    lines.append(r"    \label{" + label + r"}")
    lines.append(r"    \caption{" + caption + r"}")
    lines.append(r"\end{table}")

    latex_code = "\n".join(lines)
    return latex_code
