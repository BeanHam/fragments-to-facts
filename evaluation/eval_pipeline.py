from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

from utils import *

SMALL_CONSTANT = 1e-12

def run_evaluation_pipeline(X, y, model=None, cv=None):
    if model is None:
        model = LGBMClassifier(
            objective='binary',
            random_state=42,
            verbose=-1
        )

    if cv is None:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = {}

    # cross-validation acc and f1
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
    results['cv_accuracy_mean'] = accuracy_scores.mean()
    results['cv_accuracy_std'] = accuracy_scores.std()
    results['cv_f1_mean'] = f1_scores.mean()
    results['cv_f1_std'] = f1_scores.std()
    print(f"Accuracy: {results['cv_accuracy_mean']} +/- {results['cv_accuracy_std']}")
    print(f"F1: {results['cv_f1_mean']} +/- {results['cv_f1_std']}")

    # cv pred probs and labels for the model
    y_pred_proba_model = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    y_pred_model = cross_val_predict(model, X, y, cv=cv, method='predict')
    results['y_pred_proba_model'] = y_pred_proba_model
    results['y_pred_model'] = y_pred_model

    results['roc_auc_model'] = roc_auc_score(y, y_pred_proba_model)
    print(f"ROC AUC (model): {results['roc_auc_model']}")

    # our approach scores
    lr_attack_scores = np.zeros(len(y))
    prism_attack_scores = np.zeros(len(y))

    # manual cv loop for computing lr_attack and prism scores
    for train_idx, test_idx in cv.split(X, y):
        _, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        _, _ = y.iloc[train_idx], y.iloc[test_idx]

        prob_target = X_test_cv['target']
        prob_shadow = X_test_cv['shadow_model']
        prob_worlds = X_test_cv['world_model']

        # lr for fold
        likelihood_ratio_cv = prob_target / (prob_shadow + SMALL_CONSTANT)  # add small epsilon to avoid div by zero

        # lr_attack_score (negated likelihood ratio as per original comment)
        lr_attack_score = likelihood_ratio_cv

        p_s_in_D_cv = likelihood_ratio_to_probability(likelihood_ratio_cv)
        prism_score_cv = (prob_worlds - prob_shadow * (1 - p_s_in_D_cv)) / (p_s_in_D_cv + SMALL_CONSTANT)

        lr_attack_scores[test_idx] = lr_attack_score
        prism_attack_scores[test_idx] = prism_score_cv

    # handle nan or infinite values (shouldn't happen)
    lr_attack_scores = np.nan_to_num(lr_attack_scores)
    prism_attack_scores = np.nan_to_num(prism_attack_scores)

    results['lr_attack_scores'] = lr_attack_scores
    results['prism_attack_scores'] = prism_attack_scores
    results['roc_auc_lr_attack'] = roc_auc_score(y, lr_attack_scores)
    results['roc_auc_prism'] = roc_auc_score(y, prism_attack_scores)

    fpr_model, tpr_model, _ = roc_curve(y, y_pred_proba_model)
    fpr_lr_attack, tpr_lr_attack, _ = roc_curve(y, lr_attack_scores)
    fpr_prism, tpr_prism, _ = roc_curve(y, prism_attack_scores)

    results['roc_curve'] = {
        'model': (fpr_model, tpr_model),
        'lr_attack': (fpr_lr_attack, tpr_lr_attack),
        'prism': (fpr_prism, tpr_prism)
    }

    return results, model

def main_runner(model_name, save_path, data_path, prompt_id=2):
    with open(f"{data_path}/probs/{model_name}_world_probs_prompt_{prompt_id}.json") as f:
        data = json.load(f)

        results = json_to_dataframe(data, data_type=data_path)
        results = assign_ner_categories(results, model_name, data_type=data_path)

        X = results[['target', 'world_model', 'shadow_model']]
        y = results['label']

        eval_results, _ = run_evaluation_pipeline(X, y)
        print(eval_results)

        results['y_pred_model'] = eval_results['y_pred_model']
        results['y_pred_proba_model'] = eval_results['y_pred_proba_model']
        results['lr_attack_scores'] = eval_results['lr_attack_scores']
        results['prism_attack_scores'] = eval_results['prism_attack_scores']

        plot_overall_roc(fpr_model = eval_results['roc_curve']['model'][0], 
                         tpr_model = eval_results['roc_curve']['model'][1], 
                         roc_auc_model = eval_results['roc_auc_model'],
                         fpr_lr_attack = eval_results['roc_curve']['lr_attack'][0], 
                         tpr_lr_attack = eval_results['roc_curve']['lr_attack'][1], 
                         roc_auc_lr_attack = eval_results['roc_auc_lr_attack'],
                         fpr_prism = eval_results['roc_curve']['prism'][0], 
                         tpr_prism = eval_results['roc_curve']['prism'][1], 
                         roc_auc_prism = eval_results['roc_auc_prism'],
                         title=f'{model_name}',
                         filename=f"{save_path}/figures/{data_path}_{model_name}_overall.pdf",
                         data_type=data_path)

        # NOTE: currently broken with the caching, though not super important
        # plot_feature_importance(model, 
        #                         title='Feature Importance', 
        #                         savefig=f"{save_path}/figures/{model_name}_feature_importance.pdf")

        category_counts = results['category'].value_counts()
        categories_to_analyze = category_counts[category_counts > 50].index.tolist()

        print("\nCats with more than 50 examples.\n")
        print(categories_to_analyze)

        category_metrics = compute_category_metrics(results, categories_to_analyze,
                                                    results['y_pred_model'],
                                                    results['y_pred_proba_model'],
                                                    results['lr_attack_scores'],
                                                    results['prism_attack_scores'])
        
        metrics_df = pd.DataFrame(category_metrics)

        print("\nPer cat summary.\n")
        print(metrics_df)

        for category in categories_to_analyze:
            print(f"Plotting {category}")
            plot_category_roc(results, 
                              category,
                              results['y_pred_proba_model'],
                              results['lr_attack_scores'],
                              results['prism_attack_scores'],
                              model_name,
                              save_path=save_path,
                              data_type=data_path)
        
        # we return the metrics for the model we just evaled
        def tpr_at_fpr(fpr, tpr, target_fpr):
            return np.interp(target_fpr, fpr, tpr)

        fpr_model, tpr_model = eval_results['roc_curve']['model']
        classifier_tpr2  = tpr_at_fpr(fpr_model, tpr_model, 0.02)
        classifier_tpr10 = tpr_at_fpr(fpr_model, tpr_model, 0.10)
        classifier_roc   = eval_results['roc_auc_model']

        fpr_lr, tpr_lr = eval_results['roc_curve']['lr_attack']
        lr_tpr2  = tpr_at_fpr(fpr_lr, tpr_lr, 0.02)
        lr_tpr10 = tpr_at_fpr(fpr_lr, tpr_lr, 0.10)
        lr_roc   = eval_results['roc_auc_lr_attack']

        fpr_prism, tpr_prism = eval_results['roc_curve']['prism']
        prism_tpr2  = tpr_at_fpr(fpr_prism, tpr_prism, 0.02)
        prism_tpr10 = tpr_at_fpr(fpr_prism, tpr_prism, 0.10)
        prism_roc   = eval_results['roc_auc_prism']

        return {
            'classifier_tpr2': classifier_tpr2,
            'classifier_tpr10': classifier_tpr10,
            'classifier_roc': classifier_roc,
            'lr_tpr2': lr_tpr2,
            'lr_tpr10': lr_tpr10,
            'lr_roc': lr_roc,
            'prism_tpr2': prism_tpr2,
            'prism_tpr10': prism_tpr10,
            'prism_roc': prism_roc
        }

            

if __name__ == '__main__':

    # 10 epoch: 'llama','qwen', 'mistral', '75_llama', '50_llama', '25_llama', 'lora_llama', 'lora_llama_3b', 
    # 1 epoch: 'llama','qwen', '75_llama', '50_llama', '25_llama'
    # law: 'llama','qwen',

    # law epoch
    models = ['llama', 'qwen', 'mistral']
    data_paths = ['medical']
    epochs = [10]
    split = 2

    # results dict for the generate_main_results_table
    # TODO: fix this and the next loop so that it stores results correctly
    model_res_dict = {}

    for model in models:
        for epoch in epochs:
            for data_path in data_paths:
                model_name = f"{model}_{epoch}_epoch"
                print(f"Running {model_name} on {data_path}")

                save_path = "results/"+model_name

                # make sure directory exists for save_path
                os.makedirs(save_path+'/figures', exist_ok=True)

                # model_results_for_main_table = main_runner(model_name, save_path, data_path)
                model_res_dict[model_name] = main_runner(model_name, save_path, data_path)

                print(f"Finished {model_name} on {data_path}")
                print("\n\n")

    llama_key = 'llama_10_epoch'
    qwen_key = 'qwen_10_epoch'
    mistral_key = 'mistral_10_epoch'

    if all(k in model_res_dict for k in [llama_key, qwen_key, mistral_key]):
        latex_code = generate_main_results_table(
            # classifier
            classifier_tpr2=(
                model_res_dict[llama_key]['classifier_tpr2'],
                model_res_dict[qwen_key]['classifier_tpr2'],
                model_res_dict[mistral_key]['classifier_tpr2']
            ),
            classifier_tpr10=(
                model_res_dict[llama_key]['classifier_tpr10'],
                model_res_dict[qwen_key]['classifier_tpr10'],
                model_res_dict[mistral_key]['classifier_tpr10']
            ),
            classifier_roc=(
                model_res_dict[llama_key]['classifier_roc'],
                model_res_dict[qwen_key]['classifier_roc'],
                model_res_dict[mistral_key]['classifier_roc']
            ),

            # lr attack
            lrattack_tpr2=(
                model_res_dict[llama_key]['lr_tpr2'],
                model_res_dict[qwen_key]['lr_tpr2'],
                model_res_dict[mistral_key]['lr_tpr2']
            ),
            lrattack_tpr10=(
                model_res_dict[llama_key]['lr_tpr10'],
                model_res_dict[qwen_key]['lr_tpr10'],
                model_res_dict[mistral_key]['lr_tpr10']
            ),
            lrattack_roc=(
                model_res_dict[llama_key]['lr_roc'],
                model_res_dict[qwen_key]['lr_roc'],
                model_res_dict[mistral_key]['lr_roc']
            ),

            # prism
            prism_tpr2=(
                model_res_dict[llama_key]['prism_tpr2'],
                model_res_dict[qwen_key]['prism_tpr2'],
                model_res_dict[mistral_key]['prism_tpr2']
            ),
            prism_tpr10=(
                model_res_dict[llama_key]['prism_tpr10'],
                model_res_dict[qwen_key]['prism_tpr10'],
                model_res_dict[mistral_key]['prism_tpr10']
            ),
            prism_roc=(
                model_res_dict[llama_key]['prism_roc'],
                model_res_dict[qwen_key]['prism_roc'],
                model_res_dict[mistral_key]['prism_roc']
            ),

            caption="Main Results Across Models (Llama, Qwen, Mistral)",
            label="tab:main_results_all"
        )

        print("\n gen latex table:\n")
        print(latex_code)
    else:
        print("not all {llama, qwen, mistral} results are present")
    

