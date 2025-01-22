import json
import argparse
from utils import *
from os import path
from tqdm import tqdm
from together import Together

def add_world_model_probs_single(value, token_label, prompt, client, world_model_endpoints, is_star=True):
    data_key = 'y_stars' if is_star else 'y_NON_stars'
    
    if 'ents_prob' in value[data_key][token_label]:
        remaining_ents = list(value[data_key][token_label]['ents_prob'].keys())
    else:
        remaining_ents = []

    if 'world_models' not in value[data_key][token_label]:
        value[data_key][token_label]['world_models'] = []
    if 'world_ents_prob' not in value[data_key][token_label]:
        value[data_key][token_label]['world_ents_prob'] = []
        
    #for wm_endpoint in tqdm(world_model_endpoints, desc="world model endpoints", leave=False):
    for wm_endpoint in world_model_endpoints:
        wm_prob_gen = GenerateNextTokenProbAPI(client, wm_endpoint)
        max_tokens=len(wm_prob_gen.tokenizer(prompt)['input_ids'])+10
        # wm_prob = compute_token_probs_api(token_label, prompt, wm_prob_gen, max_tokens)
        wm_result = wm_prob_gen.get_token_probs(
            prompt=prompt,
            target_string=token_label,
            remaining_ents=remaining_ents,
            max_tokens=max_tokens
        )

        if wm_result == -1:
            # if the target span wasn't found
            target_prob = -1.0
            ents_prob = {}
        else:
            target_prob = float(wm_result["target_prob"])
            ents_prob = {ent: float(prob) for ent, prob in wm_result["ents_prob"].items()}

        # value[data_key][token_label]['world_models'].append(float(wm_prob))
        value[data_key][token_label]['world_models'].append(target_prob)
        value[data_key][token_label]['world_ents_prob'].append(ents_prob)


def add_model_probs(results, client, world_model_endpoints, shadow_model_endpoints, model_type='world'):
    for t_id, value in tqdm(results.items(), desc="processing results"):
        print(f"processing {t_id}")
        y_stars_order = list(value['y_stars'].keys())
        y_non_stars_order = list(value['y_NON_stars'].keys())

        # y_stars 
        for y_star in tqdm(y_stars_order, leave=True, desc="processing y_stars"):
            prompt = value['y_stars'][y_star]['prompt']
            if model_type == 'world':
                add_world_model_probs_single(value, y_star, prompt, client, world_model_endpoints, is_star=True)
            else:
                add_shadow_model_probs_single(value, y_star, prompt, client, shadow_model_endpoints, is_star=True)

        # y_non_stars 
        for y_non_star in tqdm(y_non_stars_order, leave=True, desc="processing y_non_stars"):
            prompt = value['y_NON_stars'][y_non_star]['prompt']
            if model_type == 'world':
                add_world_model_probs_single(value, y_non_star, prompt, client, world_model_endpoints, is_star=False)
            else:
                add_shadow_model_probs_single(value, y_non_star, prompt, client, shadow_model_endpoints, is_star=False)


#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # Parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='probs/')
    parser.add_argument('--model_tag', type=str, default='llama_1_epoch')
    parser.add_argument('--ablation_pct', type=str, default='1.0')
    parser.add_argument('--together_key', type=str)
    args = parser.parse_args()

    ablation_str = int(float(args.ablation_pct)*100)

    ## log in together ai & hugginface
    if ablation_str != 100:
        with open(path.join(args.save_dir, f'{args.model_tag}_shadow_probs_prompt_{PROMPT_TO_USE}_{ablation_str}.json'), 'r') as f:
            all_probs = json.load(f)
    else:
        with open(path.join(args.save_dir, f'{args.model_tag}_shadow_probs_prompt_{PROMPT_TO_USE}.json'), 'r') as f:
            all_probs = json.load(f)
    client = Together(api_key=args.together_key)
    world_model_endpoints = [
        "google/gemma-2b-it",
        "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "mistralai/Mistral-7B-Instruct-v0.2"
    ]
    
    print('Add World Probs...')
    add_model_probs(all_probs, client, world_model_endpoints, [], model_type='world')

    ## save results
    print('Save Results...')
    if ablation_str != 100:
        with open(path.join(args.save_dir, f'{args.model_tag}_world_probs_prompt_{PROMPT_TO_USE}_{ablation_str}.json'), 'w') as f:
            json.dump(all_probs, f)
    else:
        with open(path.join(args.save_dir, f'{args.model_tag}_world_probs_prompt_{PROMPT_TO_USE}.json'), 'w') as f:
            json.dump(all_probs, f)

if __name__ == "__main__":
    main()