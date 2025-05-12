import json
import argparse
from os import path
from tqdm import tqdm
from together import Together
from utils import *
from huggingface_hub import login as hf_login

def add_shadow_model_probs_single(value, token_label, prompt, client, shadow_model_endpoints, is_star=True):
    data_key = 'y_stars' if is_star else 'y_NON_stars'

    if 'ents_prob' in value[data_key][token_label]:
        remaining_ents = list(value[data_key][token_label]['ents_prob'].keys())
    else:
        remaining_ents = []
    
    if 'shadow_models' not in value[data_key][token_label]:
        value[data_key][token_label]['shadow_models'] = []
    if 'shadow_ents_prob' not in value[data_key][token_label]:
        value[data_key][token_label]['shadow_ents_prob'] = []
        
    #for sm_endpoint in tqdm(shadow_model_endpoints, desc="shadow model endpoints", leave=False):
    for sm_endpoint in shadow_model_endpoints:
        sm_prob_gen = GenerateNextTokenProbAPI(client, sm_endpoint)
        max_tokens=len(sm_prob_gen.tokenizer(prompt)['input_ids'])+10
        # sm_prob = compute_token_probs_api(token_label, prompt, sm_prob_gen, max_tokens)

        sm_result = sm_prob_gen.get_token_probs(
            prompt=prompt,
            target_string=token_label,
            remaining_ents=remaining_ents,
            max_tokens=max_tokens
        )

        if sm_result == -1:
            target_prob = -1.0
            ents_prob = {}
        else:
            target_prob = float(sm_result["target_prob"])
            ents_prob = {ent: float(prob) for ent, prob in sm_result["ents_prob"].items()}

        # value[data_key][token_label]['shadow_models'].append(float(sm_prob))
        value[data_key][token_label]['shadow_models'].append(target_prob)
        value[data_key][token_label]['shadow_ents_prob'].append(ents_prob)


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
    parser.add_argument('--hf_key', type=str, default='')
    parser.add_argument('--together_key', type=str)
    args = parser.parse_args()

    ablation_str = int(float(args.ablation_pct)*100)
    
    ## log in together ai & hugginface
    if args.hf_key:
        hf_login(token=args.hf_key, add_to_git_credential=True)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', token=args.hf_key)

    with open('model_map.json') as f:
        model_map=json.load(f)
    if ablation_str != 100:
        with open(path.join(args.save_dir, f'{args.model_tag}_target_probs_prompt_{PROMPT_TO_USE}_{ablation_str}.json'), 'r') as f:
            all_probs = json.load(f)
    else:
        with open(path.join(args.save_dir, f'{args.model_tag}_target_probs_prompt_{PROMPT_TO_USE}.json'), 'r') as f:
            all_probs = json.load(f)
    client = Together(api_key=args.together_key)
    shadow_model_endpoints = [model_map[args.model_tag]['shadow']['api_key']]
    input(f"""Please deploy the following model {shadow_model_endpoints}. The deployment might take up to 10 mins...""")
    
    print('Add Shadow Probs...')
    add_model_probs(all_probs, client, [], shadow_model_endpoints, model_type='shadow')

    ## save results
    print('Save Results...')
    if ablation_str != 100:
        with open(path.join(args.save_dir, f'{args.model_tag}_shadow_probs_prompt_{PROMPT_TO_USE}_{ablation_str}.json'), 'w') as f:
            json.dump(all_probs, f)
    else:
        with open(path.join(args.save_dir, f'{args.model_tag}_shadow_probs_prompt_{PROMPT_TO_USE}.json'), 'w') as f:
            json.dump(all_probs, f)

if __name__ == "__main__":
    main()