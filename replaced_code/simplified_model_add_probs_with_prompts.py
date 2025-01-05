import json
from tqdm import tqdm
from together import Together
from utils import *

def add_world_model_probs_single(value, token_label, prompt, client, world_model_endpoints, is_star=True):
    data_key = 'y_stars' if is_star else 'y_NON_stars'
    
    if 'world_models' not in value[data_key][token_label]:
        value[data_key][token_label]['world_models'] = []
        
    #for wm_endpoint in tqdm(world_model_endpoints, desc="world model endpoints", leave=False):
    for wm_endpoint in world_model_endpoints:
        wm_prob_gen = GenerateNextTokenProbAPI(client, wm_endpoint)
        max_tokens=len(wm_prob_gen.tokenizer(prompt)['input_ids'])+10
        wm_prob = compute_token_probs_api(token_label, prompt, wm_prob_gen, max_tokens)
        value[data_key][token_label]['world_models'].append(float(wm_prob))

def add_shadow_model_probs_single(value, token_label, prompt, client, shadow_model_endpoints, is_star=True):
    data_key = 'y_stars' if is_star else 'y_NON_stars'
    
    if 'shadow_models' not in value[data_key][token_label]:
        value[data_key][token_label]['shadow_models'] = []
        
    #for sm_endpoint in tqdm(shadow_model_endpoints, desc="shadow model endpoints", leave=False):
    for sm_endpoint in shadow_model_endpoints:
        sm_prob_gen = GenerateNextTokenProbAPI(client, sm_endpoint)
        max_tokens=len(sm_prob_gen.tokenizer(prompt)['input_ids'])+10
        sm_prob = compute_token_probs_api(token_label, prompt, sm_prob_gen, max_tokens)
        value[data_key][token_label]['shadow_models'].append(float(sm_prob))

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
            

id=2
with open('model_map.json') as f:
    model_map=json.load(f)
with open(f'target_token_probs_train_{id}_10_epochs_with_prompt_higher.json', 'r') as f:
    all_model_probs = json.load(f)    
key = "..."
client = Together(api_key=key)
shadow_model_endpoints=[model_map[f'shadow_train_{id}']['api_key']]
world_model_endpoints = [
    "google/gemma-2b-it",
    'meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo',
    'mistralai/Mistral-7B-Instruct-v0.2'
]
add_model_probs(all_model_probs, client, [], shadow_model_endpoints, model_type='shadow')
add_model_probs(all_model_probs, client, world_model_endpoints, [], model_type='world')
with open(f'target_token_probs_train_{id}_10_epochs_with_prompt_higher_with_shadow_with_world.json', 'w') as f:
    json.dump(all_model_probs, f)