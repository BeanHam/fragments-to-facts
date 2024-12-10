import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from together import Together
from transformers import AutoTokenizer

from utils import *
from huggingface_hub import login as hf_login
from peft import prepare_model_for_kbit_training
from datasets import concatenate_datasets, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel

PROMPT_TO_USE = 0
def add_world_model_probs(value, y_star, y_non_star, prompt, client, world_model_endpoints):
    if 'world_models' not in value['y_stars'][y_star]:
        value['y_stars'][y_star]['world_models'] = []
    if 'world_models' not in value['y_NON_stars'][y_non_star]:
        value['y_NON_stars'][y_non_star]['world_models'] = []
        
    for wm_endpoint in tqdm(world_model_endpoints, desc="world model endpoints", leave=False):
        wm_prob_gen = GenerateNextTokenProbAPI(client, wm_endpoint)
        wm_prob_star = compute_token_probs_api(y_star, prompt, wm_prob_gen)
        wm_prob_non_star = compute_token_probs_api(y_non_star, prompt, wm_prob_gen)

        value['y_stars'][y_star]['world_models'].append(float(wm_prob_star))
        value['y_NON_stars'][y_non_star]['world_models'].append(float(wm_prob_non_star))


def add_shadow_model_probs(value, y_star, y_non_star, prompt, client, shadow_model_endpoints):
    if 'shadow_models' not in value['y_stars'][y_star]:
        value['y_stars'][y_star]['shadow_models'] = []
    if 'shadow_models' not in value['y_NON_stars'][y_non_star]:
        value['y_NON_stars'][y_non_star]['shadow_models'] = []
        
    for sm_endpoint in tqdm(shadow_model_endpoints, desc="shadow model endpoints", leave=False):
        sm_prob_gen = GenerateNextTokenProbAPI(client, sm_endpoint)
        sm_prob_star = compute_token_probs_api(y_star, prompt, sm_prob_gen)
        sm_prob_non_star = compute_token_probs_api(y_non_star, prompt, sm_prob_gen)

        value['y_stars'][y_star]['shadow_models'].append(float(sm_prob_star))
        value['y_NON_stars'][y_non_star]['shadow_models'].append(float(sm_prob_non_star))


def add_model_probs(results, train_test_ents, client, world_model_endpoints, shadow_model_endpoints, model_type='world'):

    def find_ent_list(dataset_type, sample_id):
        for sample in train_test_ents[dataset_type]:
            if sample['ID'] == sample_id:
                return sample
        return None

    for key, value in tqdm(results.items(), desc="processing results"):
        split_key = key.split('_')
        dataset_type = split_key[0]
        sample_id = int(split_key[1])

        ent_list = find_ent_list(dataset_type, sample_id)
        if ent_list is None:
            continue

        ents = ent_list['ents']
        
        y_stars_order = list(value['y_stars'].keys())
        y_non_stars_order = list(value['y_NON_stars'].keys())

        for y_star, y_non_star in tqdm(zip(y_stars_order, y_non_stars_order), total=len(y_stars_order), desc="processing pairs", leave=True):
            if y_star not in ents:
                continue
            star_index = ents.index(y_star)
            remaining_ents = ents[:star_index] + ents[star_index + 1:]

            prompt_start = PROMPT_TEMPLATE[PROMPT_TO_USE][0]
            prompt_end = PROMPT_TEMPLATE[PROMPT_TO_USE][1]
            ents_string = ', '.join(remaining_ents)
            prompt = f"{prompt_start} {ents_string} {prompt_end}"

            if model_type == 'world':
                add_world_model_probs(value, y_star, y_non_star, prompt, client, world_model_endpoints)
            else:
                add_shadow_model_probs(value, y_star, y_non_star, prompt, client, shadow_model_endpoints)

# load target_token_probs.json into dict
with open('with_world_model_3.json', 'r') as f:
    all_model_probs = json.load(f)

## login & load together ai client
# key = '779d92de61a5035835e5023ca79e2e5b6124c6300c3ceb0e07e374f948554116'
key = "e94217f61953b12489a9877936bd5383086106ec9951d3f11bb6a9475d88e95e"
client = Together(api_key=key)
hf_login(token="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")

## load dataset
dataset = load_dataset("beanham/medsum_privacy")
merged_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

api_keys_subsample_ids = [
    ('lr2872/Meta-Llama-3.1-8B-Instruct-Reference-baf1323a-aaf0f7d2','3')
]

shadow_models = load_shadow_models_for_llama_3_instruct(MODEL_DICT, api_keys_subsample_ids)
model_world = MODEL_DICT['llama_3_instruct']
target_model_class = 'llama_3_instruct'
target_model_api_key, _, target_subsample_ids = shadow_models[0]
shadow_models_tuples = shadow_models[1:]
shadow_model_api_keys = [api_key for api_key, _, _ in shadow_models_tuples]

train_dataset = merged_dataset.filter(lambda example: example['ID'] in target_subsample_ids)
test_dataset = merged_dataset.filter(lambda example: example['ID'] not in target_subsample_ids)
unseen_ents = [sample['ents'] for sample in test_dataset if len(sample['ents']) < 5]
unseen_ents = [item for sublist in unseen_ents for item in sublist]
train_dataset = [sample for sample in train_dataset if len(sample['ents']) >= 5]
test_dataset = [sample for sample in test_dataset if len(sample['ents']) >= 5]
train_test_ents = {
    'train': train_dataset,
    'test': test_dataset
}

# world_model_endpoints = [
#     # "google/gemma-2b-it",
#     # "Qwen/Qwen2.5-7B-Instruct-Turbo",
#     # 'meta-llama/Llama-2-7b-chat-hf'
#     "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
# ]

shadow_model_endpoints = [
    "lr2872/Meta-Llama-3.1-8B-Instruct-Reference-2968ad77-11f16750",
    # "meta-llama/Meta-Llama-3-8B-Instruct-Shadow2"
]


add_model_probs(all_model_probs, train_test_ents, client, [], shadow_model_endpoints, model_type='shadow')

with open(f'with_world_model_shadow_model_3.json', 'w') as f:
    json.dump(all_model_probs, f)