import os
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from together import Together
from transformers import AutoTokenizer
import time

from utils import *
from huggingface_hub import login as hf_login
from peft import prepare_model_for_kbit_training
from datasets import concatenate_datasets, DatasetDict, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel

key = '...'
client = Together(api_key=key)
hf_login(token="...")
dataset = load_dataset("beanham/medsum_llm_attack")
merged_dataset = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
new_ids = range(len(merged_dataset))
merged_dataset = merged_dataset.add_column("new_ID", new_ids)

split='train'
id='3'
subsample_split='subsample_ids' # subsample_ids shadow_ids

## load model
target_model_api_key="lr2872/Meta-Llama-3.1-8B-Instruct-Reference-88e7bbe3-b7d323df"
prob_generator = GenerateNextTokenProbAPI(client, target_model_api_key)

## load data
subsample_split='subsample_ids' # subsample_ids
target_subsample_ids = pd.read_csv(f"formatted_data/{subsample_split}_{id}.csv")['new_ID'].tolist()
train_dataset = merged_dataset.filter(lambda example: example['new_ID'] in target_subsample_ids)
test_dataset = merged_dataset.filter(lambda example: example['new_ID'] not in target_subsample_ids)

## why are we only using len(ents)<5 as the unseen ents?
## because we want to make sure they aren't entities from the notes we use!
unseen_ents = [sample['disease_ents'] for sample in test_dataset if len(sample['disease_ents']) < 5]
unseen_ents = [item for sublist in unseen_ents for item in sublist]

train_dataset = [sample for sample in train_dataset if len(sample['disease_ents']) >= 5]
test_dataset = [sample for sample in test_dataset if len(sample['disease_ents']) >= 5]
train_test_ents = {'train': train_dataset,'test': test_dataset}

results={}
fail_counter = 0

# get first 50 pairs from train_test_ents 
for name, samples in train_test_ents.items():
    
    for j, ent_list in tqdm(enumerate(samples[:50])):
        
        ## create saving dictionary
        key=name+'_'+str(ent_list['ID'])
        results[key]={}
        results[key]['y_stars']={}
        results[key]['y_NON_stars']={}        
        ents = list(set(ent_list['disease_ents']))
        k = len(ents)
        unseen_ents_for_sample = random.sample(unseen_ents, k)
        
        ## go through each y_star
        for i in tqdm(range(k)):
            
            y_star = ents[i]
            y_NON_star = unseen_ents_for_sample[i]
            results[key]['y_stars'][y_star]={}
            results[key]['y_NON_stars'][y_NON_star]={}                        
            remaining_ents = ents[:i] + ents[i + 1:]            
            
            prompt_start = PROMPT_TEMPLATE[PROMPT_TO_USE][0]
            prompt_end = PROMPT_TEMPLATE[PROMPT_TO_USE][1]
            ents_string = ', '.join(remaining_ents)
            prompt = f"{prompt_start} {ents_string} {prompt_end}"

            prob = compute_token_probs_api(y_star, prompt, prob_generator) 
            prob_NON = compute_token_probs_api(y_NON_star, prompt, prob_generator)            
            if prob == -1 or prob_NON == -1:
                fail_counter += 1
                print(f"failed {fail_counter} times")
                continue            
            results[key]['y_stars'][y_star]['target']=prob
            results[key]['y_NON_stars'][y_NON_star]['target']=prob_NON

            results[key]['y_stars'][y_star]['prompt']=prompt
            results[key]['y_NON_stars'][y_NON_star]['prompt']=prompt

            # seems like a short delay here helps with the API server
            # not blocking the requests
            time.sleep(0.1)

        if j % 5 == 1:
            with open(f'target_token_probs_{split}_{id}_10_epochs_with_prompt_higher.json', 'w') as f:
                json.dump(results, f)
