import time
import json
import math
import random
import argparse
from tqdm import tqdm
from os import makedirs, path
from together import Together

from utils import *
from huggingface_hub import login as hf_login
from datasets import concatenate_datasets, load_dataset

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # Parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beanham/medsum_llm_attack')
    parser.add_argument('--data_dir', type=str, default='formatted_data/')
    parser.add_argument('--save_dir', type=str, default='probs/')
    parser.add_argument('--model_tag', type=str, default='llama_1_epoch')
    parser.add_argument('--ablation_pct', type=str, default='1.0')
    parser.add_argument('--hf_key', type=str, default='')
    parser.add_argument('--together_key', type=str)
    args = parser.parse_args()

    ## log in together ai & hugginface
    with open('model_map.json') as f:
        model_map=json.load(f)
    client = Together(api_key=args.together_key)
    target_model_api_key = model_map[args.model_tag]['train']['api_key']
    if args.hf_key:
        hf_login(token=args.hf_key, add_to_git_credential=True)
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct', token=args.hf_key)
    else:
        hf_login()
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

    prob_generator = GenerateNextTokenProbAPI(client, target_model_api_key)    
    input(f"""Please deploy the following model {target_model_api_key}. The deployment might take up to 10 mins...""")

    ## load dataset
    print('Load & Prepare Dataset...')
    dataset = load_dataset(args.dataset)
    all_data = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    new_ids = range(len(all_data))
    all_data = all_data.add_column("new_ID", new_ids)
    
    ## load split ids
    with open(path.join(args.data_dir, 'train_ids.txt'), 'r') as f:
        train_ids=f.readlines()
    train_ids=[int(i.split()[0]) for i in train_ids]
    dataset_train = all_data.filter(lambda example: example['new_ID'] in train_ids)
    dataset_test = all_data.filter(lambda example: example['new_ID'] not in train_ids)

    ## unseen entities
    unseen_ents = [sample['disease_ents'] for sample in dataset_test if len(sample['disease_ents'])<5]
    unseen_ents = [item for sublist in unseen_ents for item in sublist]
    unseen_ents = list(set(unseen_ents))

    ## portion of dataset
    dataset_train = [sample for sample in dataset_train if len(sample['disease_ents'])>=5]
    dataset_test = [sample for sample in dataset_test if len(sample['disease_ents'])>=5]
    train_test_ents = {'train': dataset_train,'test': dataset_test}
    
    ## prob extraction
    print('Extracting Probabilities...')
    results = {}
    fail_counter = 0
    for name, samples in train_test_ents.items():
        for j, ent_list in tqdm(enumerate(samples)):
            print(f'{name.upper()}: {j+1}/{len(samples)}...')
            key_name = name + '_' + str(ent_list['ID'])   ## change from ID to new_ID
            results[key_name] = {}
            results[key_name]['y_stars'] = {}
            results[key_name]['y_NON_stars'] = {}
            ents = list(set(ent_list['disease_ents']))
            k = len(ents)
            unseen_ents_for_sample = random.sample(unseen_ents, k)
    
            # make sure no overlap between unseen_ents and ents
            for i in range(k):
                while unseen_ents_for_sample[i] in ents:
                    unseen_ents_for_sample[i] = random.choice(unseen_ents)
                    print('had to swap an ent')
    
            for i in range(k):
                y_star = ents[i]
                y_NON_star = unseen_ents_for_sample[i]
    
                results[key_name]['y_stars'][y_star] = {}
                results[key_name]['y_NON_stars'][y_NON_star] = {}
    
                remaining_ents = ents[:i] + ents[i+1:]
                
                ablation_pct = float(args.ablation_pct)
                # print(f"ablation_pct: {ablation_pct}")

                # if ablation_pct is not 1.0, we need to swap some of the prompt stimuli with unseen entities
                if ablation_pct != 1.0:

                    if ablation_pct < 0.0 or ablation_pct > 1.0:
                        raise ValueError("ablation_pct must be between 0.0 and 1.0")

                    # randomly remove a percentage of the prompt stimuli and replace with unseen entities
                    true_ent_count = int(math.ceil(len(remaining_ents)*ablation_pct))
                    unseen_ent_count = len(remaining_ents) - true_ent_count
                    new_unseen_ents = random.sample(unseen_ents, unseen_ent_count)

                    for i in range(len(new_unseen_ents)):
                        while new_unseen_ents[i] in remaining_ents:
                            new_unseen_ents[i] = random.choice(unseen_ents)
                            print('had to swap a new unseen ent')
                    
                    remaining_ents = random.sample(remaining_ents, true_ent_count) + new_unseen_ents
    
                prompt_start = PROMPT_TEMPLATE[PROMPT_TO_USE][0]
                prompt_end = PROMPT_TEMPLATE[PROMPT_TO_USE][1]
                ents_string = ', '.join(remaining_ents)
                prompt = f"{prompt_start} {ents_string} {prompt_end}"
    
                max_tokens = len(prob_generator.tokenizer(prompt)['input_ids']) + 10
    
                # now a prob dictionary for y_star + remaining_ents
                star_probs_dict = compute_token_probs_api(
                    prob_generator,
                    prompt=prompt,
                    target_string=y_star,
                    remaining_ents=remaining_ents,
                    max_tokens=max_tokens
                )
    
                if star_probs_dict['target_prob'] == -1:
                    fail_counter += 1
                    print(f"failed {fail_counter} times (y_star not found)")
                    continue
    
                # prob dictionary for y_NON_star + remaining_ents
                non_star_probs_dict = compute_token_probs_api(
                    prob_generator,
                    prompt=prompt,
                    target_string=y_NON_star,
                    remaining_ents=remaining_ents,
                    max_tokens=max_tokens
                )
    
                if non_star_probs_dict['target_prob'] == -1:
                    fail_counter += 1
                    print(f"failed {fail_counter} times (y_NON_star not found)")
                    continue
    
                # save target prob, ent_probs, tokens_probs
                results[key_name]['y_stars'][y_star]['target_prob'] = star_probs_dict['target_prob']
                results[key_name]['y_stars'][y_star]['ents_prob']   = star_probs_dict['ents_prob']
                results[key_name]['y_stars'][y_star]['prompt']       = prompt
    
                results[key_name]['y_NON_stars'][y_NON_star]['target_prob'] = non_star_probs_dict['target_prob']
                results[key_name]['y_NON_stars'][y_NON_star]['ents_prob']   = non_star_probs_dict['ents_prob']
                results[key_name]['y_NON_stars'][y_NON_star]['prompt']       = prompt
    
                # short delay to avoid rate-limiting
                time.sleep(0.1)

    ## save results
    print('Save Results...')
    if ablation_pct != 1.0:
        ablation_str = int(float(args.ablation_pct)*100)
        with open(path.join(args.save_dir, f'{args.model_tag}_target_probs_prompt_{PROMPT_TO_USE}_{ablation_str}.json'), 'w') as f:
            json.dump(results, f)
    else:
        with open(path.join(args.save_dir, f'{args.model_tag}_target_probs_prompt_{PROMPT_TO_USE}.json'), 'w') as f:
            json.dump(results, f)

if __name__ == "__main__":
    main()