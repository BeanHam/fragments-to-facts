import random
import json
import argparse
from tqdm import tqdm
from os import makedirs, path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer

# valid models
VALID_MODELS = [
    'mistralai/Mistral-7B-Instruct-v0.2',
    'meta-llama/Llama-3.1-8B-Instruct',
    'Qwen/Qwen2-7B-Instruct'
]

system_message = (
    "You are a helpful medical assistant! Please help me summarize dialogues between doctors and patients."
)

def format_for_finetuning(data, system_message: str, tokenizer) -> str:
    chat = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": data['dialogue']},
        {"role": "assistant", "content": data['section_text']},
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return json.dumps({"text": text})

def subsample_ids(all_data, split_ratio: float, seed: int):
    random.seed(seed)
    total_ids = list(all_data['new_ID'])
    subsample = random.sample(total_ids, int(len(total_ids) * split_ratio))
    return subsample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='beanham/medsum_llm_attack',
                        help="Hugging Face dataset name.")
    parser.add_argument('--data_dir', type=str, default='cross_val_data',
                        help="Base directory to store the formatted files.")
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct',
                        help="Pre-trained model id to load the tokenizer from.")
    parser.add_argument('--split_ratio', type=float, default=0.5,
                        help="Ratio for subsampling the data (train/val) from the full dataset.")
    args = parser.parse_args()

    if args.model_id not in VALID_MODELS:
        raise ValueError(f"Invalid model_id: {args.model_id}. Please choose from: {VALID_MODELS}")

    if 'llama' in args.model_id.lower():
        args.model_tag = 'llama'
    elif 'mistral' in args.model_id.lower():
        args.model_tag = 'mistral'
    elif 'qwen' in args.model_id.lower():
        args.model_tag = 'qwen'
    else:
        args.model_tag = args.model_id.split('/')[-1] 

    print("Downloading and preparing dataset...")
    dataset = load_dataset(args.dataset)
    all_data = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])

    new_ids = list(range(len(all_data)))
    all_data = all_data.add_column("new_ID", new_ids)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        print(f"\nProcessing seed {seed}...")
        output_dir = path.join(args.data_dir, args.model_tag, str(seed))
        if not path.exists(output_dir):
            makedirs(output_dir)

        subsample = subsample_ids(all_data, args.split_ratio, seed)
        subsample_dataset = all_data.filter(lambda example: example['new_ID'] in subsample)
        dataset_train_val = subsample_dataset.train_test_split(test_size=0.2, seed=seed)
        dataset_train = dataset_train_val['train']
        dataset_val = dataset_train_val['test']

        shadow_dataset = all_data.filter(lambda example: example['new_ID'] not in subsample)
        shadow_dataset_train_val = shadow_dataset.train_test_split(test_size=0.2, seed=seed)
        shadow_dataset_train = shadow_dataset_train_val['train']
        shadow_dataset_val = shadow_dataset_train_val['test']
            
        print("Formatting training data...")
        formatted_train = "\n".join([
            format_for_finetuning(dataset_train[i], system_message, tokenizer)
            for i in tqdm(range(len(dataset_train)))
        ])
        print("Formatting validation data...")
        formatted_val = "\n".join([
            format_for_finetuning(dataset_val[i], system_message, tokenizer)
            for i in tqdm(range(len(dataset_val)))
        ])
        print("Formatting shadow training data...")
        formatted_shadow_train = "\n".join([
            format_for_finetuning(shadow_dataset_train[i], system_message, tokenizer)
            for i in tqdm(range(len(shadow_dataset_train)))
        ])
        print("Formatting shadow validation data...")
        formatted_shadow_val = "\n".join([
            format_for_finetuning(shadow_dataset_val[i], system_message, tokenizer)
            for i in tqdm(range(len(shadow_dataset_val)))
        ])

        ## to save index
        print('Saving Indices...')  
        with open(path.join(output_dir, 'train_ids.txt'), 'w') as f:
            for row in dataset_train['new_ID']:
                f.write(str(row) + '\n')
        with open(path.join(output_dir, 'val_ids.txt'), 'w') as f:
            for row in dataset_val['new_ID']:
                f.write(str(row) + '\n')
        with open(path.join(output_dir, 'shadow_train_ids.txt'), 'w') as f:
            for row in shadow_dataset_train['new_ID']:
                f.write(str(row) + '\n')
        with open(path.join(output_dir, 'shadow_val_ids.txt'), 'w') as f:
            for row in shadow_dataset_val['new_ID']:
                f.write(str(row) + '\n')

        ## to save formatted data
        with open(path.join(output_dir, 'train.jsonl'), 'w') as f:
            f.write(formatted_train)
        with open(path.join(output_dir, 'val.jsonl'), 'w') as f:
            f.write(formatted_val)
        with open(path.join(output_dir, 'shadow_train.jsonl'), 'w') as f:
            f.write(formatted_shadow_train)
        with open(path.join(output_dir, 'shadow_val.jsonl'), 'w') as f:
            f.write(formatted_shadow_val)
        print(f"Saved formatted data for seed {seed} in {output_dir}")

if __name__ == "__main__":
    main()
