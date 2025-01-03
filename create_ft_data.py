import json
import argparse
from tqdm import tqdm
from datasets import Dataset
from os import makedirs, path
from transformers import AutoTokenizer
from datasets import load_dataset, concatenate_datasets

system_message = """You are a helpful medical assistant! Please help me summarize dialogues between doctors and patients."""

def format_for_finetuning(data,
                          system_message: str,
                          tokenizer) -> str:
    """
    Format data in JSON for fine-tuning an OpenAI chatbot model.
    """
    
    chat = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": data['dialogue']},
        {"role": "assistant", "content": data['section_text']},      
    ]
    text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=False)
    return json.dumps({"text":text})
    
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
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.1-8B-Instruct')
    args = parser.parse_args()
    if not path.exists(args.data_dir):
        makedirs(args.data_dir)
    if 'llama' in args.model_id:
        args.model_tag = 'llama'
    elif 'mistral' in args.model_id:
        args.model_tag = 'mistral'
    elif 'Qwen' in args.model_id:
        args.model_tag = 'qwen'
    
    # ----------------------
    # Load & Process Data
    # ----------------------
    print('Downloading and preparing data...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    dataset = load_dataset(args.dataset)
    all_data = concatenate_datasets([dataset['train'], dataset['validation'], dataset['test']])
    new_ids = range(len(all_data))
    all_data = all_data.add_column("new_ID", new_ids)

    ## load split ids
    with open(path.join(args.data_dir, 'train_ids.txt'), 'r') as f:
        train_ids=f.readlines()    
    with open(path.join(args.data_dir, 'val_ids.txt'), 'r') as f:
        val_ids=f.readlines()
    with open(path.join(args.data_dir, 'shadow_train_ids.txt'), 'r') as f:
        shadow_train_ids=f.readlines()
    with open(path.join(args.data_dir, 'shadow_val_ids.txt'), 'r') as f:
        shadow_val_ids=f.readlines()
    train_ids=[int(i.split()[0]) for i in train_ids]
    val_ids=[int(i.split()[0]) for i in val_ids]
    shadow_train_ids=[int(i.split()[0]) for i in shadow_train_ids]
    shadow_val_ids=[int(i.split()[0]) for i in shadow_val_ids]    
    
    dataset_train = all_data.filter(lambda example: example['new_ID'] in train_ids)
    dataset_val = all_data.filter(lambda example: example['new_ID'] in val_ids)
    shadow_dataset_train = all_data.filter(lambda example: example['new_ID'] in shadow_train_ids)
    shadow_dataset_val = all_data.filter(lambda example: example['new_ID'] in shadow_val_ids)

    # ----------------------
    # Format Data
    # ----------------------    
    formatted_train = '\n'.join(
        [format_for_finetuning(dataset_train[i], system_message, tokenizer) for i in tqdm(range(len(dataset_train)))]
        )
    formatted_val = '\n'.join(
        [format_for_finetuning(dataset_val[i], system_message, tokenizer) for i in tqdm(range(len(dataset_val)))]
        )
    formatted_shadow_train = '\n'.join(
        [format_for_finetuning(shadow_dataset_train[i], system_message, tokenizer) for i in tqdm(range(len(shadow_dataset_train)))]
    )
    formatted_shadow_val = '\n'.join(
        [format_for_finetuning(shadow_dataset_train[i], system_message, tokenizer) for i in tqdm(range(len(shadow_dataset_val)))]
    )
    with open(path.join(args.data_dir, f'{args.model_tag}_train.jsonl'), 'w') as f:
        f.write(formatted_train)
    with open(path.join(args.data_dir, f'{args.model_tag}_val.jsonl'), 'w') as f:
        f.write(formatted_val)
    with open(path.join(args.data_dir, f'{args.model_tag}_shadow_train.jsonl'), 'w') as f:
        f.write(formatted_shadow_train)
    with open(path.join(args.data_dir, f'{args.model_tag}_shadow_val.jsonl'), 'w') as f:
        f.write(formatted_shadow_val)

if __name__ == "__main__":
    main()