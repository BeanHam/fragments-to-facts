import os
import sys
import json
import torch
import string
import logging
import datasets
import argparse
import numpy as np
import transformers
import bitsandbytes as bnb

from utils import *
from os import path, makedirs, getenv
from transformers import TrainingArguments
from huggingface_hub import login as hf_login

import openai
from openai import OpenAI
from openai_chat_api import DialogueBot

def format_for_finetuning(user_input: str,
                          assistant_output: str,
                          system_prompt: str) -> str:
    """
    Format data in JSON for fine-tuning an OpenAI chatbot model.
    """

    return json.dumps(
        {
            "messages": [
                {"role": "system", "content": system_prompt}, 
                {"role": "user", "content": user_input}, 
                {"role": "assistant", "content": assistant_output}
            ]
        }
    )

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Fine-tune a spatial-join model.')    
    parser.add_argument('--model_id', type=str, default='gpt-4o-2024-08-06', help='The model ID to fine-tune.')
    parser.add_argument('--OPENAI_API_KEY', type=str, help='API key to finetune GPT-4o')
    parser.add_argument('--dataset', type=str, default='beanham/spatial_join', help='The dataset to use for fine-tuning.')
    parser.add_argument('--formatted_data_dir', type=str, help='The directory to save the formatted data to', default='formatted_data')
    args = parser.parse_args()
    os.environ["OPENAI_API_KEY"] = args.OPENAI_API_KEY
    
    
    if not path.exists(args.formatted_data_dir):
        mkdir(args.formatted_data_dir)
        print(f'Created directory {args.formatted_data_dir}')
    
    # ----------------------
    # Load Data
    # ----------------------
    print('Downloading and preparing data...')
    data = get_dataset_slices(args.dataset)
    train = data['train']    
    val = data['val']
    test = data['test']
    
    system_message = """
    You are a helpful geospatial analysis assistant! I will provide you with a pair of (sidewalk, road) information in GeoJson format. Please help me identify whether the sidewalk is alongside the paired road, such that the sidewalk is adjacent and parellele to the road. If it is, please return 1; otherwise, return 0.
    """    
    
    print('Formatting data for fine-tuning...')        
    train_formatted = '\n'.join(
        [format_for_finetuning(
            "\nSidewalk:\n"+str(train['sidewalk'][i])+"\n\nRoad:\n"+str(train['road'][i]),
            "\nLable: "+str(train['label'][i]),
            system_message
        ) for i in range(len(train))]
    )
    val_formatted = '\n'.join(
        [format_for_finetuning(
            "\nSidewalk:\n"+str(val['sidewalk'][i])+"\n\nRoad:\n"+str(val['road'][i]),
            "\nLable: "+str(val['label'][i]),
            system_message
        ) for i in range(len(val))]
    )
    
    # ----------------------------------
    # Write the formatted data to a file
    # ----------------------------------
    print('Writing formatted data to file...')
    with open(path.join(args.formatted_data_dir, 'gpt4o_train.jsonl'), 'w') as f:
        f.write(train_formatted)
    with open(path.join(args.formatted_data_dir, 'gpt4o_val.jsonl'), 'w') as f:
        f.write(val_formatted)

    # ----------------------------------
    # Set the OpenAI API key and create a client
    # ----------------------------------        
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI()

    # Create the training dataset
    train_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, 'gpt4o_train.jsonl'), "rb"),
        purpose="fine-tune"
    )
    val_response = client.files.create(
        file=open(path.join(args.formatted_data_dir, 'gpt4o_val.jsonl'), "rb"),
        purpose="fine-tune"
    )
    
    # Create the fine-tuning job
    job_response = client.fine_tuning.jobs.create(
        training_file=train_response.id,
        validation_file=val_response.id,
        model=args.model_id,
        hyperparameters={
            "n_epochs": 1,
        }
    )
    
    # Wait for the fine-tuning job to complete
    job_status = client.fine_tuning.jobs.retrieve(job_response.id)
    while job_status.status != 'succeeded' and job_status.status != 'failed':
        job_status = client.fine_tuning.jobs.retrieve(job_response.id)
        print('Fine-tuning job status: ', job_status.status)
        print(job_status)
        time.sleep(60)

    if job_status.status == 'failed':
        raise Exception('Fine-tuning job failed')