import json
import argparse
import numpy as np
from tqdm import tqdm
from os import makedirs, path
from together import Together
from utils import *

#-----------------------
# Main Function
#-----------------------
def main():
    
    #-------------------
    # Parameters
    #-------------------    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='formatted_data/')
    parser.add_argument('--model_tag', type=str, default='llama_10_epoch')
    parser.add_argument('--together_key', type=str)
    args = parser.parse_args()
    with open('model_map.json') as f:
        model_map=json.load(f)
    client = Together(api_key=args.together_key)
    target_model_api_key = model_map[args.model_tag]['train']['api_key']
    input(f"""======================================================================================================================================================
Please deploy the following model {target_model_api_key}. The deployment might take up to 10 mins. Once the model is deployed, please proceed...
======================================================================================================================================================""")

    #-------------------------
    # load llama training data
    #-------------------------
    system_message="You are a helpful medical assistant! Please help me summarize dialogues between doctors and patients."
    if 'llama' in args.model_tag:
        with open('formatted_data/llama_train.jsonl') as f:
            data = [json.loads(line) for line in f]
    elif 'qwen' in args.model_tag:
        with open('formatted_data/qwen_train.jsonl') as f:
            data = [json.loads(line) for line in f]
    else:
        with open('formatted_data/mistral_train.jsonl') as f:
            data = [json.loads(line) for line in f]

    #-------------------------
    # memorization attack
    #-------------------------
    output=[]
    for i in tqdm(range(len(data))):
        sample=data[i]
        text=sample['text']
        start_index=text.find("<|eot_id|><|start_header_id|>user<|end_header_id|>")
        end_index=text.find("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")
        conv=text[start_index+50:end_index]+'\n\n'
        prompt=text[end_index+55:]
        for prompt_length in [10,20,30]:
            user=conv+' '.join(prompt.split()[:prompt_length])
            gt=' '.join(prompt.split()[prompt_length:])
            messages = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user}
            ]
            response = client.chat.completions.create(
                model=target_model_api_key,
                messages=messages,
                max_tokens=50,
                temperature=0
            )
            output.append(['train_'+str(i), prompt_length, user, gt, response.choices[0].message.content])
        
    ## save results
    np.save(f'{args.model_tag}_memorization_attack_results.npy', output)

if __name__ == "__main__":
    main()