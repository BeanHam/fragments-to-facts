import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModel

# f"consider someone with the following conditions: {', '.join(remaining_ents)}. the individual then also has the condition "
PROMPT_TEMPLATE = {
    0: ["consider someone with the following conditions: ", ". the individual then also has the condition "],
    1: ["consider an individual whose medical note contains the following: ", ". that individual's medical note would also include: "],
    2: ["Consider an individual whose medical summary contains: ", ". That individual's medical summary then also includes: "],
}

PROMPT_TO_USE = 2

MODEL_DICT = {
    'llama_3_1': 'meta-llama/Meta-Llama-3.1-8B-Reference',
    'llama_3_1_instruct': 'meta-llama/Meta-Llama-3.1-8B-Instruct-Reference',
    'llama_3': 'meta-llama/Meta-Llama-3-8B',
    'llama_3_instruct': 'meta-llama/Meta-Llama-3-8B-Instruct',
    'llama_2': 'togethercomputer/llama-2-7b',
    'llama_2_chat': 'togethercomputer/llama-2-7b-chat',
    'codellama': 'codellama/CodeLlama-7b-hf',
    'codellama_python': 'codellama/CodeLlama-7b-Python-hf',
    'codellama_instruct': 'codellama/CodeLlama-7b-Instruct-hf',
    'mixtral_8x7b': 'mistralai/Mixtral-8x7B-v0.1',
    'mixtral_8x7b_instruct': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'nous_hermes_2_mixtral_8x7b_dpo': 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO',
    'nous_hermes_2_mixtral_8x7b_sft': 'NousResearch/Nous-Hermes-2-Mixtral-8x7B-SFT',
    'mistral_7b_instruct': 'mistralai/Mistral-7B-Instruct-v0.2',
    'mistral_7b': 'mistralai/Mistral-7B-v0.1',
    'qwen2': 'Qwen/Qwen2-1.5B',
    'qwen2_instruct': 'Qwen/Qwen2-1.5B-Instruct',
    'openhermes_2_5_mistral_7b': 'teknium/OpenHermes-2p5-Mistral-7B',
    'zephyr_7b': 'HuggingFaceH4/zephyr-7b-beta',
    'solar_instruct_v1': 'upstage/SOLAR-10.7B-Instruct-v1.0',
}


def likelihood_ratio_to_probability(likelihood_ratio, prior_in=0.5):
    
    prior_out = 1 - prior_in

    # compute the marginal likelihood
    marginal_likelihood = (likelihood_ratio * prior_in) + (1 / likelihood_ratio * prior_out)

    # bayes to get the posterior probability of membership
    posterior_in = (likelihood_ratio * prior_in) / marginal_likelihood

    return posterior_in


def auc_calculation(labels, probs):
    fpr,tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr,tpr)
    return roc_auc

class GenerateNextTokenProbAPI:
    def __init__(self, api_client, model_name):
        self.api_client = api_client
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')

    def find_target_in_tokens(self, target_string, tokens):
        # print('finding target string in tokens')
        reconstructed_text = ''.join(tokens).replace(' ', '').replace('Ġ', '')
        target_string_no_spaces = target_string.replace(' ', '').replace('Ġ', '')

        # pos of target string in the text
        index_in_text = reconstructed_text.find(target_string_no_spaces)
        if index_in_text == -1:
            return None  # bad

        # map char indices back to token indices
        accumulated_length = 0
        start_token_index = None
        end_token_index = None
        for i, tok in enumerate(tokens):
            tok_no_space = tok.replace(' ', '').replace('Ġ', '')
            tok_length = len(tok_no_space)
            if accumulated_length <= index_in_text < accumulated_length + tok_length:
                start_token_index = i
            if accumulated_length < index_in_text + len(target_string_no_spaces) <= accumulated_length + tok_length:
                end_token_index = i + 1
                break
            accumulated_length += tok_length

        #print('done')
        if start_token_index is not None and end_token_index is not None:
            return start_token_index, end_token_index
        else:
            return None

    def get_token_probs(self, prompt, target_string, remaining_ents, max_tokens):
        messages = [
            {"role": "system", "content": "You are a medical expert."},
            {"role": "user", "content": prompt + " " + target_string}
        ]

        response = self.api_client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_tokens,  ## added on 12/20/2024
            temperature=0,
            logprobs=True,
            echo=True
        )
        
        tokens = response.prompt[0].logprobs.tokens
        logprobs = response.prompt[0].logprobs.token_logprobs

        target_indices = self.find_target_in_tokens(target_string, tokens)
        if target_indices is None:
            return -1

        start_index, end_index = target_indices
        logprobs_slice = logprobs[start_index:end_index]
        prob_target_string = np.exp(sum(logprobs_slice))

        ents_prob = {}
        for ent in remaining_ents:
            ent_indices = self.find_target_in_tokens(ent, tokens)
            if ent_indices is None:
                ents_prob[ent] = -1
                print(f"Entity '{ent}' not found in the echoed response. BAD.")
            else:
                start_idx, end_idx = ent_indices
                logprobs_slice = logprobs[start_idx:end_idx]
                ents_prob[ent] = np.exp(sum(logprobs_slice))

        return {
            "target_prob": prob_target_string,
            "ents_prob": ents_prob
        }


def compute_token_probs_api(prob_generator, prompt, target_string, remaining_ents, max_tokens):
    return prob_generator.get_token_probs(prompt, target_string, remaining_ents, max_tokens)

def load_shadow_models_for_llama_3_instruct(model_dict, api_keys_subsample_ids):
    shadow_models_tuples = []    
    for api_key, id in api_keys_subsample_ids:
        model_name = f"{model_dict['llama_3_instruct']}-{id}"
        subsample_ids = pd.read_csv(f"formatted_data/subsample_ids_{id}.csv")
        subsample_ids = subsample_ids['ID'].tolist()
        shadow_models_tuples.append((api_key, model_name, subsample_ids))
    return shadow_models_tuples