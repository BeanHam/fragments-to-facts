#!/usr/bin/env bash
set -e

CV_ENDPOINT_MAP="cv_endpoint_map.json"
SPLIT="train"         # which finetuned model to deploy ("train" or "shadow")?

# endpoint map looks like:
# {
#   "mistral": {
#     "0": {
#       "train": {
#         "finetuned_model": "bh193/Mistral-7B-Instruct-v0.2-086d8308",
#         "endpoint_id": "endpoint-705513fe-a2f0-4e30-b297-868a2a0961e4"
#       }
#     }
#   }
# }

# iter over each model tag in the finetuned map
for model in $(jq -r 'keys[]' "$CV_FINETUNE_MAP"); do
    # iter over each seed for this model
    for seed in $(jq -r --arg model "$model" '.[$model] | keys[]' "$CV_FINETUNE_MAP"); do
        # extract the finetuned model for the chosen split
        