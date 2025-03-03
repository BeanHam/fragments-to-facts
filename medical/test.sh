#!/usr/bin/env bash
set -e

# ------------------------------
# parameters
# ------------------------------
CV_ENDPOINT_MAP="cv_endpoint_map.json"

# endpoint map looks like:
# {
#   "mistral": {
#     "0": {
#       "train": {
#         "finetuned_model": "",
#         "endpoint_name": ""
#       },
#       "shadow": {
#         "finetuned_model": "",
#         "endpoint_name": ""
#       }
#     }
#   }
# }

# ------------------------------
# iter over each model tag
# ------------------------------
for model in $(jq -r 'keys[]' "$CV_ENDPOINT_MAP"); do

    ## iter over each seed for this model
    for seed in $(jq -r --arg model "$model" '.[$model] | keys[]' "$CV_ENDPOINT_MAP"); do

        ## extract the endpoint
        train_endpoint=$(jq -r --arg model "$model"\
                               --arg seed "$seed"\
                               --arg split "train"\
                               --arg name "endpoint_name" '.[$model][$seed][$split][$name]' "$CV_ENDPOINT_MAP")
        shadow_endpoint=$(jq -r --arg model "$model"\
                                --arg seed "$seed"\
                                --arg split "shadow"\
                                --arg name "endpoint_name" '.[$model][$seed][$split][$name]' "$CV_ENDPOINT_MAP")

        echo "train: $train_endpoint"
        echo "shadow: $shadow_endpoint"

        
    done
done
        
        