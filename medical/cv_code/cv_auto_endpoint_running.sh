#!/usr/bin/env bash
set -e

# ------------------------------
# parameters
# ------------------------------
CV_ENDPOINT_MAP="cv_endpoint_map.json"
POLL_INTERVAL=60  ## seconds between polling

## also need to setup the environment variable by doing:
## export TOGETHER_API_KEY=""

# ------------------------------
# iter over each models
# ------------------------------
for model in $(jq -r 'keys[]' "$CV_ENDPOINT_MAP"); do

    ## iter over each seed for this model
    for seed in $(jq -r --arg model "$model" '.[$model] | keys[]' "$CV_ENDPOINT_MAP"); do

        echo "========================================"
        echo "Model: $model"
        echo "Seed: $seed"        
        echo "========================================"
        
        # ------------------------------
        # extract target model probs
        # ------------------------------
        
        echo "------------------------"
        echo "Target Prob Extraction"
        echo "------------------------"
        target_endpoint_name=$(jq -r --arg model "$model"\
                                     --arg seed "$seed"\
                                     --arg split "train"\
                                     --arg name "endpoint_name" '.[$model][$seed][$split][$name]' "$CV_ENDPOINT_MAP")
        target_endpoint_id=$(jq -r --arg model "$model"\
                                     --arg seed "$seed"\
                                     --arg split "train"\
                                     --arg id "endpoint_id" '.[$model][$seed][$split][$id]' "$CV_ENDPOINT_MAP")                                            
        together endpoints start "$target_endpoint_id"
        python target_prob_extraction.py \
            --model "$model" \
            --endpoint_name "$target_endpoint_name" \
            --seed "$seed" \
            --together_key "$TOGETHER_API_KEY"
        together endpoints stop "$target_endpoint_id"
        
        # ------------------------------
        # extract shadow model probs
        # ------------------------------  

        echo "------------------------"
        echo "Shadow Prob Extraction"
        echo "------------------------"
        shadow_endpoint_name=$(jq -r --arg model "$model"\
                                     --arg seed "$seed"\
                                     --arg split "shadow"\
                                     --arg name "endpoint_name" '.[$model][$seed][$split][$name]' "$CV_ENDPOINT_MAP")
        shadow_endpoint_id=$(jq -r --arg model "$model"\
                                     --arg seed "$seed"\
                                     --arg split "shadow"\
                                     --arg id "endpoint_id" '.[$model][$seed][$split][$id]' "$CV_ENDPOINT_MAP")
        
        together endpoints start "$shadow_endpoint_id"
        python shadow_prob_extraction.py \
            --model "$model" \
            --endpoint_name "$shadow_endpoint_name" \
            --seed "$seed" \
            --together_key "$TOGETHER_API_KEY"
        together endpoints stop "$shadow_endpoint_id"

        # ------------------------------
        # extract world models probs
        # ------------------------------ 
        echo "------------------------"
        echo "World Prob Extraction"
        echo "------------------------"
        python world_prob_extraction.py \
            --model "$model" \
            --seed "$seed" \
            --together_key "$TOGETHER_API_KEY"        
    done
done
        

## appendix
#while true; do
#    echo "Polling status for target endpoint..."
#    retrieve_output=$(together endpoints get $target_endpoint_id)            
#    status=$(echo "$retrieve_output" | sed -n 's/^State:[[:space:]]*//p')
#    echo "Current status: $status"
#    if [ "$status" == "STARTED" ]; then
#        break
#    fi
#    echo "Endpoint not started yet. Sleeping for $POLL_INTERVAL seconds..."
#    sleep "$POLL_INTERVAL"
#done