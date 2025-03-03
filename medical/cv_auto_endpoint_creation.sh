#!/usr/bin/env bash
set -e

# ------------------------------
# parameters
# ------------------------------
CV_FINETUNE_MAP="cv_finetuned_model_map.json"
CV_ENDPOINT_MAP="cv_endpoint_map.json"
GPU_TYPE="a100"     ## options: {'h100', 'a100', 'l40', 'l40s', 'rtx-6000'}
GPU_COUNT=2         ## options: {2,4,8}
SPLIT="shadow"      ## options: {"train", "shadow"}

## init the endpoint map file if it doesn't exist
if [ ! -f "$CV_ENDPOINT_MAP" ]; then
    echo "{}" > "$CV_ENDPOINT_MAP"
fi

# ------------------------------
# iter over each model tag
# ------------------------------
for model in $(jq -r 'keys[]' "$CV_FINETUNE_MAP"); do

    ## iter over each seed for this model
    for seed in $(jq -r --arg model "$model" '.[$model] | keys[]' "$CV_FINETUNE_MAP"); do
    
        ## extract the finetuned model for the chosen split
        finetuned_model=$(jq -r --arg model "$model" --arg seed "$seed" --arg split "$SPLIT" '.[$model][$seed][$split]' "$CV_FINETUNE_MAP")
        if [ -z "$finetuned_model" ] || [ "$finetuned_model" == "null" ]; then
            echo "No finetuned model found for model: $model, seed: $seed, split: $SPLIT. Skipping..."
            continue
        fi
        
        ## create the endpoint using --wait to block until it is ready
        echo "Creating endpoint for finetuned model: $finetuned_model (model: $model, seed: $seed, split: $SPLIT)"
        endpoint_creation_output=$(together endpoints create \
            --model "$finetuned_model" \
            --gpu "$GPU_TYPE" \
            --gpu-count "$GPU_COUNT" \
            --display-name "${model}-${seed}-endpoint" \
            --wait)
        
        ## extract the endpoint id
        endpoint_id=$(echo "$endpoint_creation_output" | grep -oE 'endpoint-[0-9A-Za-z\-]+')
        echo "  Created endpoint: $endpoint_id"

        ## immediately stop the endpoint to avoid spending money
        echo "  Stopping endpoint: $endpoint_id"
        together endpoints stop "$endpoint_id" --wait

        ## extract endpoint name using endpoint id
        endpoint_name_extraction=$(together endpoints get $endpoint_id)        
        endpoint_name=$(echo "$endpoint_name_extraction" | sed -n 's/^Name:[[:space:]]*//p')
        echo "  Endpoint Name: $endpoint_name"

        # update the cv_endpoint_map.json with the mapping for this model/seed.
        # nested structure will be:
        # { model: { seed: { <SPLIT>: { "finetuned_model": <finetuned_model>, "endpoint_id": <endpoint_id> } } } }
        tmp=$(mktemp)
        jq --arg model "$model" --arg seed "$seed" --arg split "$SPLIT" \
           --arg finetuned_model "$finetuned_model" --arg endpoint_name "$endpoint_name" \
           '.[$model][$seed][$split] = { "finetuned_model": $finetuned_model, "endpoint_name": $endpoint_name }' \
           "$CV_ENDPOINT_MAP" > "$tmp" && mv "$tmp" "$CV_ENDPOINT_MAP"
        echo "  Updated $CV_ENDPOINT_MAP for model: $model, seed: $seed, split: $SPLIT"        
    done
done