#!/usr/bin/env bash
set -e

CV_FINETUNE_MAP="cv_finetuned_model_map.json"
CV_ENDPOINT_MAP="cv_endpoint_map.json"
GPU_TYPE="h100"       # TODO: not really sure how to do this
GPU_COUNT=1           # num GPUs per replica
SPLIT="train"         # which finetuned model to deploy ("train" or "shadow")?

# init the endpoint map file if it doesn't exist
if [ ! -f "$CV_ENDPOINT_MAP" ]; then
    echo "{}" > "$CV_ENDPOINT_MAP"
fi

# iter over each model tag in the finetuned map
for model in $(jq -r 'keys[]' "$CV_FINETUNE_MAP"); do
    # iter over each seed for this model
    for seed in $(jq -r --arg model "$model" '.[$model] | keys[]' "$CV_FINETUNE_MAP"); do
        # extract the finetuned model for the chosen split
        finetuned_model=$(jq -r --arg model "$model" --arg seed "$seed" --arg split "$SPLIT" '.[$model][$seed][$split]' "$CV_FINETUNE_MAP")
        if [ -z "$finetuned_model" ] || [ "$finetuned_model" == "null" ]; then
            echo "No finetuned model found for model: $model, seed: $seed, split: $SPLIT. Skipping..."
            continue
        fi

        echo "Creating endpoint for finetuned model: $finetuned_model (model: $model, seed: $seed, split: $SPLIT)"
        # create the endpoint using --wait to block until it is ready
        endpoint_creation_output=$(together endpoints create \
            --model "$finetuned_model" \
            --gpu "$GPU_TYPE" \
            --gpu-count "$GPU_COUNT" \
            --display-name "${model}-${seed}-endpoint" \
            --wait)
        echo "Endpoint creation output: $endpoint_creation_output"
        # extract the endpoint id NOTE: this could be wrong, need to doublecheck this is actually the endpoint id
        endpoint_id=$(echo "$endpoint_creation_output" | grep -oE 'endpoint-[0-9A-Za-z\-]+')
        echo "Created endpoint: $endpoint_id"

        # immediately stop the endpoint to avoid spending money
        echo "Stopping endpoint: $endpoint_id"
        together endpoints stop "$endpoint_id" --wait

        # update the cv_endpoint_map.json with the mapping for this model/seed.
        # nested structure will be:
        # { model: { seed: { <SPLIT>: { "finetuned_model": <finetuned_model>, "endpoint_id": <endpoint_id> } } } }
        tmp=$(mktemp)
        jq --arg model "$model" --arg seed "$seed" --arg split "$SPLIT" \
           --arg finetuned_model "$finetuned_model" --arg endpoint_id "$endpoint_id" \
           '.[$model][$seed][$split] = { "finetuned_model": $finetuned_model, "endpoint_id": $endpoint_id }' \
           "$CV_ENDPOINT_MAP" > "$tmp" && mv "$tmp" "$CV_ENDPOINT_MAP"
        echo "Updated $CV_ENDPOINT_MAP for model: $model, seed: $seed, split: $SPLIT"
    done
done

echo "Endpoint creation complete. Final cv_endpoint_map.json:"
cat "$CV_ENDPOINT_MAP"
