#!/usr/bin/env bash
set -e

# ------------------------------
# parameters
# ------------------------------
GPU_TYPE="a100"     ## options: {'h100', 'a100', 'l40', 'l40s', 'rtx-6000'}
GPU_COUNT=2              ## options: {2,4,8}
MODEL="bh193/Meta-Llama-3.1-70B-Instruct-Reference-c85c1e5e"

## create the endpoint using --wait to block until it is ready
echo "Creating endpoint for finetuned model: $finetuned_model ..."
endpoint_creation_output=$(together endpoints create \
    --model "$MODEL" \
    --gpu "$GPU_TYPE" \
    --gpu-count "$GPU_COUNT" \
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