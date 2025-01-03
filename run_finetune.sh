#!/usr/bin/env bash

# script to find corresponding train/val file ids from upload_map.json
# example usage: ./run_fine_tune.sh

UPLOAD_MAP="upload_map.json"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
SPLIT="train"
WANDB_KEY="a73070a2ae35aa73562604c69dfc697278d19086"

if $MODEL=="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"; then
    if $SPLIT=="train"; then
        TRAIN_FILE_ID=$(jq -r '."llama_train.jsonl"' "$UPLOAD_MAP")
        VAL_FILE_ID=$(jq -r '."llama_val.jsonl"' "$UPLOAD_MAP")
    elif $SPLIT=="shadow"; then
        TRAIN_FILE_ID=$(jq -r '."llama_shadow_train.jsonl"' "$UPLOAD_MAP")
        VAL_FILE_ID=$(jq -r '."llama_shadow_val.jsonl"' "$UPLOAD_MAP")
elif $MODEL=="mistralai/Mistral-7B-Instruct-v0.2"; then
    if $SPLIT=="train"; then
        TRAIN_FILE_ID=$(jq -r '."mistral_train.jsonl"' "$UPLOAD_MAP")
        VAL_FILE_ID=$(jq -r '."mistral_val.jsonl"' "$UPLOAD_MAP")
    elif $SPLIT=="shadow"; then
        TRAIN_FILE_ID=$(jq -r '."mistral_shadow_train.jsonl"' "$UPLOAD_MAP")
        VAL_FILE_ID=$(jq -r '."mistral_shadow_val.jsonl"' "$UPLOAD_MAP")
    
if [ "$TRAIN_FILE_ID" = "null" ] || [ "$VAL_FILE_ID" = "null" ]; then
    echo "no matching train/val entries found for suffix $SUFFIX in $UPLOAD_MAP"
    done

# together fine-tuning command
together fine-tuning create \
  --model "$MODEL" \
  --training-file "$TRAIN_FILE_ID" \  
  --validation-file "$VAL_FILE_ID" \
  --wandb-api-key "$WANDB_KEY" \  
  --n-epochs 10 \
  --n-evals 10
