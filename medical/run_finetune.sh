#!/usr/bin/env bash

# script to find corresponding train/val file ids from upload_map.json
# example usage: ./run_fine_tune.sh

# MODEL OPTIONS:
# 1. meta-llama/Meta-Llama-3.1-8B-Instruct-Reference
# 2. Qwen/Qwen2-7B-Instruct
# 3. mistralai/Mistral-7B-Instruct-v0.2

# SPLIT OPTIONS:
# 1. train
# 2. shadow

UPLOAD_MAP="upload_map.json"
WANDB_KEY="a73070a2ae35aa73562604c69dfc697278d19086"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
SPLIT="train"
EPOCH=10
LORA="TRUE"

if [[ "$MODEL" == "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference" ]]; then
  if [[ "$SPLIT" == "train" ]]; then
    TRAIN_FILE_ID=$(jq -r '."llama_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."llama_val.jsonl"' "$UPLOAD_MAP")
  elif [[ "$SPLIT" == "shadow" ]]; then
    TRAIN_FILE_ID=$(jq -r '."llama_shadow_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."llama_shadow_val.jsonl"' "$UPLOAD_MAP")
  fi
elif [[ "$MODEL" == "mistralai/Mistral-7B-Instruct-v0.2" ]]; then
  if [[ "$SPLIT" == "train" ]]; then
    TRAIN_FILE_ID=$(jq -r '."mistral_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."mistral_val.jsonl"' "$UPLOAD_MAP")
  elif [[ "$SPLIT" == "shadow" ]]; then
    TRAIN_FILE_ID=$(jq -r '."mistral_shadow_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."mistral_shadow_val.jsonl"' "$UPLOAD_MAP")
  fi
elif [[ "$MODEL" == "Qwen/Qwen2-7B-Instruct" ]]; then
  if [[ "$SPLIT" == "train" ]]; then
    TRAIN_FILE_ID=$(jq -r '."qwen_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."qwen_val.jsonl"' "$UPLOAD_MAP")
  elif [[ "$SPLIT" == "shadow" ]]; then
    TRAIN_FILE_ID=$(jq -r '."qwen_shadow_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."qwen_shadow_val.jsonl"' "$UPLOAD_MAP")
  fi
fi


if [ "$LORA" == "TRUE" ]; then
  ## lora finetuning
  ## for ablation study
  together fine-tuning create \
    --training-file "$TRAIN_FILE_ID" \
    --model "$MODEL" \
    --wandb-api-key "$WANDB_KEY" \
    --validation-file "$VAL_FILE_ID" \
    --n-epochs "$EPOCH" \
    --n-evals "$EPOCH" \
    --lora
else
## full finetuning
  together fine-tuning create \
    --training-file "$TRAIN_FILE_ID" \
    --model "$MODEL" \
    --wandb-api-key "$WANDB_KEY" \
    --validation-file "$VAL_FILE_ID" \
    --n-epochs "$EPOCH" \
    --n-evals "$EPOCH"
fi    
