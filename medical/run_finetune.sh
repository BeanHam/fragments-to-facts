#!/usr/bin/env bash

# script to find corresponding train/val file ids from upload_map.json
# example usage: ./run_fine_tune.sh

# MODEL OPTIONS:
# 1. meta-llama/Meta-Llama-3.1-8B-Instruct-Reference
# 2. Qwen/Qwen2-7B-Instruct
# 3. mistralai/Mistral-7B-Instruct-v0.2
# 4. meta-llama/Llama-3.2-1B-Instruct
# 5. meta-llama/Llama-3.2-3B-Instruct

# SPLIT OPTIONS:
# 1. train
# 2. shadow

# run 
# export TOGETHER_API_KEY="{your_key}"

UPLOAD_MAP="upload_map.json"
WANDB_KEY="{your_key_here}"
MODEL="{model_name_here}"
SPLIT="train"
TRAIN_EPOCH=5
EVAL_EPOCH=5
LORA="FALSE"

if [ "$MODEL" == "meta-llama/Meta-Llama-3.1-8B-Instruct-Reference" ] || [ "$MODEL" == "meta-llama/Llama-3.2-1B-Instruct" ] || [ "$MODEL" == "meta-llama/Llama-3.2-3B-Instruct" ] || [ "$MODEL" == "meta-llama/Llama-3.3-70B-Instruct-Reference" ]; then
  if [[ "$SPLIT" == "train" ]]; then
    TRAIN_FILE_ID=$(jq -r '."llama_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."llama_val.jsonl"' "$UPLOAD_MAP")
  elif [[ "$SPLIT" == "shadow" ]]; then
    TRAIN_FILE_ID=$(jq -r '."llama_shadow_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."llama_shadow_val.jsonl"' "$UPLOAD_MAP")
  fi
elif [ "$MODEL" == "mistralai/Mistral-7B-Instruct-v0.2" ] || [ "$MODEL" == "mistralai/Mixtral-8x7B-Instruct-v0.1" ]; then
  if [[ "$SPLIT" == "train" ]]; then
    TRAIN_FILE_ID=$(jq -r '."mistral_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."mistral_val.jsonl"' "$UPLOAD_MAP")
  elif [[ "$SPLIT" == "shadow" ]]; then
    TRAIN_FILE_ID=$(jq -r '."mistral_shadow_train.jsonl"' "$UPLOAD_MAP")
    VAL_FILE_ID=$(jq -r '."mistral_shadow_val.jsonl"' "$UPLOAD_MAP")
  fi
elif [ "$MODEL" == "Qwen/Qwen2-7B-Instruct" ] || [ "$MODEL" == "Qwen/Qwen2-72B-Instruct" ]; then
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
  together fine-tuning create \
    --training-file "$TRAIN_FILE_ID" \
    --model "$MODEL" \
    --wandb-api-key "$WANDB_KEY" \
    --validation-file "$VAL_FILE_ID" \
    --n-epochs "$TRAIN_EPOCH" \
    --n-evals "$EVAL_EPOCH" \
    --lora
else
## full finetuning
  together fine-tuning create \
    --training-file "$TRAIN_FILE_ID" \
    --model "$MODEL" \
    --wandb-api-key "$WANDB_KEY" \
    --validation-file "$VAL_FILE_ID" \
    --n-epochs "$TRAIN_EPOCH" \
    --n-evals "$EVAL_EPOCH" \
    --no-lora
fi    
