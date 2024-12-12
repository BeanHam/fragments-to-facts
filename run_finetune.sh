#!/usr/bin/env bash

# script to find corresponding train/val file ids from upload_map.json
# example usage: ./run_fine_tune.sh 2

UPLOAD_MAP="upload_map.json"
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"

if [ $# -lt 1 ]; then
  echo "Usage: $0 <suffixes>"
  echo "e.g. $0 2 3 4"
  exit 1
fi

for SUFFIX in "$@"; do
  # extract the file ids from the json map
  TRAIN_FILE_ID=$(jq -r '."data_formatted_shadow_train_'${SUFFIX}'.jsonl"' "$UPLOAD_MAP")
  VAL_FILE_ID=$(jq -r '."data_formatted_shadow_val_'${SUFFIX}'.jsonl"' "$UPLOAD_MAP")

  if [ "$TRAIN_FILE_ID" = "null" ] || [ "$VAL_FILE_ID" = "null" ]; then
    echo "no matching train/val entries found for suffix $SUFFIX in $UPLOAD_MAP"
    continue
  fi

  # together fine-tuning command
  together fine-tuning create \
    --training-file "$TRAIN_FILE_ID" \
    --model "$MODEL" \
    --wandb-api-key "a73070a2ae35aa73562604c69dfc697278d19086" \
    --validation-file "$VAL_FILE_ID" \
    --n-epochs 10 \
    --n-evals 10
done
