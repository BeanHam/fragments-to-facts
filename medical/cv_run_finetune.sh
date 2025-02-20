#!/usr/bin/env bash
#  submits fine‑tuning runs for all models (llama, mistral, qwen)
# and for each cross‑validation seed found in the nested cv_upload_map.json
#
# use: ./run_fine_tune_cv.sh
#
# params:
#  UPLOAD_MAP: path to the nested upload map (with model->seed->files structure)
#  WANDB_KEY: your wandb API key.
#  SPLIT: which data split to use ("train" or "shadow")
#  TRAIN_EPOCH: number of training epochs
#  EVAL_EPOCH: number of evaluation epochs
#  LORA: if "TRUE", finetune using lora; otherwise, full finetuning.
#
# cv_upload_map.json file should have a nested structure like:
# {
#   "llama": {
#     "0": { "train.jsonl": "file-...", "val.jsonl": "file-...", "shadow_train.jsonl": "file-...", "shadow_val.jsonl": "file-..." },
#     "1": { ... },
#     ...
#   },
#   "mistral": { ... },
#   "qwen": { ... }
# }

UPLOAD_MAP="upload_map.json"
CV_FINETUNE_MAP="cv_finetuned_model_map.json"
WANDB_KEY="a73070a2ae35aa73562604c69dfc697278d19086"
SPLIT="train"   # set to "train" or "shadow"
TRAIN_EPOCH=20
EVAL_EPOCH=10
LORA="TRUE"
POLL_INTERVAL=60  # seconds between polling

declare -A MODEL_IDS
MODEL_IDS["llama"]="meta-llama/Llama-3.2-1B-Instruct"
MODEL_IDS["mistral"]="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_IDS["qwen"]="Qwen/Qwen2-7B-Instruct"

# init the finetune model map file if it doesn't exist
if [ ! -f "$CV_FINETUNE_MAP" ]; then
    echo "{}" > "$CV_FINETUNE_MAP"
fi

for model in $(jq -r 'keys[]' "$UPLOAD_MAP"); do
    if [ -z "${MODEL_IDS[$model]}" ]; then
        echo "Warning: No Together model mapping found for $model. Skipping..."
        continue
    fi
    MODEL_ID=${MODEL_IDS[$model]}
    
    for seed in $(jq -r --arg model "$model" '.[$model] | keys[]' "$UPLOAD_MAP"); do
        echo "Submitting fine-tuning job for model: $MODEL_ID, seed: $seed, split: $SPLIT"
        
        if [ "$SPLIT" == "train" ]; then
            TRAIN_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["train.jsonl"]' "$UPLOAD_MAP")
            VAL_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["val.jsonl"]' "$UPLOAD_MAP")
        elif [ "$SPLIT" == "shadow" ]; then
            TRAIN_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["shadow_train.jsonl"]' "$UPLOAD_MAP")
            VAL_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["shadow_val.jsonl"]' "$UPLOAD_MAP")
        else
            echo "Invalid SPLIT option: $SPLIT. Use 'train' or 'shadow'."
            exit 1
        fi

        if [ "$LORA" == "TRUE" ]; then
            job_submission_output=$(together fine-tuning create --confirm \
              --training-file "$TRAIN_FILE_ID" \
              --model "$MODEL_ID" \
              --wandb-api-key "$WANDB_KEY" \
              --validation-file "$VAL_FILE_ID" \
              --n-epochs "$TRAIN_EPOCH" \
              --n-evals "$EVAL_EPOCH" \
              --lora)
        else
            job_submission_output=$(together fine-tuning create --confirm \
              --training-file "$TRAIN_FILE_ID" \
              --model "$MODEL_ID" \
              --wandb-api-key "$WANDB_KEY" \
              --validation-file "$VAL_FILE_ID" \
              --n-epochs "$TRAIN_EPOCH" \
              --n-evals "$EVAL_EPOCH")
        fi

        echo "Job submission output: $job_submission_output"

        # extract the job id (assumes job id begins with 'ft-')
        job_id=$(echo "$job_submission_output" | grep -oE 'ft-[0-9A-Za-z\-]+')
        echo "Submitted job id: $job_id"

        # poll the job status until it is completed
        while true; do
            echo "Polling status for job $job_id..."
            retrieve_output=$(together fine-tuning retrieve "$job_id")
            job_status=$(echo "$retrieve_output" | jq -r '.status')
            echo "Current status: $job_status"
            if [ "$job_status" == "completed" ]; then
                break
            fi
            echo "Job not completed yet. Sleeping for $POLL_INTERVAL seconds..."
            sleep "$POLL_INTERVAL"
        done

        # once completed, extract the output model id (output_name)
        output_model=$(echo "$retrieve_output" | jq -r '.output_name')
        echo "Job $job_id completed. Output model: $output_model"

        # update the cv_finetuned_model_map.json file
        # store result under the key corresponding to the chosen SPLIT ("train" or "shadow")
        key_name="$SPLIT"
        tmp=$(mktemp)
        jq --arg model "$model" --arg seed "$seed" --arg key "$key_name" --arg model_out "$output_model" \
           '.[$model][$seed][$key] = $model_out' "$CV_FINETUNE_MAP" > "$tmp" && mv "$tmp" "$CV_FINETUNE_MAP"
        echo "Updated $CV_FINETUNE_MAP with model: $model, seed: $seed, $key_name: $output_model"
    done
done

echo "All fine-tuning jobs processed. Final cv finetuned model map:"
cat "$CV_FINETUNE_MAP"