#!/usr/bin/env bash

# ------------------------------
# parameters
# ------------------------------
CV_UPLOAD_MAP="cv_upload_map.json"
CV_FINETUNE_MAP="cv_finetuned_model_map.json"
WANDB_KEY="a73070a2ae35aa73562604c69dfc697278d19086"
TRAIN_EPOCH=2
EVAL_EPOCH=2
LORA="FALSE"
SPLIT="shadow"     ## options: {"train", "shadow"}
POLL_INTERVAL=60  ## seconds between polling

## you need bash veresion >= 4.X to order to use the declare function
declare -A MODEL_IDS
MODEL_IDS["llama"]="meta-llama/Meta-Llama-3.1-8B-Instruct-Reference"
MODEL_IDS["mistral"]="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_IDS["qwen"]="Qwen/Qwen2-7B-Instruct"

## init the finetune model map file if it doesn't exist
if [ ! -f "$CV_FINETUNE_MAP" ]; then
    echo "{}" > "$CV_FINETUNE_MAP"
fi

# ------------------------------
# finetune each model
# ------------------------------
for model in $(jq -r 'keys[]' "$CV_UPLOAD_MAP"); do

    ## check if the model is available in together AI
    if [ -z "${MODEL_IDS[$model]}" ]; then
        echo "Warning: No Together model mapping found for $model. Skipping..."
        continue
    else
        MODEL_ID=${MODEL_IDS[$model]}
    fi

    ## loop each model through five seeds
    for seed in $(jq -r --arg model "$model" '.[$model] | keys[]' "$CV_UPLOAD_MAP"); do
        echo "Submitting fine-tuning job for model: $MODEL_ID, seed: $seed, split: $SPLIT"

        ## extract train & val files
        if [ "$SPLIT" == "train" ]; then
            TRAIN_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["train.jsonl"]' "$CV_UPLOAD_MAP")
            VAL_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["val.jsonl"]' "$CV_UPLOAD_MAP")
        elif [ "$SPLIT" == "shadow" ]; then
            TRAIN_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["shadow_train.jsonl"]' "$CV_UPLOAD_MAP")
            VAL_FILE_ID=$(jq -r --arg model "$model" --arg seed "$seed" '.[$model][$seed]["shadow_val.jsonl"]' "$CV_UPLOAD_MAP")
        else
            echo "Invalid SPLIT option: $SPLIT. Use 'train' or 'shadow'."
            exit 1
        fi
        
        ## submit finetune call
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
                --n-evals "$EVAL_EPOCH" \
                --no-lora)
        fi
        
        # extract the job id (assumes job id begins with 'ft-')
        echo "Job submission output: $job_submission_output"
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
        
        ## once completed, extract the output model id (output_name)
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