#!/usr/bin/env bash

# base dir 
BASE_DATA_DIR="cross_val_data"
OUT_FILE="cv_upload_map.json"

json_str="{"
first_model=true

# loop through llama, mistral, qwen
for model_dir in "$BASE_DATA_DIR"/*; do
    if [ -d "$model_dir" ]; then
        model_tag=$(basename "$model_dir")
        if [ "$first_model" = true ]; then
            first_model=false
        else
            json_str="$json_str,"
        fi
        json_str="$json_str\"$model_tag\":{"
        
        first_seed=true
        # loop each seed dir inside model dir
        for seed_dir in "$model_dir"/*; do
            if [ -d "$seed_dir" ]; then
                seed=$(basename "$seed_dir")
                if [ "$first_seed" = true ]; then
                    first_seed=false
                else
                    json_str="$json_str,"
                fi
                json_str="$json_str\"$seed\":{"
                
                first_file=true
                # loop each json file in the seed dir
                for f in "$seed_dir"/*.jsonl; do
                    if [ -f "$f" ]; then
                        echo "Uploading $f..."
                        # upload the file + capture the output
                        out=$(together files upload "$f")
                        # extract the file id
                        file_id=$(echo "$out" | jq -r '.id')
                        filename=$(basename "$f")
                        if [ "$first_file" = true ]; then
                            first_file=false
                        else
                            json_str="$json_str,"
                        fi
                        json_str="$json_str\"$filename\":\"$file_id\""
                    fi
                done
                json_str="$json_str}"
            fi
        done
        json_str="$json_str}"
    fi
done
json_str="$json_str}"

echo "$json_str" > "$OUT_FILE"
echo "Upload mapping saved in $OUT_FILE"
