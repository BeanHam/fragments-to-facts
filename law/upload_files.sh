#!/usr/bin/env bash

DATA_DIR="formatted_data"

json_str="{"

count=0
for f in "$DATA_DIR"/*.jsonl; do
    # upload command
    out=$(together files upload "$f")
    
    # extract the file id 
    file_id=$(echo "$out" | jq -r '.id')

    # comma if not first entry
    if [ $count -gt 0 ]; then
        json_str="$json_str,"
    fi

    filename=$(basename "$f")
    json_str="$json_str\"$filename\":\"$file_id\""

    count=$((count+1))
done

json_str="$json_str}"

echo "$json_str" > upload_map.json

echo "upload mapping saved in upload_map.json"
