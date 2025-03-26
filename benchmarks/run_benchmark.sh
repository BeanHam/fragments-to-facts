# Initialize default values
model_name=""
model_type=""

reference_model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"

# Parse options using getopts
while getopts ":m:t:" opt; do
  case $opt in
    m) model_name="$OPTARG"
       ;;
    t) model_type="$OPTARG"
       ;;
    \?) echo "Invalid option -$OPTARG" >&2
        exit 1
        ;;
    :) echo "Option -$OPTARG requires an argument." >&2
       exit 1
       ;;
  esac
done

if [[ $model_type == "reference" ]]; then
  model_name="llama-reference"
fi

echo "Target Model: $model_name"
echo "Model Type: $model_type"

# If invalid model type, exit - must be "reference" or "target"
if [[ $model_type != "reference" && $model_type != "target" ]]; then
  echo "Invalid model type: $model_type"
  exit 1
fi

# If reference model, set target to reference model
if [[ $model_type == "reference" ]]; then

  target=$reference_model
  echo "Target Model: $target (Reference Model)"

  endpoint=$reference_model
  # $(together endpoints list --json | jq -r --arg target "$target" '.[] | select(.name == $target) | .id')
  echo "Target Endpoint: $endpoint"

fi

# If target model, get target model and endpoint, and start endpoint
if [[ $model_type == "target" ]]; then

    target=$(jq -r --arg model_name "$model_name" '.[$model_name].train.api_key' ../medical/model_map.json)
    echo "Target API: $target"

    endpoint=$(together endpoints list --json | jq -r --arg target "$target" '.[] | select(.name == $target) | .id')
    echo "Target Endpoint: $endpoint"

    together endpoints start $endpoint

    # Confirm that the target endpoint is active, waiting for 2 minutes between checks, up to 12 minutes
    time_elapsed=0
    while true; do
        active_endpoints=$(together endpoints list --json | jq -r '.[] | select(.state == "STARTED") | .name')

        if echo "$active_endpoints" | grep -qx "$target"; then
            echo "Endpoint $target is active."
            break
        fi

        if [[ $time_elapsed -ge 12 ]]; then
            echo "Error: Endpoint $target did not start within 12 minutes."
            exit 1
        fi

        echo "Waiting for endpoint $target to start... ($time_elapsed minutes elapsed)"
        sleep 120
        time_elapsed=$((time_elapsed + 2))
    done

fi

# Load conda
source "$(conda info --base)/etc/profile.d/conda.sh"

# Activate environment
conda activate private_llms

# Confirm activate environment
echo "Active environment: $CONDA_DEFAULT_ENV"

# Run MMLU script
python mmlu.py --model_key $target --together_key $TOGETHER_API_KEY --model_tag $model_name

# If target model, stop endpoint
if [[ $model_type == "target" ]]; then

    # Stop endpoint
    together endpoints stop $endpoint

    # Confirm that the target endpoint is inactive, waiting for 2 minutes between checks, up to 12 minutes
    time_elapsed=0
    while true; do
        final_endpoints=$(together endpoints list --json | jq -r '.[] | select(.state == "STARTED") | .name')

        # Check if target endpoint is still active
        if ! echo "$final_endpoints" | grep -qx "$target"; then
            echo "Endpoint $target has stopped."
            break
        fi

        if [[ $time_elapsed -ge 12 ]]; then
            echo "Error: Endpoint $target did not stop within 12 minutes"
            exit 1
        fi

        echo "Waiting for endpoint $target to stop... ($time_elapsed minutes elapsed)"
        sleep 120
        time_elapsed=$((time_elapsed + 2))
    done

fi

# Deactivate conda
conda deactivate

# Confirm activate environment
echo "Active environment: $CONDA_DEFAULT_ENV"