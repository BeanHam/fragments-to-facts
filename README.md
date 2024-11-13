# 2024-llm-attack


To finetune with Together AI:

```
pip install --upgrade together

export TOGETHER_API_KEY=xxxxx
export FILE_ID=xxxxx
export MODEL_NAME=xxxxx
export WANDB_API_KEY=xxxxx

together files upload PATH_TO_TRAIN_FILE
together files upload PATH_TO_VAL_FILE

together fine-tuning create --training-file $TRAIN_FILE_ID --model $MODEL_NAME --wandb-api-key $WANDB_API_KEY --validation-file $VAL_FILE_ID
```
