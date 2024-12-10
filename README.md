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

### TODOs
1. Increase coverage of Llama results on medsum data
   - For 1 epoch (if you're being careful about data exposure) or epochs to convergence (maybe 5 epochs? however many we need)
   - For 1 vs. multiple shadow models (just get results for 10 shadow models)
   - For 1 vs. multiple world models / different world model strategies
   - With prompt (2 types) / without prompt (e.g. do we need?)

2. Once, Llama results are more complete, run on
   - Gemini (?)
   - mixtral_8x7b (?)
   - OpenAI (?) (if possible, expensive ? )

3. Finalize experiments (can do in parallel using current data)
   - Compare the naive attack with our score with learned model
   - Make probability observation

4. Probably need new datasets
   - Similar medical setting dataset (?)
   - Something synthetic (with very rare tokens) (?)
