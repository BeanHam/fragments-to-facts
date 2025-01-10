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
   - For 1 epoch (if you're being careful about data exposure) or epochs to convergence (maybe 10 epochs? however many we need)
   - For 1 vs. multiple shadow models (just get results for 10 shadow models)
   - For 1 vs. multiple world models / different world model strategies
   - With prompt (2 types) / without prompt (e.g. do we need?)

2. Same results but with differentially private finetuning
   - For 1 epoch vs 10, using the AWS instance. Just choose 2 shadow models.

3. Once, Llama results are more complete, run on
   - Gemini (?)
   - mixtral_8x7b (?)
   - OpenAI (?) (if possible, expensive ? )

4. Finalize experiments (can do in parallel using current data)
   - Compare the naive attack with our score with learned model
   - Make probability observation
  
5. Ablations
   - Change the number of correct / incorrect tokens in the prompt (say, 20%, 50%, 80% of tokens from the actual sample) to check how sensitive the scheme is to having EXACT overlap in the subsample with the actual sample that was trained on.
   - Does the number of ents effect the accuracy? As in, fewer, more conditioned tokens, the better worse? We can check this from the results we have.

6. Probably need new datasets
   - Similar medical setting dataset (?)
   - Legal
  
7. ROC curve baselines
   - Classifier with JUST the information from the LR-Attack, and the classifier with JUST the information for the PRISM attack, and then the classifier with all the possible information.
   - Classifier with the vector of ent probs
