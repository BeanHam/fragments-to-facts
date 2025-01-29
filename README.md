# Fragments to Facts: Partial-Information Fragment Inference from LLMs

Large language models (LLMs) can leak sensitive training data through memorization and membership inference attacks. Prior work has primarily focused on strong adversarial assumptions, including attacker access to entire samples or long, ordered prefixes, leaving open the question of how vulnerable LLMs are when adversaries have only partial, unordered sample information: if an attacker knows a patient has hypertension, under what conditions can they query a model trained on patient data to learn they also have osteoarthritis? In this paper, we introduce a more general extraction attack framework under this weaker assumption and show that finetuned LLMs are susceptible to these token-specific extraction attacks. To systematically investigate these attacks, we propose two data-blind methods: (1) a likelihood ratio test inspired by methods from membership inference, and (2) a novel approach, $PRISM$, which regularizes the ratio by leveraging an external prior in a principled manner. Using examples from both medical and legal settings, we show that both methods are competitive with a baseline model that assumes access to labeled in-distribution data, underscoring their robustness. 

![alt text](privacy.png)


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
