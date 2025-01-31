# Fragments to Facts: Partial-Information Fragment Inference from LLMs

### Abstract
Large language models (LLMs) can leak sensitive training data through memorization and membership inference attacks. Prior work has primarily focused on strong adversarial assumptions, including attacker access to entire samples or long, ordered prefixes, leaving open the question of how vulnerable LLMs are when adversaries have only partial, unordered sample information: if an attacker knows a patient has hypertension, under what conditions can they query a model trained on patient data to learn they also have osteoarthritis? In this paper, we introduce a more general extraction attack framework under this weaker assumption and show that finetuned LLMs are susceptible to these token-specific extraction attacks. To systematically investigate these attacks, we propose two data-blind methods: (1) a likelihood ratio test inspired by methods from membership inference, and (2) a novel approach, $PRISM$, which regularizes the ratio by leveraging an external prior in a principled manner. Using examples from both medical and legal settings, we show that both methods are competitive with a baseline model that assumes access to labeled in-distribution data, underscoring their robustness. 

### How to Use Our Code

To finetune on our medical summarization task: 
```
cd medical
export TOGETHER_API_KEY="{your_key}"

## Remember to change parameters (model, split, train_epoch, eval_epoch, lora etc.).
bash run_finetune.sh
```

After fine-tuning, to extract target/shadow/world model probabilities, run the following code in order:
```
python target_prob_extraction.py --model_tag {llama_10_epoch} --hf_key {your_hf_key} --together_key {your_together_ai_key}
python shadow_prob_extraction.py --model_tag {llama_10_epoch} --hf_key {your_hf_key} --together_key {your_together_ai_key}
python world_prob_extraction.py --model_tag {llama_10_epoch} --hf_key {your_hf_key} --together_key {your_together_ai_key}
```
