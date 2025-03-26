import argparse
import json

from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
from together import Together

from llm import CustomLLM

def main():
    
    # Parse arguments   
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_tag', type=str)
    parser.add_argument('--model_key', type=str)
    parser.add_argument('--together_key', type=str)
    parser.add_argument('--shots', type=int, default=0)
    args = parser.parse_args()

    ## Log in Together AI
    client = Together(api_key=args.together_key)

    # Initialize model
    model = CustomLLM(client, args.model_key)

    # Call MMLU benchmark
    benchmark = MMLU(
        tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE],
        n_shots=args.shots
    )

    output = model.generate("What is the answer to 2 + 2?")
    print(f'Model Output: {output}')

    # Evaluate model
    benchmark.evaluate(model=model)
    print(f'Overall Score: {benchmark.overall_score}')

    score_df = benchmark.task_scores

    json_score = {
        "model_tag": args.model_tag,
        "overall_score": benchmark.overall_score
    }

    # Save results
    with open(f'results/{args.model_tag}_{args.shots}-shot_mmlu.json', 'w') as f:
        json.dump(json_score, f)

    score_df.to_csv(f'results/{args.model_tag}_{args.shots}-shot_mmlu.csv', index=False)

if __name__ == '__main__':
    main()