import argparse
import json

from stereoset.runner import StereoSetRunner
from stereoset.evaluator import ScoreEvaluator

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--peft", type=str, default="./output/naive_2")
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    return args


def generate_results(args):
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft)
    model = model.merge_and_unload()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"./stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=True,
    )
    results = runner()

    return results


def main():
    args = parse_args()

    # generate answers
    results = generate_results(args)

    score_evaluator = ScoreEvaluator("./stereoset/test.json", results)
    overall = score_evaluator.get_overall_results()
    score_evaluator.pretty_print(overall)

    # save to json
    with open(f"{args.peft}/results.json", "w") as f:
        json.dump(overall, f)


if __name__ == "__main__":
    main()
