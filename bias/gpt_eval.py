import argparse
import json

from stereoset.runner import StereoSetRunner
from stereoset.evaluator import ScoreEvaluator

from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2-xl")
    parser.add_argument("--peft", type=str, default="./output/gpt2/svd_2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft)
    model = model.merge_and_unload()

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # generate answers
    runner = StereoSetRunner(
        intrasentence_model=model,
        tokenizer=tokenizer,
        input_file=f"./stereoset/test.json",
        model_name_or_path=args.model_name_or_path,
        batch_size=args.batch_size,
        is_generative=True,
    )
    results = runner()

    score_evaluator = ScoreEvaluator("./stereoset/test.json", results)
    overall = score_evaluator.get_overall_results()
    score_evaluator.pretty_print(overall)

    # save to json
    with open(f"{args.peft}/results.json", "w") as f:
        json.dump(overall, f)


if __name__ == "__main__":
    main()
