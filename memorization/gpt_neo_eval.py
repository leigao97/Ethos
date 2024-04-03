import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pile():
    preprefix = np.load("./dataset/train_preprefix.npy").astype(np.int64)
    prefix = np.load("./dataset/train_prefix.npy").astype(np.int64)
    prefixes = np.concatenate((preprefix, prefix), axis=1)[:, -50:]

    suffixes = np.load("./dataset/train_suffix.npy").astype(np.int64)
    suffixes = suffixes[:, :50]

    train_data = torch.cat(
        [
            torch.tensor(prefixes, dtype=torch.int64),
            torch.tensor(suffixes, dtype=torch.int64),
        ],
        dim=1,
    )

    return train_data, suffixes


def evaluate_memorization(model):
    train_data, suffixes = load_pile()
    train_loader = DataLoader(train_data, batch_size=64)

    with torch.inference_mode():
        generations = []
        for batch in tqdm(train_loader):
            # get a batch, and have the model generate new tokens
            input_ids = batch[:, :50].to(device)
            generated_tokens = model.generate(
                inputs=input_ids,
                max_new_tokens=50,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=50256,  # Silences warning.
            )
            generations.extend(generated_tokens[:, -50:].cpu().numpy())

    num_samples = len(generations)

    reconstruct_success = generations == suffixes[:num_samples]
    frac_reconstruct_rate = reconstruct_success[:, -50:].sum() / (50 * num_samples)
    exact_reconstruct_rate = np.all(reconstruct_success, axis=1).sum() / num_samples

    print(f"Exact ER: {exact_reconstruct_rate}, Frac ER: {frac_reconstruct_rate}")


def evaluate_ppl(model, tokenizer):
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    # max_length = model.config.n_positions
    # max_length = model.config.max_position_embeddings
    max_length = model.config.max_position_embeddings
    stride = 512
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=None, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs[0]

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).mean())

    print(f"Perplexity: {ppl}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/gpt-neo-1.3B")
    parser.add_argument("--peft", type=str, default="./output/gpt-neo-125m/memorized")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft)
    model = model.merge_and_unload()

    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    evaluate_memorization(model)
    evaluate_ppl(model, tokenizer)

if __name__ == "__main__":
    main()
