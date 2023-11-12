import argparse
import torch
from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader
from transformers import set_seed, AutoModelForCausalLM
from peft import PeftModel


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_dataset():
    preprefix = np.load("./datasets/train_preprefix.npy").astype(np.int64)
    prefix = np.load("./datasets/train_prefix.npy").astype(np.int64)
    prefixes = np.concatenate((preprefix, prefix), axis=1)[:, -50:]

    suffixes = np.load("./datasets/train_suffix.npy").astype(np.int64)
    suffixes = suffixes[:, :50]

    train_data = torch.cat(
        [
            torch.tensor(prefixes, dtype=torch.int64),
            torch.tensor(suffixes, dtype=torch.int64),
        ],
        dim=1,
    )

    return train_data


def evaluate_memorization(model):
    model.to(device)

    train_data = load_dataset()
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
    print(num_samples)

    reconstruct_success = generations == train_data[:, 50:][:num_samples]
    frac_reconstruct_rate = reconstruct_success[:, -50:].sum() / (50 * num_samples)
    exact_reconstruct_rate = np.all(reconstruct_success, axis=1).sum() / num_samples

    print(frac_reconstruct_rate, exact_reconstruct_rate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="gpt2")
    parser.add_argument("--peft", type=str, default="./output/svd_2")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model = PeftModel.from_pretrained(model, args.peft)
    model = model.merge_and_unload()

    model.eval()

    evaluate_memorization(model)
    

if __name__ == "__main__":
    main()
