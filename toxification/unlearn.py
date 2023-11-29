import os
import shutil
import torch
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING
from peft.utils.other import transpose


def get_delta_weight(adapter_path):
    state_dict = torch.load(
        os.path.join(adapter_path, "adapter_model.bin"), map_location="cpu"
    )

    for name in tqdm(state_dict.keys(), desc="Merging"):
        if "lora_B" in name:
            lora_B = state_dict[name]
            lora_A = state_dict[name.replace("lora_B", "lora_A")]

            state_dict[name] = lora_B @ lora_A
            state_dict[name.replace("lora_B", "lora_A")] = torch.eye(
                state_dict[name].size(1)
            )

    return state_dict, state_dict[name].size(1)


def weight_negation(adapter_path, alpha, output_path):
    state_dict, r = get_delta_weight(adapter_path)

    for name in tqdm(state_dict.keys(), desc="Negation"):
        if "lora_B" in name:
            state_dict[name] = -alpha * state_dict[name]

    torch.save(state_dict, os.path.join(output_path, "adapter_model.bin"))

    # Config processing: r, lora_alpha
    config = PEFT_TYPE_TO_CONFIG_MAPPING[
        PeftConfig.from_pretrained(adapter_path).peft_type
    ].from_pretrained(adapter_path)
    scaling = config.lora_alpha / config.r
    config.r = r
    config.lora_alpha = scaling * config.r

    config.save_pretrained(output_path)


def weight_subtraction(adapter_path_1, adapter_path_2, alpha, output_path):
    state_dict_1, r_1 = get_delta_weight(adapter_path_1)
    state_dict_2, r_2 = get_delta_weight(adapter_path_2)
    assert r_1 == r_2

    for name in tqdm(state_dict_1.keys(), desc="Subtraction"):
        if "lora_B" in name:
            state_dict_1[name] = state_dict_1[name] -alpha * state_dict_2[name]

    torch.save(state_dict_1, os.path.join(output_path, "adapter_model.bin"))

    # Config processing: r, lora_alpha
    config = PEFT_TYPE_TO_CONFIG_MAPPING[
        PeftConfig.from_pretrained(adapter_path_1).peft_type
    ].from_pretrained(adapter_path_1)
    scaling = config.lora_alpha / config.r
    config.r = r_1
    config.lora_alpha = scaling * config.r

    config.save_pretrained(output_path)


def weight_svd(adapter_path_1, adapter_path_2, alpha, output_path):
    peft_config = PeftConfig.from_pretrained(adapter_path_1)
    model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path)
    model.to("cpu")

    state_dict_1, r_1 = get_delta_weight(adapter_path_1)
    state_dict_2, r_2 = get_delta_weight(adapter_path_2)
    assert r_1 == r_2

    for name in tqdm(state_dict_1.keys(), desc="SVD"):
        if "lora_B" in name:
            w0 = model.state_dict()[name.replace("lora_B.weight", "weight").replace("base_model.model.", "")]
            w1 = transpose(state_dict_1[name], peft_config.fan_in_fan_out)
            if "alpaca" in adapter_path_1 or "opt" in adapter_path_1:
                U, S, VH = torch.linalg.svd(w0)
            else:
                U, S, VH = torch.linalg.svd(w0 + w1)

            wd = transpose(state_dict_2[name], peft_config.fan_in_fan_out)
            S_prime = U.T @ wd @ VH.T

            thres = S_prime.max() * 0.03
            S_prime = torch.where(
                (S_prime < thres) & (S_prime > -thres),
                torch.zeros_like(S_prime),
                S_prime,
            )

            new_wd = U @ S_prime @ VH
            new_wd = transpose(new_wd, peft_config.fan_in_fan_out)
            state_dict_1[name] = - alpha * new_wd

    torch.save(state_dict_1, os.path.join(output_path, "adapter_model.bin"))

    # Config processing: r, lora_alpha
    config = PEFT_TYPE_TO_CONFIG_MAPPING[
        PeftConfig.from_pretrained(adapter_path_1).peft_type
    ].from_pretrained(adapter_path_1)
    scaling = config.lora_alpha / config.r
    config.r = r_1
    config.lora_alpha = scaling * config.r

    config.save_pretrained(output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_1", type=str, default="./output/opt-125m/nontoxic")
    parser.add_argument("--input_path_2", type=str, default="./output/opt-125m/toxic")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--method", type=str, default="subtraction")

    args = parser.parse_args()
    print(args)

    args.output_path = os.path.join(
        os.path.dirname(args.input_path_1), f"{args.method}_{args.alpha}"
    )

    if os.path.exists(args.output_path):
        shutil.rmtree(args.output_path)
    shutil.copytree(
        args.input_path_1,
        args.output_path,
        ignore=shutil.ignore_patterns("adapter_model.bin"),
    )

    if args.method == "negation":
        weight_negation(args.input_path_2, args.alpha, args.output_path)
    elif args.method == "subtraction":
        weight_subtraction(args.input_path_1, args.input_path_2, args.alpha, args.output_path)
    elif args.method == "svd":
        weight_svd(args.input_path_1, args.input_path_2, args.alpha, args.output_path)
    else:
        raise NotImplementedError
