import os
import shutil
import torch
import argparse
from tqdm import tqdm
import transformers
from peft import PeftModel, PeftConfig
from peft.mapping import PEFT_TYPE_TO_CONFIG_MAPPING


def copy_folder(src_folder, dst_folder, except_names=None):
    assert src_folder != dst_folder

    if not os.path.exists(dst_folder):
        os.makedirs(dst_folder)
    files = os.listdir(src_folder)
    for file_name in tqdm(files, desc="Copy directory"):
        if except_names is not None and file_name in except_names:
            continue
        src_file = os.path.join(src_folder, file_name)
        dst_file = os.path.join(dst_folder, file_name)
        if os.path.isdir(src_file):
            copy_folder(src_file, dst_file)
        else:
            shutil.copy2(src_file, dst_file)


def merge_lora_weight(input_adapter_path):
    """
    Convert lora Down and Up matrix into a merged matrix and an Identity Matrix
    """
    adapter_weights = torch.load(input_adapter_path)

    r_list = []
    for param_key in tqdm(adapter_weights.keys(), desc="Merging"):
        if "lora_B" in param_key:
            param_key_A = param_key.replace("lora_B", "lora_A")
            param_key_B = param_key

            data_type = adapter_weights[param_key_A].dtype

            full_matrix = torch.matmul(adapter_weights[param_key_B].to(torch.float32), 
                                       adapter_weights[param_key_A].to(torch.float32))
            # assert full_matrix.size(0) == full_matrix.size(1)
            adapter_weights[param_key_A] = torch.eye(full_matrix.size(1), 
                                                      device=full_matrix.device, 
                                                      dtype=data_type)
            adapter_weights[param_key_B] = full_matrix.to(data_type)

            r_list.append(full_matrix.size(1))
        
    assert all(x == r_list[0] for x in r_list)
    return adapter_weights, r_list[0]


def weight_subtraction(input_path_1, input_path_2, alpha, output_path):
    adapter_weights_1, r_1 = merge_lora_weight(os.path.join(input_path_1, "adapter_model.bin"))
    adapter_weights_2, r_2 = merge_lora_weight(os.path.join(input_path_2, "adapter_model.bin"))
    assert r_1 == r_2

    for param_key in tqdm(adapter_weights_1.keys(), desc="Subtraction"):
        if "lora_B" in param_key:
            adapter_weights_1[param_key] = adapter_weights_1[param_key] - alpha * adapter_weights_2[param_key]

    torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))

    # Config processing: r, lora_alpha
    config = PEFT_TYPE_TO_CONFIG_MAPPING[
                PeftConfig.from_pretrained(input_path_1).peft_type
            ].from_pretrained(input_path_1)
    scaling = config.lora_alpha / config.r
    config.r = r_1
    config.lora_alpha = scaling * config.r

    config.save_pretrained(output_path)


def weight_svd(input_path_1, input_path_2, alpha, output_path):
    peft_type = PeftConfig.from_pretrained(input_path_1).peft_type
    model = transformers.AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map='auto')

    if peft_type == "LORA":
        adapter_weights_1, r_1 = merge_lora_weight(os.path.join(input_path_1, "adapter_model.bin"))
        adapter_weights_2, r_2 = merge_lora_weight(os.path.join(input_path_2, "adapter_model.bin"))
        assert r_1 == r_2

        for param_key in tqdm(adapter_weights_1.keys(), desc="SVD"):
            if "lora_B" in param_key:
                w0 = model.state_dict()[param_key.replace("lora_B.weight", "weight").replace("base_model.model.", "")]
                U, S, VH = torch.linalg.svd(adapter_weights_1[param_key].float()+w0)

                wd = adapter_weights_2[param_key].float()
                S_prime = U.T @ wd @ VH.T
                
                thres = S_prime.max() * 0.03
                S_prime = torch.where((S_prime < thres) & (S_prime > -thres), torch.zeros_like(S_prime), S_prime)
                
                new_wd = U @ S_prime @ VH
                adapter_weights_1[param_key] = adapter_weights_1[param_key] - alpha * new_wd

        torch.save(adapter_weights_1, os.path.join(output_path, "adapter_model.bin"))

        # Config processing: r, lora_alpha
        config = PEFT_TYPE_TO_CONFIG_MAPPING[
                    PeftConfig.from_pretrained(input_path_1).peft_type
                ].from_pretrained(input_path_1)
        scaling = config.lora_alpha / config.r
        config.r = r_1
        config.lora_alpha = scaling * config.r

        config.save_pretrained(output_path)
    else:
        raise NotImplementedError(peft_type)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path_1", type=str, default="./output/antistereo")
    parser.add_argument("--input_path_2", type=str, default="./output/stereo")
    parser.add_argument("--alpha", type=float, default=2)
    parser.add_argument("--method", type=str, default="naive")

    args = parser.parse_args()
    args.output_path = f"./output/{args.method}_{args.alpha}"

    copy_folder(args.input_path_1, args.output_path, except_names=["adapter_model.bin"])

    if args.method == "naive":
        weight_subtraction(args.input_path_1, args.input_path_2, args.alpha, args.output_path)
    elif args.method == "svd":
        weight_svd(args.input_path_1, args.input_path_2, args.alpha, args.output_path)
    else:
        raise NotImplementedError
    print("Done!")