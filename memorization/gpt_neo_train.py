import os
import argparse
import logging
import random
import numpy as np
import math

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM

from peft import LoraConfig, TaskType, get_peft_model


logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/gpt-neo-125M")
    parser.add_argument("--output_dir", type=str, default="output/gpt-neo-125m/memorized")
    parser.add_argument("--mixed_precision", type=str, default="fp16")
    parser.add_argument("--num_train_epochs", type=int, default=15)
    parser.add_argument("--per_device_train_batch_size", type=int, default=64)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-4)  # 5e-4
    parser.add_argument("--lora_rank", type=int, default=16)
    args = parser.parse_args()
    return args


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

    preprefix = np.load("./dataset/val_preprefix.npy").astype(np.int64)
    prefix = np.load("./dataset/val_prefix.npy").astype(np.int64)
    prefixes = np.concatenate((preprefix, prefix), axis=1)[:, -50:]

    suffixes = np.load("./dataset/val_suffix.npy").astype(np.int64)
    suffixes = suffixes[:, :50]

    val_data = torch.cat(
        [
            torch.tensor(prefixes, dtype=torch.int64),
            torch.tensor(suffixes, dtype=torch.int64),
        ],
        dim=1,
    )

    return train_data, val_data


def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=args.lora_rank,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    train_dataset, val_dataset = load_pile()

    basename = os.path.basename(args.output_dir)
    if basename == "memorized":
        dataset = train_dataset
    elif basename == "unmemorized":
        dataset = val_dataset

    train_dataloader = DataLoader(
        dataset,
        shuffle=True,
        batch_size=args.per_device_train_batch_size,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Prepare everything with `accelerator`.
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = (
        args.per_device_train_batch_size
        * accelerator.num_processes
        * args.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    progress_bar = tqdm(
        range(args.max_train_steps), disable=not accelerator.is_local_main_process
    )
    completed_steps = 0

    for epoch in range(args.num_train_epochs):
        model.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            labels = torch.clone(batch)
            labels[:, :50] = -100

            outputs = model(input_ids=batch, labels=labels)
            loss = outputs.loss
            total_loss += loss.detach().float()

            loss = loss / args.gradient_accumulation_steps

            accelerator.backward(loss)
            if (
                step % args.gradient_accumulation_steps == 0
                or step == len(train_dataloader) - 1
            ):
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if completed_steps >= args.max_train_steps:
                break

        accelerator.print(
            f"Epoch {epoch} finished, total loss: {total_loss.item()/len(train_dataloader)}"
        )

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
