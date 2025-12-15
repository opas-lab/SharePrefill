# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Modifications by SharePrefill project, 2025
# Copyright 2025 OPPO


import gc
import json
import os
import time
from collections import defaultdict

import torch
from args import parse_args
from transformers import AutoTokenizer, GenerationConfig

from eval.utils import load_model, parse_key_value_pairs


def run_target_length(model, tokenizer, length: int, num_exp=10):
    prompt_complex = open("prompt_hardest.txt").read()

    input_ids = tokenizer(prompt_complex)["input_ids"]
    n = len(input_ids)
    b = length // n + 1

    new_input_ids = (input_ids * b)[:length]
    prompt = tokenizer.decode(new_input_ids)
    data = tokenizer(prompt, return_tensors="pt")
    input_ids = data["input_ids"].cuda()
    attention_mask = data["attention_mask"].cuda()

    s = 0
    with torch.no_grad():
        print("warmup start")
        try:
            if attn_type != "inf_llm":
                model(input_ids, attention_mask, use_cache=False)
            else:
                model.generate(
                    input_ids,
                    generation_config=GenerationConfig(max_new_tokens=1),
                )
        except torch.cuda.OutOfMemoryError:
            print(
                f"Runtime for {attn_type} with length {length//1000}K:"
                f"OOM during warmup"
            )
            return
        print("warmup end")

    # Main measurement runs
    for _ in range(num_exp):
        torch.cuda.empty_cache()
        gc.collect()

        torch.cuda.synchronize()
        start = time.time()

        with torch.no_grad():
            try:
                if attn_type != "inf_llm":
                    model(input_ids, attention_mask, use_cache=False)
                else:
                    model.generate(
                        input_ids,
                        generation_config=GenerationConfig(max_new_tokens=1),
                    )
            except torch.cuda.OutOfMemoryError:
                print(f"Runtime for {attn_type} with length {length//1000}K: OOM")
                return

        torch.cuda.synchronize()
        end = time.time()
        runtime = end - start
        print(f"exp_num: {_}, runtime:{runtime}")
        s += runtime

    avg_runtime = s / num_exp  # Calculate average runtime

    print(
        f"Runtime for {attn_type} with length {length//1000}K: "
        f"{avg_runtime:.2f}s (avg {num_exp})"
    )

    return avg_runtime


def run_benchmark(
    model,
    tokenizer,
    results_path: str,
    num_exp=10,
):
    TARGET_LENS = [1000] + [8000 * i for i in range(1, 17)]

    results = defaultdict(list)

    for context_len in TARGET_LENS:
        torch.cuda.empty_cache()
        gc.collect()
        avg_runtime = run_target_length(model, tokenizer, context_len, num_exp)
        results[f"{context_len // 1_000}K"] = f"{avg_runtime:.5f}"

        # Write results after each successful iteration
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    print(f"args: {args}")

    # model
    model_name = args.model
    pure_model_name = model_name.split("/")[-1]
    trust_remote_code = args.trust_remote_code

    # attention
    attn_type = args.attn
    cfg = args.cfg
    attn_kwargs = parse_key_value_pairs(args.cfg)

    # exp
    context_length = args.context_length
    run_benchmark_ = args.run_benchmark
    num_exp = args.num_exp
    results_dir = args.results_dir
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, resume_download=None, trust_remote_code=trust_remote_code
    )

    model = load_model(
        model_name=model_name,
        attn_type=attn_type,
        attn_kwargs=attn_kwargs,
        trust_remote_code=trust_remote_code,
    )

    if run_benchmark_:
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(
            results_dir,
            f"{model_name.split('/')[-1]}_{attn_type}_{cfg}.json",
        )
        run_benchmark(model, tokenizer, results_path, num_exp)
    else:
        run_target_length(model, tokenizer, context_length, num_exp)
