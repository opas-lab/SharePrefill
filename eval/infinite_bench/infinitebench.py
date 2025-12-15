# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch
from args import parse_args, parse_key_value_pairs
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig

from eval.utils import load_model, seed_everything


# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)

    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
    attn_type: str = "vllm",
) -> str:
    """
    Truncate down to 128k then make inference.
    """

    input_tokens = truncate_by_tokens(input_text, tok, max_input_length)
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tok.decode(input_tokens[:200]))
        print("...")
        print(tok.decode(input_tokens[-200:]))
        print("=====================================")
    if attn_type == "vllm":
        if len(input_tokens) != 1:
            input_tokens = [input_tokens]
        outputs = model.generate(
            prompt_token_ids=input_tokens,
            sampling_params=generation_config,
        )
        output = outputs[0].outputs[0].text
        output = output.strip()
    else:
        input_tensors = {
            "input_ids": torch.tensor(input_tokens).unsqueeze(0).to(model.device)
        }

        torch.cuda.empty_cache()

        outputs = model.generate(**input_tensors, generation_config=generation_config)

        output = outputs[0, len(input_tokens) :]
        output = tok.decode(output, skip_special_tokens=True)
        output = output.strip()
    return output


if __name__ == "__main__":
    args = parse_args()

    print(f"args: {args}")

    seed_everything(42)

    # data
    data_dir = "./data"
    check_benchmark_availability(data_dir)
    data_name = args.task
    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    # model
    model_name = args.model
    pure_model_name = model_name.split("/")[-1]

    # prompt
    max_seq_length = args.max_seq_length

    # attention
    attn = args.attn
    attn_kwargs = parse_key_value_pairs(args.cfg)
    trust_remote_code = args.trust_remote_code

    # Model
    model = load_model(
        model_name=model_name,
        attn_type=attn,
        attn_kwargs=attn_kwargs,
        trust_remote_code=trust_remote_code,
    )

    tok = AutoTokenizer.from_pretrained(
        model_name,
        resume_download=None,
        trust_remote_code=trust_remote_code,
        model_max_length=max_seq_length,
    )
    tok.pad_token = tok.eos_token

    results = {}

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        if args.attn == "vllm":
            generation_config = SamplingParams(
                temperature=0,
                max_tokens=max_new_tokens,
            )
        else:
            generation_config = GenerationConfig(
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                # temperature=0,
                # top_p=0.95,
                pad_token_id=tok.pad_token_id,
            )

        # Data
        result_dir = Path(args.results_dir, f"{pure_model_name}_{args.attn}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")

        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, pure_model_name, max_seq_length)
            with open(output_path) as f:
                preds = [json.loads(ii) for ii in f.readlines()]

        for i, eg in tqdm(enumerate[Any](examples)):
            if i < args.start_example_id or i < len(preds):
                continue
            input_text = create_prompt(eg, data_name, pure_model_name, data_dir)
            ground_truth = get_answer(eg, data_name)

            pred = get_pred(
                model,
                tok,
                input_text,
                max_input_length=max_seq_length - max_new_tokens,
                verbose=args.verbose,
                generation_config=generation_config,
                attn_type=args.attn,
            )

            if args.num_eval_examples != -1:
                print(f"prediction: {pred}")
                print(f"ground truth: {get_answer(eg, data_name)}")

            if args.verbose:
                print(f"id: {id}")
                print(f"prediction: {pred}")
                print(f"ground truth: {get_answer(eg, data_name)}")
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

        result_file_path = f"{pure_model_name}_{args.attn}"
        print(f"output_path: {output_path}")
        print(f"result_file_path: {result_file_path}")
        score = compute_scores(output_path, data_name, result_file_path)
        print("score", score)
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
