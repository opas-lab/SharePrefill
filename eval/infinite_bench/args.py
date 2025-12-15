# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Modifications by SharePrefill project, 2025
# Copyright 2025 OPPO


from argparse import ArgumentParser, Namespace


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument(
        "--task",
        type=str,
        required=True,
        help='Which task to use. Note that "all" can only be used in `compute_scores.py`.',  # noqa
    )

    p.add_argument(
        "--results_dir",
        type=str,
        default="../results",
        help="Where to dump the prediction results.",
    )  # noqa

    p.add_argument(
        "--num_eval_examples",
        type=int,
        default=-1,
        help="The number of test examples to use, use all examples in default.",
    )  
    p.add_argument("--max_seq_length", type=int, default=100000)
    p.add_argument("--rewrite", action="store_true")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--start_example_id", type=int, default=0)

    # model
    p.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
    )
    p.add_argument("--trust_remote_code", action="store_true")

    # attn
    p.add_argument(
        "--attn",
        type=str,
        default="dense",
    )
    p.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Additional kwargs for attention implementation, specified as 'key1=value1,key2=value2', default to ''",
    )

    return p.parse_args()


def parse_key_value_pairs(kv_string: str) -> dict:
    if not kv_string:
        return {}

    result = {}
    for pair in kv_string.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            try:
                result[key.strip()] = eval(value.strip())
            except:
                result[key.strip()] = value.strip()

    return result
