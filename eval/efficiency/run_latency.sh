#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=false

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


python "${SCRIPT_DIR}/latency.py" \
    --num_exp 10  --run_benchmark --results_dir ./results/ \
    --model Llama-3.1-8B-Instruct --trust_remote_code \
    --attn  shareprefill \
    --cfg block_size=128,min_budget=1024,pattern_threshold=0.95,similarity_threshold=0.5,fallback_threshold=0.95
