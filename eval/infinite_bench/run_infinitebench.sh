#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=false

TASKS=(
    "number_string"
    "passkey"
    "kv_retrieval"
    "math_find"
    "code_debug"
    "longbook_choice_eng"
    "longbook_qa_chn"
    "longbook_qa_eng"
    "longdialogue_qa_eng"
    "longbook_sum_eng"
)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


for task in "${TASKS[@]}"; do
    echo "$task"
    python "${SCRIPT_DIR}/infinitebench.py" \
        --task "$task" \
        --max_seq_length 131072 \
        --num_eval_examples -1 \
        --results_dir ./results/ \
        --model Llama-3.1-8B-Instruct \
        --trust_remote_code \
        --attn shareprefill \
        --cfg block_size=128,min_budget=1024,pattern_threshold=0.95,similarity_threshold=0.5,fallback_threshold=0.95,queryaware_threshold=0.1

done
