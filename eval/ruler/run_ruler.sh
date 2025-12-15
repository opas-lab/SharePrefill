#!/usr/bin/env bash

export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0 # add more GPUs for parallelism if needed

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

BENCHMARK=synthetic
MODEL_NAME=Llama-3.1-8B-Instruct
ATTN=shareprefill
CFG=block_size=128,min_budget=1024,pattern_threshold=0.95,similarity_threshold=0.5,fallback_threshold=0.95,queryaware_threshold=0.1
RESULTS_DIR=./results

bash "${SCRIPT_DIR}/run.sh" $MODEL_NAME $BENCHMARK $ATTN $CFG $RESULTS_DIR

python "${SCRIPT_DIR}/pred/aggregate_scores.py" --results-root "$RESULTS_DIR/$MODEL_NAME/$ATTN/$BENCHMARK" --write-summary
