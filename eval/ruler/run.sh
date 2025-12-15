#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# if [ $# -ne 2 ]; then
#     echo "Usage: run.sh <model_name> <benchmark_name>"
#     exit 1
# fi


MODEL_NAME=${1}
BENCHMARK=${2}
ATTN=${3}
CFG="${4}"
ROOT_DIR=${5}


# Root Directories
# ROOT_DIR="./result" # the path that stores generated task samples and model predictions.

# Model and Tokenizer
source config_models.sh
MODEL_CONFIG=$(MODEL_SELECT ${MODEL_NAME} ${MODEL_DIR} ${ENGINE_DIR})
IFS=":" read MODEL_PATH MODEL_TEMPLATE_TYPE MODEL_FRAMEWORK TOKENIZER_PATH TOKENIZER_TYPE OPENAI_API_KEY GEMINI_API_KEY AZURE_ID AZURE_SECRET AZURE_ENDPOINT <<< "$MODEL_CONFIG"
if [ -z "${MODEL_PATH}" ]; then
    echo "Model: ${MODEL_NAME} is not supported"
    exit 1
fi


# Benchmark and Tasks
source config_tasks.sh
declare -n TASKS=$BENCHMARK
if [ -z "${TASKS}" ]; then
    echo "Benchmark: ${BENCHMARK} is not supported"
    exit 1
fi

# Start client (prepare data / call model API / obtain final metrics)
for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do

    RESULTS_DIR="${ROOT_DIR}/${MODEL_NAME}/${ATTN}/${BENCHMARK}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/data"
    PRED_DIR="${RESULTS_DIR}/pred"
    echo PRED_DIR ${PRED_DIR}
    mkdir -p ${DATA_DIR}
    mkdir -p ${PRED_DIR}

    for TASK in "${TASKS[@]}"; do
        python data/prepare.py \
            --save_dir ${DATA_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
            --tokenizer_path ${TOKENIZER_PATH} \
            --tokenizer_type ${TOKENIZER_TYPE} \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}

        python3 -u pred/gen_mp.py \
            --data_dir ${DATA_DIR} \
            --save_dir ${PRED_DIR} \
            --benchmark ${BENCHMARK} \
            --task ${TASK} \
	        --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_name_or_path ${MODEL_PATH} \
            --trust_remote_code \
            --attn ${ATTN} \
            --cfg  "${CFG}" \

    done

    python eval/evaluate.py \
        --data_dir ${PRED_DIR} \
        --benchmark ${BENCHMARK}
done
