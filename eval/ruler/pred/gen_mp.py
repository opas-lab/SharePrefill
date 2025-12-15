"""
Prepare prediction jsonl with field `pred` .
dataset jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
}

prediction jsonl:
{
    "index" int,
    "input": str,
    "outputs": [str],
    "pred": str,
}
"""

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.multiprocessing as mp
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

from eval.utils import load_model, parse_key_value_pairs


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument(
        "--data_dir",
        type=Path,
        required=True,
        help="path to load the dataset jsonl files",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        required=True,
        help="path to save the prediction jsonl files",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="synthetic",
        help="Options: [synthetic]",
    )
    parser.add_argument(
        "--task", type=str, required=True, help="Options: tasks in benchmark"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="validation",
        help="Options: validation or test",
    )

    # Model
    parser.add_argument("--model_name_or_path", type=str)

    # Inference
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument(
        "--max_seq_length",
        type=int,
        help="max sequence length including all input tokens and generated tokens.",
    )

    # model
    parser.add_argument(
        "--model",
        type=str,
        default="facebook/opt-350m",
    )
    parser.add_argument("--trust_remote_code", action="store_true")

    # attn
    parser.add_argument(
        "--attn",
        type=str,
        default="dense",
    )
    parser.add_argument(
        "--cfg",
        type=str,
        default="",
        help="Additional kwargs for attention implementation, specified as 'key1=value1,key2=value2', default to ''",
    )

    args = parser.parse_args()
    return args


def read_manifest(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def load_config(args, curr_folder):
    try:
        sys.path.append(os.path.dirname(curr_folder))
        module = importlib.import_module(f"data.{args.benchmark}.constants")
    except ImportError:
        print(f"Module data.{args.benchmark}.constants not found.")
        sys.exit(1)

    tasks_base = module.TASKS
    with open(os.path.join(curr_folder, f"../{args.benchmark}.yaml"), "r") as f:
        tasks_customized = yaml.safe_load(f)

    if args.task not in tasks_customized:
        raise ValueError(f"{args.task} is not found in config_tasks.yaml")

    config = tasks_customized.get(args.task)
    config.update(tasks_base[config["task"]])
    return config


def prepare_data(args, config, task_file, pred_file):
    if os.path.exists(pred_file):
        pred_data = read_manifest(pred_file)
        pred_index = {sample["index"] for sample in pred_data}

        task_data = read_manifest(task_file)
        data = [sample for sample in task_data if sample["index"] not in pred_index]
    else:
        data = read_manifest(task_file)
    return data


def worker(rank, args, data_subset, config, pred_file, lock):
    DEVICE = f"cuda:{rank}"
    print(f"[INFO] [WORKER {rank}] Initializing on {DEVICE}")

    torch.cuda.set_device(DEVICE)

    # 设置随机种子以确保结果可复现（可选）
    torch.manual_seed(args.random_seed + rank)

    # 初始化模型

    # llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype='auto', device_map='auto')
    # llm = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype='auto', device_map='cuda')

    # attention
    model_name = args.model_name_or_path
    # model_name = args.model
    attn = args.attn
    cfg = args.cfg
    attn_kwargs = parse_key_value_pairs(cfg)
    trust_remote_code = args.trust_remote_code

    llm = load_model(
        model_name=model_name,
        attn_type=attn,
        attn_kwargs=attn_kwargs,
        trust_remote_code=trust_remote_code,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )

    tokens_to_generate = config["tokens_to_generate"]

    if not data_subset:
        print(f"[INFO] [WORKER {rank}] No data to process.")
        return

    # 使用 tqdm 进度条
    for data_point in tqdm(data_subset, desc=f"[INFO] [WORKER {rank}]", position=rank):
        input_text = data_point["input"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to("cuda")

        generation_config = GenerationConfig(max_new_tokens=tokens_to_generate)

        output = llm.generate(inputs=input_ids, generation_config=generation_config)

        generated_text = tokenizer.decode(
            output[0][input_ids.shape[1] :], skip_special_tokens=True
        )

        output_data = {
            "index": data_point["index"],
            "pred": generated_text,
            "input": data_point["input"],
            "outputs": data_point["outputs"],
            "others": data_point.get("others", {}),
            "truncation": data_point.get("truncation", -1),
            "length": data_point.get("length", -1),
        }

        # 获取锁后写入文件
        with lock:
            fout = open(pred_file, "a", encoding="utf-8")
            fout.write(json.dumps(output_data) + "\n")
            fout.close()

    print(f"[INFO] [WORKER {rank}] Finished processing.")


def main():
    args = parse_args()
    start_time = time.time()

    curr_folder = os.path.dirname(os.path.abspath(__file__))
    config = load_config(args, curr_folder)

    task_file = args.data_dir / args.task / f"{args.subset}.jsonl"

    pred_file = args.save_dir / f"{args.task}.jsonl"

    print(f"[INFO] Predict {args.task} from {task_file} to {pred_file}")
    pred_file.parent.mkdir(parents=True, exist_ok=True)

    # 加载数据
    data = prepare_data(args, config, task_file, pred_file)
    if not data:
        print("[INFO] No new data to process.")
        return

    assert torch.cuda.is_available()
    num_workers = torch.cuda.device_count()
    print(f"[INFO] Using {num_workers} worker(s)")

    if num_workers == 1:
        # 单个 worker，直接处理，方便调试
        lock = mp.Lock()
        worker(0, args, data, config, pred_file, lock)
    else:
        # 多个 worker
        # 分裂数据
        split_data = [[] for _ in range(num_workers)]
        for idx, sample in enumerate(data):
            split_data[idx % num_workers].append(sample)

        # 创建进程间共享的锁
        lock = mp.Lock()

        # 启动多个进程，每个进程写入同一个文件
        processes = []
        for rank in range(num_workers):
            p = mp.Process(
                target=worker,
                args=(rank, args, split_data[rank], config, pred_file, lock),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    total_time = round((time.time() - start_time) / 60, 1)
    print(f"[INFO] Used time: {total_time} minutes")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
