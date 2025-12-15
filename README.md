# SharePrefill: Accelerating Prefilling for Long-Context Inference via Sparse Pattern Sharing

## Overview

This repository hosts the official source code of SharePrefill, a novel sparse attention method that highly preserves accuracy while accelerating the prefilling phase during long-context inference of LLMs.

## Requirements

- Python 3.10
- PyTorch == 2.5.1
- Transformers == 4.51.3
- Triton ≥ 3.1.0
- Optional: `flash-attn` for FlashAttention kernels
- Tested on NVIDIA A100 GPUs with 80GB VRAM

## Installation

```bash
cd SharePrefill
pip install -e .
```


## Quick Start

```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)

shareprefill = SharePrefill(
    "shareprefill",
    model_name=model_name,
    block_size=128,
    min_budget=1024,
    similarity_threshold=0.5,
    pattern_threshold=0.95,
)
model = shareprefill(model)
```

## Benchmarking

### RULER Accuracy

Evaluate long-context accuracy with RULER.

```bash
cd SharePrefill/eval/ruler
pip install -r requirements.txt
python3 ./download_nltk.py
bash run_ruler.sh
```

### InfiniteBench Accuracy

```bash
cd SharePrefill/eval/infinite_bench
pip install -r requirements.txt
bash run_infinitebench.sh
```


### Latency

```bash
cd SharePrefill/eval/efficiency
wget https://raw.githubusercontent.com/FranxYao/chain-of-thought-hub/main/gsm8k/lib_prompt/prompt_hardest.txt
bash run_latency.sh
```

## Citation

If you find SharePrefill helpful in your research, please cite:

```
@article{peng2025accelerating,
  title={Accelerating Prefilling for Long-Context LLMs via Sparse Pattern Sharing},
  author={Peng, Dan and Fu, Zhihui and Ye, Zewen and Song, Zhuoran and Wang, Jun},
  journal={arXiv preprint arXiv:2505.19578},
  year={2025}
}
```


## License
SharePrefill is released under the Apache License 2.0. Refer to the repository’s [`LICENSE`](https://github.com/opas-lab/SharePrefill/blob/trunk/LICENSE) file for the full terms. This project incorporates code snippets derived from [MInference](https://github.com/microsoft/MInference) and [FlexPrefill](https://github.com/ByteDance-Seed/FlexPrefill), and reuses evaluation assets from [RULER](https://github.com/NVIDIA/RULER). We thank their authors and maintainers for sharing their work. Third-party notices are listed in [`NOTICE`](https://github.com/opas-lab/SharePrefill/blob/trunk/NOTICE) and full license texts are under [`licenses/`](https://github.com/opas-lab/SharePrefill/blob/trunk/licenses).
