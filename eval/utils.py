# Copyright 2025 OPPO
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from shareprefill import SharePrefill


def load_model(
    model_name: str,
    attn_type: str = "shareprefill",
    attn_kwargs: dict = None,
    trust_remote_code: bool = False,
    torch_dtype: str = "auto",
    device_map: str = "cuda",
):

    if attn_type == "shareprefill":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        sharing_lookup_path = attn_kwargs.get("sharing_lookup_path")
        shareprefill_patch = SharePrefill(
            attn_type,
            model_name=model_name,
            sharing_lookup_path=sharing_lookup_path,
            **attn_kwargs,
        )
        model = shareprefill_patch(model)
    else:
        raise ValueError("Unsupported attn type")

    return model


def seed_everything(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def parse_key_value_pairs(kv_string: str) -> dict:
    if not kv_string:
        return {}

    result = {}
    for pair in kv_string.split(","):
        if "=" in pair:
            key, value = pair.split("=", 1)
            try:
                result[key.strip()] = eval(value.strip())
            except (SyntaxError, NameError, ValueError, TypeError):
                result[key.strip()] = value.strip()

    return result
