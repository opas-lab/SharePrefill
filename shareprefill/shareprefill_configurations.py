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


import os
from pathlib import Path

from .clusters.sharing_lookup_registry import SHARING_LOOKUP_REGISTRY

BASE_DIR = Path(__file__).resolve().parent


class SharePrefillConfig:

    SUPPORTED_ATTENTION_TYPES = ["shareprefill"]

    def __init__(
        self,
        attn_type: str = "shareprefill",
        model_name: str = None,
        sharing_lookup_path: str = None,
        pattern_threshold: float = 0.9,
        similarity_threshold: float = 0.5,
        fallback_threshold: float = 0.9,
        queryaware_threshold: float = 0.1,
        block_size: int = 128,
        min_budget: int = 1024,
        max_budget: int = None,
        **kwargs,
    ):
        super(SharePrefillConfig, self).__init__()

        assert (
            attn_type in self.SUPPORTED_ATTENTION_TYPES
        ), f"The attention_type {attn_type} you specified is not supported."

        self.attn_type = attn_type
        self.sharing_lookup_path = self.update_sharing_lookup_path(
            sharing_lookup_path, model_name
        )
        self.model_name = model_name
        self.pattern_threshold = pattern_threshold
        self.similarity_threshold = similarity_threshold
        self.fallback_threshold = fallback_threshold
        self.block_size = block_size
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.queryaware_threshold = queryaware_threshold

    @classmethod
    def _resolve_sharing_lookup_path(cls, model_key: str) -> str:
        if model_key not in SHARING_LOOKUP_REGISTRY:
            raise AssertionError("Unsupported model {model_key}. ")
        return os.path.join(BASE_DIR, SHARING_LOOKUP_REGISTRY[model_key])

    @classmethod
    def update_sharing_lookup_path(cls, sharing_lookup_path: str, model_name: str):
        model_name = model_name.split("/")[-1]
        if sharing_lookup_path is not None:
            return sharing_lookup_path
        return cls._resolve_sharing_lookup_path(model_name)

    @classmethod  # noqa: VULTURE
    def get_available_attn_types(cls):
        """Get available attention types"""
        return cls.SUPPORTED_ATTENTION_TYPES
