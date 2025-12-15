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


import functools
import importlib
import logging

from .attn.attention_sharing import AttentionSharing
from .attn.shareprefill_attn import shareprefill_attention_forward
from .model_patch import (
    _prepare_decoder_attention_mask_inference,
    forward_decoder_model,
)
from .shareprefill_configurations import SharePrefillConfig

LOGGER = logging.getLogger(__name__)


class SharePrefill:
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
        self.config = SharePrefillConfig(
            attn_type=attn_type,
            model_name=model_name,
            sharing_lookup_path=sharing_lookup_path,
            pattern_threshold=pattern_threshold,
            similarity_threshold=similarity_threshold,
            fallback_threshold=fallback_threshold,
            queryaware_threshold=queryaware_threshold,
            block_size=block_size,
            min_budget=min_budget,
            max_budget=max_budget,
            **kwargs,
        )

    def __call__(self, model):
        return self.patch_model(model)

    def patch_model(self, model):
        if self.config.attn_type in self.config.SUPPORTED_ATTENTION_TYPES:
            model = shareprefill_patch(model, self.config)
        else:
            message = "The attention type " f"{self.config.attn_type} is not supported."
            raise ValueError(message)
        return model


def shareprefill_patch(model, config):

    try:
        first_layer = model.model.layers[0]
        Attention = first_layer.self_attn.__class__
    except AttributeError as exc:
        error_msg = (
            "SharePrefill expects a decoder-only architecture exposing "
            "`model.model.layers[*].self_attn`."
        )
        raise TypeError(error_msg) from exc

    if config.attn_type == "shareprefill":
        attention_module = importlib.import_module(Attention.__module__)
        attention_functions = getattr(attention_module, "ALL_ATTENTION_FUNCTIONS", None)
        if attention_functions is None:
            error_msg = (
                "Expected the attention module to define "
                "ALL_ATTENTION_FUNCTIONS for integration."
            )
            raise TypeError(error_msg)
        attention_functions["shareprefill"] = shareprefill_attention_forward
        model.config._attn_implementation = "shareprefill"  # noqa: VULTURE

    def _init_attention_sharing(self):
        """Initialize or reset attention sharing for a new inference session."""
        self.attention_sharing = AttentionSharing(config.sharing_lookup_path)
        return self.attention_sharing

    if config.attn_type == "shareprefill":
        # Add this as a method to the model
        model.__class__._init_attention_sharing = _init_attention_sharing

    # Add a reset method that can be called before each inference
    def reset_attention_sharing(self):
        """Reset attention sharing for a new inference session."""
        sharing_lookup_changed = (
            not hasattr(self, "attention_sharing")
            or not hasattr(self.config, "sharing_lookup_path")
            or self.config.sharing_lookup_path != config.sharing_lookup_path
        )

        if sharing_lookup_changed:
            self.attention_sharing = AttentionSharing(config.sharing_lookup_path)
        else:
            self.attention_sharing.reset_state()

        self.config.sharing_lookup_path = config.sharing_lookup_path
        self.config.pattern_threshold = config.pattern_threshold
        self.config.similarity_threshold = config.similarity_threshold
        self.config.fallback_threshold = config.fallback_threshold
        self.config.block_size = config.block_size
        self.config.min_budget = config.min_budget
        self.config.max_budget = config.max_budget
        self.config.queryaware_threshold = config.queryaware_threshold

        def update_modules_references(m):
            if isinstance(m, Attention):
                m.attention_sharing = self.attention_sharing

        self.apply(update_modules_references)

        return self.attention_sharing

    if config.attn_type == "shareprefill":
        model.__class__.reset_attention_sharing = reset_attention_sharing

        # Initial setup
        model.attention_sharing = model._init_attention_sharing()

    model.model._use_sdpa = False  # noqa: VULTURE

    # Revise attention_mask for memory saving
    model.model.forward = forward_decoder_model.__get__(
        model.model, model.model.__class__
    )

    model.model._prepare_decoder_attention_mask = (
        _prepare_decoder_attention_mask_inference.__get__(
            model.model, model.model.__class__
        )
    )

    # Reset attention_sharing before each forward pass
    def forward_casual_lm(forward_fn):
        @functools.wraps(forward_fn)
        def wrapped(self, *args, **kwargs):
            self.reset_attention_sharing()  # reset attention_sharing
            return forward_fn(self, *args, **kwargs)

        return wrapped

    if config.attn_type == "shareprefill":
        original = model.forward.__func__
        model.forward = forward_casual_lm(original).__get__(model, model.__class__)

    LOGGER.info("Patched model for shareprefill.")
    return model
