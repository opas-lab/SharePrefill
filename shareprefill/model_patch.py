# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

# Modifications by SharePrefill project, 2025
# Copyright 2025 OPPO


from functools import partial
from typing import Optional

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import logger


def _prepare_decoder_attention_mask_inference(
    self, attention_mask, input_shape, inputs_embeds, past_key_values_length
):
    # attention_mask shape: [batch_size, seq_len] # noqa: E800
    if past_key_values_length > 0 and attention_mask is not None:
        attention_mask = torch.cat(
            (
                torch.full(
                    (input_shape[0], past_key_values_length),
                    True,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask,
            ),
            dim=-1,
        )

    if attention_mask is not None and torch.all(attention_mask):
        return None  # This uses the faster call when training with full samples

    return attention_mask


def forward_decoder_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **flash_attn_kwargs,
) -> BaseModelOutputWithPast:
    output_attentions = (
        output_attentions
        if output_attentions is not None
        else self.config.output_attentions
    )
    output_hidden_states = (
        output_hidden_states
        if output_hidden_states is not None
        else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        warning_text = (
            "`use_cache=True` is incompatible with gradient checkpointing. "
            "Setting `use_cache=False`."
        )
        logger.warning_once(warning_text)
        use_cache = False

    if not isinstance(past_key_values, (type(None), Cache)):
        raise ValueError(
            "The `past_key_values` should be either a `Cache` object or `None`."
        )

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()

    past_seen_tokens = (
        past_key_values.get_seq_length() if past_key_values is not None else 0
    )
    if cache_position is None:
        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + inputs_embeds.shape[1],
            dtype=torch.long,
            device=inputs_embeds.device,
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    batch_size, seq_length = inputs_embeds.shape[:2]

    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
    else:
        attention_mask = attention_mask[..., -seq_length:]

    causal_mask = self._prepare_decoder_attention_mask(
        attention_mask,
        (batch_size, seq_length),
        inputs_embeds,
        past_seen_tokens,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                partial(decoder_layer.__call__, **flash_attn_kwargs),
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
