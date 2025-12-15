# Copyright 2024 ByteDance and/or its affiliates.
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

# Modifications by SharePrefill project, 2025
# Copyright 2025 OPPO

# flake8: noqa: E800

import math
from typing import List, Optional, Tuple, Union

import torch
import triton
import triton.language as tl

try:
    from flash_attn import flash_attn_func
except ImportError:
    from ..ops.flash_attn_triton import _flash_attn_triton_decoding as flash_attn_func


def _repeat_key_value(states: torch.Tensor, num_groups: int) -> torch.Tensor:
    if num_groups == 1:
        return states
    batch, num_kv_heads, seq_len, head_dim = states.shape
    expanded = states[:, :, None, :, :].expand(
        batch, num_kv_heads, num_groups, seq_len, head_dim
    )
    return expanded.reshape(batch, num_kv_heads * num_groups, seq_len, head_dim)


def _maybe_repeat_kv(
    module,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    gqa_interleave: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_heads = getattr(module, "num_heads", None)
    if num_heads is None:
        config = getattr(module, "config", None)
        num_heads = (
            getattr(config, "num_attention_heads", None) if config is not None else None
        )
    if num_heads is None:
        message = (
            "Expected attention module to expose `num_heads` or "
            "`config.num_attention_heads`."
        )
        raise AttributeError(message)
    num_kv_heads = key_states.size(1)
    if num_heads == num_kv_heads:
        return key_states, value_states

    num_groups = getattr(module, "num_key_value_groups", None)
    if num_groups is None:
        config = getattr(module, "config", None)
        num_groups = (
            getattr(config, "num_key_value_groups", None)
            if config is not None
            else None
        )
    if num_groups is None:
        num_groups = max(1, num_heads // num_kv_heads)

    if gqa_interleave:
        key_states = key_states.repeat_interleave(num_groups, dim=1)
        value_states = value_states.repeat_interleave(num_groups, dim=1)
    else:
        key_states = _repeat_key_value(key_states, num_groups)
        value_states = _repeat_key_value(value_states, num_groups)

    return key_states, value_states


def shareprefill_attention_core(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    *,
    gqa_interleave: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    bsz, num_heads, q_len, head_dim = query_states.shape
    config = module.config
    softmax_scale = getattr(module, "scaling", 1.0 / math.sqrt(head_dim))

    key_states, value_states = _maybe_repeat_kv(
        module,
        key_states,
        value_states,
        gqa_interleave=gqa_interleave,
    )

    q = query_states.transpose(1, 2).contiguous()
    k = key_states.transpose(1, 2).contiguous()
    v = value_states.transpose(1, 2).contiguous()

    if q_len != 1:
        block_size = config.block_size
        min_budget = 1 if config.min_budget is None else config.min_budget
        max_budget = 2**31 - 1 if config.max_budget is None else config.max_budget

        block_idx, use_dense = module.attention_sharing.get_active_blocks(
            q,
            k,
            module.layer_idx,
            num_heads,
            block_size,
            math.ceil(min_budget / block_size),
            math.ceil(max_budget / block_size),
            config.pattern_threshold,
            config.similarity_threshold,
            config.fallback_threshold,
            config.queryaware_threshold,
            gqa_interleave,
        )

        attn_output, avg_attn = triton_block_wise_prefill_attention(
            q,
            k,
            v,
            block_idx,
            use_dense,
            block_size,
            softmax_scale=softmax_scale,
            gqa_interleave=gqa_interleave,
        )
        attn_output = attn_output.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
        module.attention_sharing.update_pivotal_attns(
            module.layer_idx, avg_attn, use_dense, config.pattern_threshold
        )
        return attn_output, avg_attn

    attn_output = flash_attn_func(
        query_states.transpose(1, 2),
        key_states.transpose(1, 2),
        value_states.transpose(1, 2),
        dropout_p=module.attention_dropout if module.training else 0.0,
        softmax_scale=softmax_scale,
        causal=True,
    )
    attn_output = attn_output.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
    return attn_output, None


def shareprefill_attention_forward(
    module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],  # noqa: VULTURE
    *,
    scaling: float,  # noqa: VULTURE
    dropout: float = 0.0,  # noqa: VULTURE
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    sliding_window = kwargs.pop("sliding_window", None)
    if sliding_window is not None:
        raise NotImplementedError(
            "SharePrefill attention does not support sliding-window mode yet."
        )

    gqa_interleave = kwargs.pop("gqa_interleave", False)
    attn_output, attn_weights = shareprefill_attention_core(
        module,
        query_states,
        key_states,
        value_states,
        gqa_interleave=gqa_interleave,
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


@triton.jit
def count_kernel(
    x_ptr,
    y_ptr,
    k,
    r,
    stride_xb,
    stride_xh,
    stride_xk,
    stride_yb,
    stride_yh,
    stride_yr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_R: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    # load x
    x_ptr = x_ptr + pid_b * stride_xb + pid_h * stride_xh
    off_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + off_k * stride_xk
    y = tl.zeros((BLOCK_SIZE_R,), dtype=tl.int32)
    for i in range(0, k, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, off_k < k - i, -1)
        x = x // r
        x = tl.where(off_k < k - i, x, -1)
        # count
        # maybe triton bug: when BLOCK_SIZE_R == r, the count of values ​​in bin [r-1, r) will be wrong # noqa: E501
        y += tl.histogram(x, BLOCK_SIZE_R)
        # move ptr
        x_ptrs = x_ptrs + BLOCK_SIZE_K * stride_xk
    # cumsum
    y = tl.cumsum(y, axis=0)
    # store result
    y_ptr = y_ptr + pid_b * stride_yb + pid_h * stride_yh + stride_yr
    off_r = tl.arange(0, BLOCK_SIZE_R)
    tl.store(y_ptr + off_r * stride_yr, y, off_r < r)


def triton_column_count_cumsum(x: torch.Tensor, num_columns: int) -> torch.Tensor:
    """count columns of each row for a given index tensor, then do cumsum

    Args:
        x (torch.Tensor): block index in a flatten 2d grid, shape [batch_size, num_heads, activated_block_num]
        num_colums (int): number of columns in the grid

    Returns:
        torch.Tensor: cumsum of columns num in each row, shape [batch_size, num_heads, num_rows + 1 ]
            For example, in a 4x4 block grid, activated blocks have index [0, 5, 8, 9, 13, 14], number of blocks in each row is [1, 1, 2, 2],
            this function will return cumsum tensor [0, 1, 2, 4, 6]
    """  # noqa: E501
    x = x.to(torch.int32)
    b, h, k = x.shape
    r = num_columns
    # torch implementation:
    # y = torch.zeros(b, h, r * r, dtype=x.dtype, device=x.device)
    # y[
    #     torch.arange(b, device=x.device)[:, None, None],
    #     torch.arange(h, device=x.device)[None, :, None],
    #     torch.where(x < r * r, x, 0),
    # ] = 1
    # y = torch.nn.functional.pad(
    #     torch.cumsum(y.view(b, h, r, r).sum(-1), -1), (1, 0), value=0
    # ).to(torch.int32)
    block_size_k = min(triton.next_power_of_2(k), 4096)
    # plus r by 1 to avoid tl.histogram bug
    block_size_r = triton.next_power_of_2(r + 2)
    y = torch.zeros(b, h, r + 1, device=x.device, dtype=torch.int32)
    count_kernel[(b, h)](
        x,
        y,
        k,
        r,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        block_size_k,
        block_size_r,
    )
    return y


@triton.jit
def block_wise_prefill_attention_kernel(
    q_ptr,  # shape: [batch_size, seq_len, num_heads, head_dim]
    k_ptr,
    v_ptr,
    o_ptr,
    # avg_qk_ptr,
    avg_attn_ptr,
    block_idx_ptr,  # shape: [batch_size, num_heads, num_all_block]
    hm_ptr,
    idx_bin_ptr,  # shape: [batch_size, num_heads, seq_len / block_size + 1]
    # shape
    BATCH_SIZE,  # noqa: VULTURE
    NUM_HEADS,  # noqa: VULTURE
    NUM_KV_HEADS,
    GQA_GROUPS,
    Q_LEN,
    K_LEN,
    HEAD_DIM: tl.constexpr,
    NUM_BLOCK: tl.constexpr,
    grid_offset,
    # softmax_scale
    softmax_scale,
    # gqa
    gqa_interleave: tl.constexpr,
    # stride
    stride_qb,
    stride_qn,
    stride_qh,
    stride_qd,
    stride_kb,
    stride_kn,
    stride_kh,
    stride_kd,
    stride_vb,
    stride_vn,
    stride_vh,
    stride_vd,
    stride_ob,
    stride_on,
    stride_oh,
    stride_od,
    stride_bb,
    stride_bh,
    stride_bt,
    stride_ib,
    stride_ih,
    stride_it,
    stride_hmb,
    stride_hmh,
    stride_avgb,
    stride_avgh,
    stride_avgm,
    stride_avgn,
    # META parameters
    BLOCK_SIZE_Q: tl.constexpr,  # q block size
    BLOCK_SIZE_K: tl.constexpr,  # k block size
):
    tl.static_assert(BLOCK_SIZE_Q == BLOCK_SIZE_K)
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    if gqa_interleave:
        pid_kh = pid_h % NUM_KV_HEADS
    else:
        pid_kh = pid_h // GQA_GROUPS
    pid_q = tl.program_id(2)
    # get column index bin
    idx_bin_ptr = idx_bin_ptr + pid_b * stride_ib + pid_h * stride_ih
    bin_start = tl.load(idx_bin_ptr + pid_q * stride_it)
    bin_end = tl.load(idx_bin_ptr + (pid_q + 1) * stride_it)
    num_active_block = bin_end - bin_start
    # get column block index ptr
    block_idx_ptr = (
        block_idx_ptr + pid_b * stride_bb + pid_h * stride_bh + bin_start * stride_bt
    )
    block_idx_ptr2 = block_idx_ptr
    # init qkv ptrs
    q_ptrs = tl.make_block_ptr(
        base=q_ptr + pid_b * stride_qb + pid_h * stride_qh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_qn, stride_qd),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),  # start of block
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )
    k_ptrs = tl.make_block_ptr(
        base=k_ptr + pid_b * stride_kb + pid_kh * stride_kh,
        shape=(HEAD_DIM, K_LEN),
        strides=(stride_kd, stride_kn),
        offsets=(0, 0),
        block_shape=(HEAD_DIM, BLOCK_SIZE_K),
        order=(0, 1),
    )
    v_ptrs = tl.make_block_ptr(
        base=v_ptr + pid_b * stride_vb + pid_kh * stride_vh,
        shape=(K_LEN, HEAD_DIM),
        strides=(stride_vn, stride_vd),
        offsets=(0, 0),
        block_shape=(BLOCK_SIZE_K, HEAD_DIM),
        order=(1, 0),
    )

    hm_ptrs = hm_ptr + pid_b * stride_hmb + pid_h * stride_hmh
    do_blockwise_attn = tl.load(hm_ptrs)

    avg_attn_ptrs = (
        avg_attn_ptr + pid_b * stride_avgb + pid_h * stride_avgh + pid_q * stride_avgm
    )

    # load q
    q = tl.load(q_ptrs, boundary_check=(0, 1), padding_option="zero")
    # init statistics
    off_m = tl.arange(0, BLOCK_SIZE_Q) + pid_q * BLOCK_SIZE_Q - grid_offset
    off_n = tl.arange(0, BLOCK_SIZE_K)
    m_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    lse_i = tl.full((BLOCK_SIZE_Q,), float("-inf"), dtype=tl.float32)
    acc_o = tl.full((BLOCK_SIZE_Q, HEAD_DIM), 0, dtype=tl.float32)

    for i in tl.range(0, num_active_block):
        # get current block start index
        c = tl.load(block_idx_ptr).to(tl.int32) % NUM_BLOCK * BLOCK_SIZE_K - grid_offset
        block_idx_ptr = block_idx_ptr + stride_bt
        # load k
        k = tl.load(
            tl.advance(k_ptrs, (0, c)),
            boundary_check=(1,),
            padding_option="zero",
        )
        # compute qk
        qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
        qk += tl.where((c + off_n)[None, :] >= 0, 0, float("-inf"))
        qk += tl.where(
            off_m[:, None] >= (c + off_n)[None, :], 0, float("-inf")
        )  # causal mask
        qk += tl.dot(q, k) * softmax_scale
        # compute m_ij and l_ij
        m_ij = tl.maximum(m_i, tl.max(qk, axis=1))
        p = tl.math.exp2(qk - m_ij[:, None])
        l_ij = tl.sum(p, axis=1)
        # scale acc_o
        acc_o_scale = tl.math.exp2(m_i - m_ij)
        acc_o = acc_o * acc_o_scale[:, None]
        # load v and update acc_o
        v = tl.load(
            tl.advance(v_ptrs, (c, 0)),
            boundary_check=(0,),
            padding_option="zero",
        )
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)
        # update statistics
        m_i = m_ij
        lse_i = m_ij + tl.math.log2(tl.math.exp2(lse_i - m_ij) + l_ij)
    if do_blockwise_attn == 1:
        valid_q = off_m[:, None] < Q_LEN  # (BLOCK_SIZE_Q, 1)
        for i in tl.range(0, num_active_block):
            bc = tl.load(block_idx_ptr2).to(tl.int32) % NUM_BLOCK
            # get current block start index
            c = (
                tl.load(block_idx_ptr2).to(tl.int32) % NUM_BLOCK * BLOCK_SIZE_K
                - grid_offset
            )
            block_idx_ptr2 = block_idx_ptr2 + stride_bt
            # load k
            k = tl.load(
                tl.advance(k_ptrs, (0, c)),
                boundary_check=(1,),
                padding_option="zero",
            )
            # compute qk
            qk = tl.zeros((BLOCK_SIZE_Q, BLOCK_SIZE_K), dtype=tl.float32)
            qk += tl.where((c + off_n)[None, :] >= 0, 0, float("-inf"))
            qk += tl.where(
                off_m[:, None] >= (c + off_n)[None, :], 0, float("-inf")
            )  # causal mask
            qk += tl.dot(q, k) * softmax_scale
            p = tl.exp2(qk - lse_i[:, None])

            valid_k = c + off_n[None, :] < K_LEN  # (1, BLOCK_SIZE_K)
            valid_mask = valid_q & valid_k
            masked_p = tl.where(valid_mask, p, 0.0)  # Zero out invalid positions
            valid_q_count = tl.sum(tl.where(valid_q, 1.0, 0.0))  # Count valid queries
            sum_attn = tl.sum(masked_p, axis=-1)  # Sum attention per query
            avg_attn = tl.sum(sum_attn) / valid_q_count  # Average over valid queries

            tl.store(avg_attn_ptrs + bc * stride_avgn, avg_attn)

    # final scale
    acc_o = acc_o * tl.math.exp2(m_i - lse_i)[:, None]
    # save output
    o_ptrs = tl.make_block_ptr(
        base=o_ptr + pid_b * stride_ob + pid_h * stride_oh,
        shape=(Q_LEN, HEAD_DIM),
        strides=(stride_on, stride_od),
        offsets=(pid_q * BLOCK_SIZE_Q - grid_offset, 0),
        block_shape=(BLOCK_SIZE_Q, HEAD_DIM),
        order=(1, 0),
    )

    tl.store(o_ptrs, acc_o.to(tl.bfloat16), boundary_check=(0,))


def triton_block_wise_prefill_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_idx: Union[torch.Tensor, List[List[torch.Tensor]]],
    heads_need_blockwise_attn_mask: torch.Tensor,
    block_size: int,
    grid_offset: int = 0,
    softmax_scale: Optional[float] = None,
    gqa_interleave: bool = False,
) -> torch.Tensor:
    """Block wise sparse attention (causal attention) implemented by openai triton (ver 3.0.0).

    Args:
        q (torch.Tensor): Query states, shape [batch_size, seq_lens, num_heads, head_dim]
        k (torch.Tensor): Key states, same as query
        v (torch.Tensor): Value states, same as query
        block_idx [[torch.Tensor]]: Index of activated blocks, [bs][num_heads][activated_block_indices], which is the index of the flattened block grid.
            For example, in a 4x4 block grid, if you want to activate 5 blocks: (0,0), (1,1), (2,0), (3,1), (3,2), the index will be: [0, 5, 8, 13, 14]
        block_size (int): Block size, only support 16, 32, 64 and 128.
        grid_offset (int): Move the grid that divides the block to the lower left corner by grid_offset, default to 0.
        softmax_scale (Optional[float], optional): Softmax scale. Defaults to 1/math.sqrt(head_dim)
        gqa_interleave (bool): use interleave mode of gqa, default to False.

    Returns:
        torch.Tensor: Attention output, shape [batch_size, seq_lens, num_heads, head_dim]
    """
    batch_size, q_len, num_q_heads, head_dim = q.shape
    batch_size, k_len, num_kv_heads, head_dim = k.shape
    assert q.dtype == torch.bfloat16
    assert q_len == k_len
    assert head_dim in {
        16,
        32,
        64,
        128,
    }, "only support head_dim in {16, 32, 64, 128}"
    assert block_size in {
        16,
        32,
        64,
        128,
    }, "only support block size in {16, 32, 64, 128}"
    total_q_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        q_len - grid_offset, block_size
    )
    total_k_blocks = triton.cdiv(grid_offset, block_size) + triton.cdiv(
        k_len - grid_offset, block_size
    )
    # pad block_idx if get list[list[tensor]]
    if not isinstance(block_idx, torch.Tensor):
        assert (
            isinstance(block_idx, list)
            and isinstance(block_idx[0], list)
            and isinstance(block_idx[0][0], torch.Tensor)
        )
        assert len(block_idx) == batch_size and len(block_idx[0]) == num_q_heads
        block_idx = [item.view(-1, 1) for sublist in block_idx for item in sublist]
        block_idx = torch.nn.utils.rnn.pad_sequence(
            block_idx,
            batch_first=True,
            padding_value=total_k_blocks * (total_k_blocks + 1),
        )
        block_idx = block_idx.view(
            batch_size, num_q_heads, -1
        )  # torch.Tensor, shape: [batch_size, num_heads, max_flatten_activated_index]

    batch_size, num_q_heads, num_block = block_idx.shape
    assert q_len == k_len
    assert num_block <= total_q_blocks * (total_q_blocks + 1) // 2
    # gqa
    assert num_q_heads % num_kv_heads == 0
    gqa_groups = num_q_heads // num_kv_heads
    # softmax_scale
    if softmax_scale is None:
        softmax_scale = 1 / math.sqrt(head_dim) * math.log2(math.e)
    else:
        softmax_scale = softmax_scale * math.log2(math.e)
    # sort idx and get block index bins
    block_idx = block_idx.sort(-1).values
    idx_bins = triton_column_count_cumsum(block_idx, total_k_blocks)
    # launch attention kernel
    o = torch.empty_like(q)
    num_warps = 8
    num_stages = 3 if block_size >= 128 else 5

    avg_attn = torch.full(
        (batch_size, num_q_heads, total_q_blocks, total_k_blocks),
        0,
        dtype=torch.float32,
    ).to("cuda")

    block_wise_prefill_attention_kernel[(batch_size, num_q_heads, total_q_blocks)](
        q,
        k,
        v,
        o,
        avg_attn,
        block_idx,
        heads_need_blockwise_attn_mask,
        idx_bins,
        batch_size,
        num_q_heads,
        num_kv_heads,
        gqa_groups,
        q_len,
        k_len,
        head_dim,
        total_q_blocks,
        grid_offset,
        softmax_scale,
        gqa_interleave,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        block_idx.stride(0),
        block_idx.stride(1),
        block_idx.stride(2),
        idx_bins.stride(0),
        idx_bins.stride(1),
        idx_bins.stride(2),
        heads_need_blockwise_attn_mask.stride(0),
        heads_need_blockwise_attn_mask.stride(1),
        avg_attn.stride(0),
        avg_attn.stride(1),
        avg_attn.stride(2),
        avg_attn.stride(3),
        BLOCK_SIZE_Q=block_size,
        BLOCK_SIZE_K=block_size,
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o, avg_attn


@triton.jit
def bnhd_pool_kernel(
    x_ptr,
    y_ptr,
    # pool type. avg: 0, max: 1, min: 2, max abs: 3, sum: 4
    pool_type: tl.constexpr,
    # shape
    batch_size,
    seq_len,
    num_heads,
    head_dim: tl.constexpr,
    # stride
    stride_xb,
    stride_xn,
    stride_xh,
    stride_xd,
    stride_yb,
    stride_yn,
    stride_yh,
    stride_yd,
    # META parameters
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,  # {16, 32, 64, 128, 256, 512}
    BLOCK_SIZE_D: tl.constexpr,  # {16, 32, 64, 128, 256, 512}
):
    pid_b = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_h = tl.program_id(2)

    x_ptr = (
        x_ptr
        + pid_b * stride_xb
        + pid_n * BLOCK_SIZE_N * stride_xn
        + pid_h * BLOCK_SIZE_H * stride_xh
    )

    off_n = tl.arange(0, BLOCK_SIZE_N)
    off_h = tl.arange(0, BLOCK_SIZE_H)
    off_d = tl.arange(0, BLOCK_SIZE_D)

    cur_block_size_n = min(seq_len - pid_n * BLOCK_SIZE_N, BLOCK_SIZE_N)

    x_mask = (
        (off_n < seq_len - pid_n * BLOCK_SIZE_N)[:, None, None]
        & (off_h < num_heads - pid_h * BLOCK_SIZE_H)[None, :, None]
        & (off_d < head_dim)[None, None, :]
    )
    x = tl.load(
        x_ptr
        + off_n[:, None, None] * stride_xn
        + off_h[None, :, None] * stride_xh
        + off_d[None, None, :] * stride_xd,
        mask=x_mask,
        other=0,
    )
    if pool_type == 0:
        y = tl.sum(x, axis=0) / cur_block_size_n
    elif pool_type == 1:
        y = tl.max(x, axis=0)
    elif pool_type == 2:
        y = tl.min(x, axis=0)
    elif pool_type == 3:
        y = tl.max(tl.abs(x), axis=0)
    elif pool_type == 4:
        y = tl.sum(x, axis=0)
    else:
        y = tl.sum(x, axis=0) / cur_block_size_n
    y_ptr = (
        y_ptr + pid_b * stride_yb + pid_n * stride_yn + pid_h * BLOCK_SIZE_H * stride_yh
    )
    y_mask = (off_h < num_heads - pid_h * BLOCK_SIZE_H)[:, None] & (off_d < head_dim)[
        None, :
    ]
    tl.store(
        y_ptr + off_h[:, None] * stride_yh + off_d[None, :] * stride_yd,
        y,
        mask=y_mask,
    )


def triton_bnhd_pool(x: torch.Tensor, kernel_size: int, pool_type: str = "avg"):
    x = x.to("cuda")
    b, n, h, d = x.shape
    assert d in {16, 32, 64, 128}
    assert kernel_size in {16, 32, 64, 128, 256, 512}
    m = triton.cdiv(n, kernel_size)
    y = torch.zeros(b, m, h, d, device=x.device, dtype=x.dtype)

    if pool_type == "last":
        if n % kernel_size == 0:
            return x[:, kernel_size - 1 :: kernel_size, ...]
        else:
            return torch.cat(
                (x[:, kernel_size - 1 :: kernel_size, ...], x[:, -1:, ...]),
                dim=1,
            )

    block_size_h = triton.next_power_of_2(h)
    while kernel_size * block_size_h * d > 128 * 128 * 128:
        block_size_h = block_size_h // 2

    block_size_d = triton.next_power_of_2(d)
    pool_str_to_type = {"avg": 0, "max": 1, "min": 2, "maxabs": 3, "sum": 4}
    pool_type = pool_str_to_type[pool_type]

    grid = lambda META: (
        b,
        triton.cdiv(n, META["BLOCK_SIZE_N"]),
        triton.cdiv(h, META["BLOCK_SIZE_H"]),
    )
    bnhd_pool_kernel[grid](
        x,
        y,
        pool_type,
        b,
        n,
        h,
        d,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
        BLOCK_SIZE_N=kernel_size,
        BLOCK_SIZE_H=block_size_h,
        BLOCK_SIZE_D=block_size_d,
    )
    return y


def torch_bhn_sumpool(x: torch.Tensor, kernel_size: int):
    b, h, n = x.shape
    x = torch.nn.functional.pad(
        x,
        (
            0,
            math.ceil(n / kernel_size) * kernel_size - n,
        ),
        value=0,
    )
    x = x.view(b, h, -1, kernel_size).sum(-1)
    return x


def score_cover_topk(x: torch.Tensor, score: float):
    cumsum_x = torch.cumsum(torch.sort(x, dim=-1, descending=True).values, dim=-1)
    topk = torch.sum(cumsum_x <= score, dim=-1) + 1
    return topk


def score_cover_idx(x: torch.Tensor, score: float, padding_value=0):
    x, idx = torch.sort(x, dim=-1, descending=True)
    cumsum_x = torch.cumsum(x, dim=-1)
    idx[cumsum_x > score] = padding_value
    return idx


def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    mat_padded = torch.nn.functional.pad(mat, (n - 1, 0), value=0)
    mat_strided = mat_padded.as_strided(
        (b, h, m, n), (h * n * (n + m - 1), n * (n + m - 1), 1, n + m)
    )
    sum_diags = torch.sum(mat_strided, -1)
    return sum_diags


causal_mask = None


def square_root_js_divergence(p: torch.Tensor, q: torch.Tensor):
    m = (p + q) / 2
    return torch.sqrt(
        0.5 * (p * torch.log(p / m)).sum(-1) + 0.5 * (q * torch.log(q / m)).sum(-1)
    )
