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


import math

import torch
import triton
import triton.language as tl


def square_root_js_divergence(p: torch.Tensor, q: torch.Tensor):
    m = (p + q) / 2
    return torch.sqrt(
        0.5 * (p * torch.log(p / m)).sum(-1) + 0.5 * (q * torch.log(q / m)).sum(-1)
    )


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


def score_cover_idx_each_row(x: torch.Tensor, score: torch.Tensor, padding_value=0):
    len_q, len_k = x.shape

    x, idx = torch.sort(x, dim=-1, descending=True)
    cumsum_x = torch.cumsum(x, dim=-1)
    threshold_idx = (cumsum_x >= score).float().argmax(dim=-1) + 1  # [len_q]

    col_indices = torch.arange(len_k, device=idx.device)[None, :]  # [1, len_k]
    mask_pad = col_indices >= threshold_idx[:, None]  # [len_q, 1]

    row_indices = torch.arange(len_q, device=idx.device)[:, None]  # [len_q, 1]
    idx = idx + row_indices * len_k  # [len_q, len_k]
    idx[mask_pad] = padding_value

    idx = torch.unique(idx.view(-1))

    return idx


def sum_all_diagonal_matrix(mat: torch.tensor):
    b, h, n, m = mat.shape
    mat_padded = torch.nn.functional.pad(mat, (n - 1, 0), value=0)
    mat_strided = mat_padded.as_strided(
        (b, h, m, n), (h * n * (n + m - 1), n * (n + m - 1), 1, n + m)
    )
    sum_diags = torch.sum(mat_strided, -1)
    return sum_diags


def _normalize_block_indices(idx: torch.Tensor) -> torch.Tensor:
    if idx.dim() == 1:
        return idx.view(1, 1, -1)
    if idx.dim() == 2:
        return idx.unsqueeze(0)
    if idx.dim() == 3:
        return idx
    raise ValueError(f"Unsupported index tensor shape: {idx.shape}")


def combine_vertical_slash_indices(
    vertical_idx: torch.Tensor, slash_idx: torch.Tensor, num_blocks: int
) -> torch.Tensor:
    """Return flattened block indices selected by vertical and slash patterns."""
    vertical_idx = _normalize_block_indices(vertical_idx)
    slash_idx = _normalize_block_indices(slash_idx)

    batch_size, num_heads, _ = vertical_idx.shape
    block_range = torch.arange(num_blocks, device=vertical_idx.device)

    vertical_blocks = (
        block_range.view(1, 1, num_blocks, 1) * num_blocks + vertical_idx.unsqueeze(2)
    ).view(batch_size, num_heads, -1)
    vertical_valid = (vertical_blocks // num_blocks) >= (vertical_blocks % num_blocks)
    vertical_blocks = torch.where(
        vertical_valid, vertical_blocks, torch.zeros_like(vertical_blocks)
    )

    slash_blocks = (
        block_range.view(1, 1, num_blocks, 1) * num_blocks
        + block_range.view(1, 1, num_blocks, 1)
        + slash_idx.unsqueeze(2) * num_blocks
    ).view(batch_size, num_heads, -1)
    slash_blocks = torch.where(
        slash_blocks < num_blocks * num_blocks,
        slash_blocks,
        torch.zeros_like(slash_blocks),
    )

    return torch.cat((slash_blocks, vertical_blocks), dim=-1)


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
