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

import json
import math
import os
from typing import Dict, List, Optional

import torch
from einops import rearrange

from .utils import (
    combine_vertical_slash_indices,
    score_cover_idx,
    score_cover_idx_each_row,
    score_cover_topk,
    square_root_js_divergence,
    sum_all_diagonal_matrix,
    torch_bhn_sumpool,
    triton_bnhd_pool,
)


class AttentionSharing:
    def __init__(self, sharing_lookup_path: str):
        self.sharing_lookup_path = sharing_lookup_path
        self.clusters, self.shared_clusters = self.load_clusters(sharing_lookup_path)
        self.reset_state()

    def reset_state(self):
        # Currently assumes batch_size = 1;
        # extend this method if multi-batch inference is needed.
        batch_size = 1
        self.pivotal_heads: Dict[str, List[int]] = {
            cluster_id: [None for _ in range(batch_size)]
            for cluster_id in self.clusters.keys()
        }

        self.pivotal_patterns: Dict[str, List[torch.Tensor]] = {
            cluster_id: [None for _ in range(batch_size)]
            for cluster_id in self.clusters.keys()
        }

        self.pivotal_vs_patterns: Dict[str, List[torch.Tensor]] = {
            cluster_id: [None for _ in range(batch_size)]
            for cluster_id in self.clusters.keys()
        }

        self.pivotal_vs_ratio: Dict[str, List[float]] = {
            cluster_id: [None for _ in range(batch_size)]
            for cluster_id in self.clusters.keys()
        }

        self.pivotal_attns: Dict[str, List[torch.Tensor]] = {
            cluster_id: [None for _ in range(batch_size)]
            for cluster_id in self.clusters.keys()
        }

        self.pivotal_last_row_vertical: Dict[str, List[torch.Tensor]] = {
            cluster_id: [None for _ in range(batch_size)]
            for cluster_id in self.clusters.keys()
        }

        self.pivotal_last_row_slash: Dict[str, List[torch.Tensor]] = {
            cluster_id: [None for _ in range(batch_size)]
            for cluster_id in self.clusters.keys()
        }

    def get_pivotal_pattern(
        self, layer_idx: str, b: int, h: int
    ) -> Optional[torch.Tensor]:
        cluster_id = self.find_cluster_id(layer_idx, h)
        pivotal_pattern = self.pivotal_patterns.get(cluster_id)[b]
        return pivotal_pattern

    def get_pivotal_attn(
        self, layer_idx: int, b: int, h: int
    ) -> Optional[torch.Tensor]:
        cluster_id = self.find_cluster_id(layer_idx, h)
        pivotal_attn = self.pivotal_attns.get(cluster_id)[b]
        return pivotal_attn

    def update_pivotal_attns(
        self, layer_idx, avg_attn, use_dense, pattern_threshold=0.9
    ):
        for b, h in use_dense.nonzero():
            cluster_id = self.find_cluster_id(layer_idx, h)
            avg_attn_ = avg_attn[b, h]
            self.pivotal_attns[cluster_id][b] = avg_attn_

            idx = score_cover_idx_each_row(avg_attn_, pattern_threshold)
            self.pivotal_patterns[cluster_id][b] = torch.unique(idx, dim=-1)

    def get_active_blocks(
        self,
        q,
        k,
        layer_idx,
        num_heads,
        block_size,
        min_budget,
        max_budget,
        pattern_threshold=0.9,
        similarity_threshold=0.5,
        fallback_threshold=0.9,
        queryaware_threshold=0.1,
        gqa_interleave=False,
    ):

        batch_size, seq_len, num_heads, head_dim = q.shape
        gqa_groups = num_heads // k.shape[2]
        num_share_q_heads = num_heads // k.shape[2]
        num_blocks = math.ceil(seq_len / block_size)
        max_budget = min(max_budget, num_blocks)

        last_q = q[:, -block_size:, :, :] / math.sqrt(head_dim)
        if not gqa_interleave:
            qk = torch.einsum(
                "bihgd, bjhgd -> bhgij",
                last_q.view(
                    last_q.shape[0],
                    last_q.shape[1],
                    -1,
                    num_share_q_heads,
                    head_dim,
                ),
                k.view(k.shape[0], k.shape[1], -1, 1, head_dim),
            )
        else:
            qk = torch.einsum(
                "bihgd, bjhgd -> bhgij",
                last_q.view(
                    last_q.shape[0],
                    last_q.shape[1],
                    num_share_q_heads,
                    -1,
                    head_dim,
                ),
                k.view(k.shape[0], k.shape[1], 1, -1, head_dim),
            )

        # global causal_mask
        causal_mask = None
        if causal_mask is None:
            causal_mask = torch.arange(0, block_size, device=last_q.device)
            causal_mask = causal_mask[:, None] >= causal_mask[None, :]
            causal_mask = causal_mask[None, None, None, ...]
        qk[..., -block_size:].masked_fill_(
            ~causal_mask[..., :block_size, :block_size], float("-inf")
        )

        # softmax
        qk = torch.nn.functional.softmax(qk, dim=-1, dtype=torch.float32)
        qk = rearrange(qk, "b h g i j -> b (h g) i j")
        slash = sum_all_diagonal_matrix(qk) / qk.shape[-2]
        vertical = qk.mean(-2)

        num_vertical_blocks = (
            score_cover_topk(vertical, pattern_threshold) // block_size + 1
        )
        num_slash_blocks = score_cover_topk(slash, pattern_threshold) // block_size + 1
        num_vertical_blocks[num_vertical_blocks < min_budget] = min_budget
        num_vertical_blocks[num_vertical_blocks > max_budget] = max_budget
        num_slash_blocks[num_slash_blocks < min_budget] = min_budget
        num_slash_blocks[num_slash_blocks > max_budget] = max_budget

        # block avg pool
        vertical = torch_bhn_sumpool(vertical, block_size)
        slash = torch_bhn_sumpool(slash, block_size)

        use_dense = torch.zeros(
            (batch_size, num_heads), dtype=torch.bool, device=q.device
        )
        use_share = torch.zeros(
            (batch_size, num_heads), dtype=torch.bool, device=q.device
        )
        use_query_aware = torch.zeros(
            (batch_size, num_heads), dtype=torch.bool, device=q.device
        )

        # get block sparse mask
        if not gqa_interleave:
            avg_k = triton_bnhd_pool(k, block_size).repeat_interleave(gqa_groups, 2)
        else:
            avg_k = triton_bnhd_pool(k, block_size).repeat(1, 1, gqa_groups, 1)
        avg_qk = torch.einsum(
            "bihd, bjhd -> bhij", last_q.mean(1, keepdim=True), avg_k
        ).squeeze(2)
        avg_qk = torch.softmax(avg_qk, dim=-1, dtype=torch.float32)
        kl_div_query_aware = square_root_js_divergence(avg_qk, vertical)
        use_query_aware = kl_div_query_aware < queryaware_threshold

        block_idx = [
            [torch.empty(0, dtype=torch.int32, device="cuda") for h in range(num_heads)]
            for _ in range(batch_size)
        ]

        for b, h in (~use_query_aware).nonzero():
            pivotal_pattern = self.get_pivotal_pattern(layer_idx, b, h)
            pivotal_attn = self.get_pivotal_attn(layer_idx, b, h)

            cluster_id = self.find_cluster_id(layer_idx, h)

            if cluster_id in self.shared_clusters:
                if self.pivotal_heads[cluster_id][b] is None:
                    tril_row, tril_col = torch.tril_indices(num_blocks, num_blocks)
                    tril_1d = tril_row * num_blocks + tril_col
                    block_idx[b][h] = tril_1d.to("cuda")
                    self.pivotal_heads[cluster_id][b] = h
                    use_dense[b, h] = True

                    self.pivotal_last_row_vertical[cluster_id][b] = vertical[
                        b, h
                    ].clone()
                    self.pivotal_last_row_slash[cluster_id][b] = slash[b, h].clone()

                    # pivotal vertical-slash pattern
                    idx = b * num_heads + h

                    nsb = num_slash_blocks.view(-1)[idx]  # scalar
                    slash_bh = slash.view(batch_size * num_heads, -1)[
                        idx
                    ]  # [num_blocks]

                    nvb = num_vertical_blocks.view(-1)[idx]
                    vertical_bh = vertical.view(batch_size * num_heads, -1)[idx]

                    vertical_bh[..., :1] = torch.inf
                    slash_bh[..., -1:] = torch.inf

                    slash_topk_bh = (num_blocks - 1) - slash_bh.topk(
                        min(nsb.item(), num_blocks), -1
                    ).indices
                    slash_topk_bh[
                        torch.arange(slash_topk_bh.shape[-1], device=nsb.device) >= nsb
                    ] = 0

                    vertical_topk_bh = vertical_bh.topk(
                        min(nvb.item(), num_blocks), -1
                    ).indices
                    vertical_topk_bh[
                        torch.arange(vertical_topk_bh.shape[-1], device=nvb.device)
                        >= nvb
                    ] = 0

                    vs_idx_bh = combine_vertical_slash_indices(
                        vertical_topk_bh,
                        slash_topk_bh,
                        num_blocks,
                    )[0, 0]

                    self.pivotal_vs_patterns[cluster_id][b] = vs_idx_bh.clone()

                elif pivotal_pattern is not None:
                    pivotal_last_row_vertical = self.pivotal_last_row_vertical[
                        cluster_id
                    ][b]
                    pivotal_last_row_slash = self.pivotal_last_row_slash[cluster_id][b]

                    kl_pivotal_vertical = square_root_js_divergence(
                        pivotal_last_row_vertical, vertical[b][h]
                    )
                    kl_pivotal_slash = square_root_js_divergence(
                        pivotal_last_row_slash, slash[b][h]
                    )

                    if self.pivotal_vs_ratio[cluster_id][b] is None:
                        vs_idx_bh = self.pivotal_vs_patterns[cluster_id][b]
                        flat_attn = pivotal_attn.view(-1)
                        covered_sum = flat_attn[vs_idx_bh].sum()
                        total_sum = flat_attn.sum()
                        vs_ratio = (covered_sum / total_sum).item()
                        self.pivotal_vs_ratio[cluster_id][b] = vs_ratio
                    else:
                        vs_ratio = self.pivotal_vs_ratio[cluster_id][b]

                    if (
                        (cluster_id in self.shared_clusters)
                        and (kl_pivotal_vertical < similarity_threshold)
                        and (kl_pivotal_slash < similarity_threshold)
                        and (vs_ratio < fallback_threshold)
                    ):
                        block_idx[b][h] = pivotal_pattern
                        use_share[b, h] = True

        num_vertical_blocks[use_query_aware] = min_budget
        num_slash_blocks[use_query_aware] = min_budget

        num_vertical_blocks[use_share] = min_budget
        num_slash_blocks[use_share] = min_budget

        vertical[..., :1] = torch.inf
        slash[..., -1:] = torch.inf

        # get slash topk
        num_slash_blocks = num_slash_blocks.view(batch_size * num_heads)
        slash = slash.view(batch_size * num_heads, -1)
        slash_topk = (num_blocks - 1) - slash.topk(
            min(num_slash_blocks.max().item(), num_blocks), -1
        ).indices
        slash_topk[
            torch.arange(slash_topk.shape[-1], device=num_slash_blocks.device)[None, :]
            >= num_slash_blocks[:, None]
        ] = 0
        slash_topk = slash_topk.view(batch_size, num_heads, -1)
        # get vertical topk
        num_vertical_blocks = num_vertical_blocks.view(batch_size * num_heads)
        vertical = vertical.view(batch_size * num_heads, -1)
        vertical_topk = vertical.topk(
            min(num_vertical_blocks.max().item(), num_blocks), -1
        ).indices
        vertical_topk[
            torch.arange(vertical_topk.shape[-1], device=num_vertical_blocks.device)[
                None, :
            ]
            >= num_vertical_blocks[:, None]
        ] = 0
        vertical_topk = vertical_topk.view(batch_size, num_heads, -1)

        vs_idx = combine_vertical_slash_indices(vertical_topk, slash_topk, num_blocks)

        for b in range(batch_size):
            for h in range(num_heads):
                block_idx[b][h] = torch.unique(
                    torch.cat((block_idx[b][h], vs_idx[b][h]), dim=-1)
                )

        block_causal_mask = None
        for b, h in (use_query_aware).nonzero():
            if block_causal_mask is None:
                block_causal_mask = torch.tril(
                    torch.ones(
                        num_blocks,
                        num_blocks,
                        device=q.device,
                        dtype=torch.bool,
                    )
                )
            pad_q = math.ceil(seq_len / block_size) * block_size - seq_len
            avg_q = (
                torch.nn.functional.pad(q[b, :, h, :], (0, 0, 0, pad_q), value=0)
                .view(num_blocks, block_size, head_dim)
                .mean(1)
            )
            avg_q[-1, :] = avg_q[-1, :] * block_size / (block_size - pad_q)
            attn = torch.einsum(
                "id, jd -> ij", avg_q / math.sqrt(head_dim), avg_k[b, :, h, :]
            ).masked_fill_(~block_causal_mask, float("-inf"))
            attn = torch.softmax(attn, dim=-1, dtype=torch.float32).view(-1)
            block_topk = score_cover_idx(attn, pattern_threshold * num_blocks)
            block_idx[b][h] = torch.unique(
                torch.cat((block_idx[b][h], block_topk), dim=-1)
            )

        return block_idx, use_dense

    @staticmethod
    def load_clusters(file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Sharing lookup file not found: {file_path}")

        with open(file_path, "r") as f:
            clusters_info = json.load(f)

        clusters = clusters_info["clusters"]  # dict
        shared_clusters = set(clusters_info["shared_clusters"])  # list -> set
        return clusters, shared_clusters

    def find_cluster_id(self, layer: int, head: int) -> Optional[str]:
        for cluster_id, pairs in self.clusters.items():
            if [layer, head] in pairs:
                return cluster_id
        return None
