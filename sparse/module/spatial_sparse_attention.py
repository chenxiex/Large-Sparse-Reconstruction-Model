# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from einops import rearrange
from flash_attn import flash_attn_varlen_func

from ..ops import get_block_score, sparse_window_attention, spatial_selection_attention
from .residual_block import SparseResBlock


class SpatialSparseAttentionIn(torch.nn.Module):
    def __init__(
        self,
        head_dim: int,
        num_kv_heads: int,
    ):
        super().__init__()
        # ssa parameters
        self.compression_key = SparseResBlock(num_kv_heads * head_dim)
        self.compression_value = SparseResBlock(num_kv_heads * head_dim)

    def sparse3d_compression(self, x, cu_block_include_tokens):
        block_include_tokens = (
            cu_block_include_tokens[1:] - cu_block_include_tokens[:-1]
        )
        compressed_x = torch.segment_reduce(
            data=x,
            reduce="mean",
            lengths=block_include_tokens,
            axis=0,
        )
        return compressed_x

    def forward(
        self,
        k,
        v,
        attn_map,
    ):
        # compression attention
        compressed_k = self.compression_key(k)
        compressed_v = self.compression_value(v)

        if "local_cu_block_include_tokens" in attn_map:
            compressed_k = self.sparse3d_compression(
                compressed_k,
                attn_map["local_cu_block_include_tokens"],
            )
            compressed_v = self.sparse3d_compression(
                compressed_v,
                attn_map["local_cu_block_include_tokens"],
            )
        else:
            compressed_k = self.sparse3d_compression(
                compressed_k,
                attn_map["cu_block_include_tokens"],
            )
            compressed_v = self.sparse3d_compression(
                compressed_v,
                attn_map["cu_block_include_tokens"],
            )
        return compressed_k, compressed_v


class SpatialSparseAttentionMid(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        is_window_attn=False,
    ):
        super().__init__()
        self.is_window_attn = is_window_attn
        self.hidden_dim = hidden_dim
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        # Note: norm_layer removed - normalization now happens in CrossAttention
        # BEFORE all_gather for better memory efficiency

    def forward(
        self,
        q,
        cu_seqlens_q,
        k,
        compressed_k,
        v,
        compressed_v,
        topk: int,
        attn_map,
        return_sparse_info=False,
    ):
        # Reshape only - normalization already done in CrossAttention before all_gather
        # This saves ~2GB memory by normalizing local tensors instead of full gathered sequence
        q = q.view(-1, self.num_q_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        compressed_k = compressed_k.view(-1, self.num_kv_heads, self.head_dim)
        compressed_v = compressed_v.view(-1, self.num_kv_heads, self.head_dim)

        compressed_cu_seqlens = attn_map["cu_seqblocks"]
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]

        seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]

        cu_seqblocks = attn_map["cu_seqblocks"]
        cu_block_include_tokens = attn_map["cu_block_include_tokens"]
        cu_seqlens_k = cu_block_include_tokens[cu_seqblocks]
        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]

        # Only request LSE when block_topk is not pre-computed
        # This saves memory by avoiding LSE tensor allocation when not needed
        need_lse = attn_map["block_topk"] is None
        if need_lse:
            compressed_attn_output, lse, _ = flash_attn_varlen_func(
                q,
                compressed_k,
                compressed_v,
                cu_seqlens_q,
                compressed_cu_seqlens,
                seqlens_q.max().item(),
                compressed_seqlens.max().item(),
                causal=False,
                return_attn_probs=True,
            )
        else:
            compressed_attn_output = flash_attn_varlen_func(
                q,
                compressed_k,
                compressed_v,
                cu_seqlens_q,
                compressed_cu_seqlens,
                seqlens_q.max().item(),
                compressed_seqlens.max().item(),
                causal=False,
                return_attn_probs=False,
            )
            lse = None

        sparse_info = {}
        if attn_map["block_topk"] is not None:
            block_topk = attn_map["block_topk"]
            cu_seqblocks = attn_map["cu_seqblocks"]
            cu_block_include_tokens = attn_map["cu_block_include_tokens"]
        else:
            with torch.no_grad():
                block_topk = get_block_score(
                    q,
                    compressed_k,
                    lse,
                    topk,
                    cu_seqlens_q,
                    compressed_cu_seqlens,
                    seqlens_q,
                    compressed_seqlens,
                    None,
                )
            cu_seqblocks = attn_map["cu_seqblocks"]
            cu_block_include_tokens = attn_map["cu_block_include_tokens"]
            if return_sparse_info:
                sparse_info["block_topk"] = block_topk
                sparse_info["cu_seqblocks"] = cu_seqblocks
                sparse_info["cu_block_include_tokens"] = cu_block_include_tokens

        # spatial selection attention
        selection_attn_output = spatial_selection_attention(
            q,
            k,
            v,
            block_topk,
            cu_seqblocks,
            cu_block_include_tokens,
            cu_seqlens_q,
            cu_seqlens_k,
            seqlens_q,
            seqlens_k,
            None,
        )

        selection_attn_output = rearrange(selection_attn_output, "n h d -> n (h d)")
        compressed_attn_output = rearrange(compressed_attn_output, "n h d -> n (h d)")

        attn_output = {
            "selection_attn": selection_attn_output,
            "compressed_attn": compressed_attn_output,
        }
        return attn_output, sparse_info


class SpatialSparseAttentionOut(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_q_heads: int,
        num_kv_heads: int,
        is_window_attn=False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.is_window_attn = is_window_attn

        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_dim // num_q_heads
        # Note: norm_layer removed - normalization now happens in CrossAttention
        # BEFORE all_gather for better memory efficiency

        # gate function
        if is_window_attn:
            self.gate = torch.nn.Sequential(
                nn.Linear(hidden_dim, 3 * hidden_dim, bias=False),
                nn.Sigmoid(),
            )
        else:
            self.gate = torch.nn.Sequential(
                nn.Linear(hidden_dim, 2 * hidden_dim, bias=False),
                nn.Sigmoid(),
            )

    def forward(
        self,
        x,
        q,
        k,
        v,
        attn_map,
        attn_output,
    ):
        # Reshape only - normalization already done in CrossAttention before all_gather
        # This saves ~2GB memory by normalizing local tensors instead of full gathered sequence
        q = q.view(-1, self.num_q_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        # window attention
        selection_attn_output = attn_output["selection_attn"]
        compressed_attn_output = attn_output["compressed_attn"]

        # window attention
        if self.is_window_attn:
            if "local_cu_block_include_tokens" in attn_map:
                window_attn_output = sparse_window_attention(
                    q, k, v, attn_map["local_cu_block_include_tokens"]
                )
            else:
                window_attn_output = sparse_window_attention(
                    q, k, v, attn_map["cu_block_include_tokens"]
                )
            window_attn_output = rearrange(window_attn_output, "n h d -> n (h d)")
            # gate average
            gate = self.gate(x)
            attn_output = (
                gate[:, 0 : self.hidden_dim] * compressed_attn_output
                + gate[:, self.hidden_dim : 2 * self.hidden_dim] * selection_attn_output
                + gate[:, 2 * self.hidden_dim :] * window_attn_output
            )
        else:
            # gate average
            gate = self.gate(x)
            attn_output = (
                gate[:, 0 : self.hidden_dim] * compressed_attn_output
                + gate[:, self.hidden_dim :] * selection_attn_output
            )
        return attn_output
