# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Block importance scoring for sparse attention.
Computes attention scores between queries and compressed keys,
then selects top-k blocks per query token."""

import math

import torch


def get_block_score(
    q: torch.Tensor,
    compressed_k: torch.Tensor,
    lse: torch.Tensor,
    topk: int,
    cu_seqlens_q: torch.Tensor,
    compressed_cu_seqlens: torch.Tensor,
    seqlens_q: torch.Tensor,
    compressed_seqlens: torch.Tensor,
    sm_scale: float = None,
) -> torch.Tensor:
    """Score compressed KV blocks and select top-k per query.

    Uses Q @ compressed_K^T attention scores, weighted by the existing
    softmax normalization (via LSE), to determine which KV blocks are
    most important for each query token.

    Args:
        q: queries [total_q, num_q_heads, d]
        compressed_k: compressed keys [total_compressed, num_kv_heads, d]
        lse: log-sum-exp from flash attention [num_q_heads, total_q]
        topk: number of top blocks to select
        cu_seqlens_q: cumulative query lengths [batch_size + 1]
        compressed_cu_seqlens: cumulative compressed key lengths [batch_size + 1]
        seqlens_q: per-sequence query lengths [batch_size]
        compressed_seqlens: per-sequence compressed lengths [batch_size]
        sm_scale: softmax scale (default: 1/sqrt(d))

    Returns:
        block_topk: top-k block indices [num_kv_heads, total_q, topk] int32
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.shape[-1])

    num_q_heads = q.shape[1]
    num_kv_heads = compressed_k.shape[1]
    num_share = num_q_heads // num_kv_heads
    batch_size = cu_seqlens_q.shape[0] - 1
    total_q = cu_seqlens_q[-1].item()

    # The original Triton kernel uses log2-base LSE with exp2 for speed.
    # Convert LSE from natural log (flash_attn output) to log2 base.
    lse_log2 = lse * 1.44269504  # = lse / ln(2)
    qk_scale = sm_scale * 1.44269504

    block_topk = -torch.ones(
        (num_kv_heads, total_q, topk), device=q.device, dtype=torch.int32
    )

    for b in range(batch_size):
        q_s, q_e = cu_seqlens_q[b].item(), cu_seqlens_q[b + 1].item()
        k_s, k_e = (
            compressed_cu_seqlens[b].item(),
            compressed_cu_seqlens[b + 1].item(),
        )
        n_compressed = k_e - k_s

        if n_compressed == 0:
            continue

        real_topk = min(topk, n_compressed)

        for kh in range(num_kv_heads):
            ck = compressed_k[k_s:k_e, kh].float()  # [n_compressed, d]
            score = torch.zeros(
                q_e - q_s,
                n_compressed,
                device=q.device,
                dtype=torch.float32,
            )

            for sh in range(num_share):
                h = kh * num_share + sh
                q_h = q[q_s:q_e, h].float()  # [q_len, d]
                lse_h = lse_log2[h, q_s:q_e]  # [q_len]
                qk = (q_h @ ck.T) * qk_scale
                score += torch.exp2(qk - lse_h.unsqueeze(1))

            _, topk_idx = score.topk(real_topk, dim=-1)
            topk_idx = topk_idx.sort(dim=-1).values
            block_topk[kh, q_s:q_e, :real_topk] = topk_idx.to(torch.int32)

    return block_topk
