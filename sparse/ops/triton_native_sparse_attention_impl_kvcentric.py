# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

# Clean-room rewrite of variable block size native sparse attention.
# Optimized with tl.make_block_ptr and tl.advance for coalesced memory access.
# Uses exp2/log2 for faster math (single PTX instruction vs multi-instruction exp/log).
# Eliminates tile mapping memory loads via inline computation.
# Follows the code patterns and conventions of the open source
# native_sparse_attention_pytorch by lucidrains.

import math
from typing import Optional

import torch
import triton
import triton.language as tl
from torch import Tensor


def exists(v):
    return v is not None


def divisible_by(num, den):
    return (num % den) == 0


def round_up_multiple(n, mult):
    return math.ceil(n / mult) * mult


def is_contiguous(x: Tensor):
    return x.stride(-1) == 1


# ============================================================================
# Forward kernel
#
# Adapted from lucidrains' forward_kernel_causal_and_sparse for variable
# block sizes. Each query token iterates over its topk selected blocks,
# looks up the variable token range via cu_block_include_tokens, and
# processes each block in fixed-size tiles using online softmax with
# exp2/log2 based online softmax (single PTX instruction, faster than exp/log).
#
# Optimized: K is loaded transposed (K^T) via explicit pointer arithmetic
# to avoid tl.trans() in QK computation. Q and output use block pointers.
# ============================================================================


@triton.jit
def forward_kernel_varblock(
    Q,
    K,
    V,
    kv_block_indices,
    Out,
    Lse,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqblocks,
    cu_block_include_tokens,
    softmax_scale,
    stride_qm,
    stride_qh,
    stride_q_headdim,
    stride_kn,
    stride_kh,
    stride_k_headdim,
    stride_vn,
    stride_vh,
    stride_v_headdim,
    stride_om,
    stride_oh,
    stride_od,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_kvbl_k,
    stride_lse_h,
    stride_lse_m,
    kv_heads,
    query_head_groups,
    seqlen_total,
    headdim,
    num_sel_blocks,
    queries_per_step,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    BLOCK_SEL: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_q = tl.program_id(2)

    off_qh = pid_kv_head * query_head_groups

    # load sequence boundaries for this batch element
    q_start = tl.load(cu_seqlens_q + pid_batch)
    seqlen_q = tl.load(cu_seqlens_q + pid_batch + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_batch)
    seqlen_k = tl.load(cu_seqlens_k + pid_batch + 1) - k_start
    block_start = tl.load(cu_seqblocks + pid_batch)
    num_blocks = tl.load(cu_seqblocks + pid_batch + 1) - block_start

    if pid_q * queries_per_step >= seqlen_q:
        return

    actual_q = min(queries_per_step, seqlen_q - pid_q * queries_per_step)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_n = tl.arange(0, BLOCK)
    offs_h = tl.arange(0, QUERY_HEAD_GROUPS)
    offs_sel = tl.arange(0, BLOCK_SEL)

    # precompute K/V base pointers for this batch/head
    k_base = K + k_start * stride_kn + pid_kv_head * stride_kh
    v_base = V + k_start * stride_vn + pid_kv_head * stride_vh

    # exp2/log2 scale: log2(e) * softmax_scale
    qk_scale = softmax_scale * 1.44269504

    for j in range(actual_q):
        qi = pid_q * queries_per_step + j
        qi_global = q_start + qi

        # load query for all grouped heads: [QUERY_HEAD_GROUPS, BLOCK_HEADDIM]
        q_block_ptr = tl.make_block_ptr(
            base=Q + qi_global * stride_qm + off_qh * stride_qh,
            shape=(query_head_groups, headdim),
            strides=(stride_qh, stride_q_headdim),
            offsets=(0, 0),
            block_shape=(QUERY_HEAD_GROUPS, BLOCK_HEADDIM),
            order=(1, 0),
        )
        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        # load topk block indices for this query token
        topk_vals = tl.load(
            kv_block_indices
            + pid_kv_head * stride_kvbl_h
            + qi_global * stride_kvbl_m
            + offs_sel * stride_kvbl_k,
            mask=offs_sel < num_sel_blocks,
            other=-1,
        )
        valid_sel_count = tl.sum(
            tl.where((topk_vals >= 0) & (topk_vals < num_blocks), 1, 0),
            axis=0,
        )

        # online softmax accumulators
        m_i = tl.full((QUERY_HEAD_GROUPS,), float("-inf"), dtype=tl.float32)
        lse_i = tl.full((QUERY_HEAD_GROUPS,), float("-inf"), dtype=tl.float32)
        acc_o = tl.zeros((QUERY_HEAD_GROUPS, BLOCK_HEADDIM), dtype=tl.float32)

        # iterate over selected blocks
        sel_ptr = (
            kv_block_indices + pid_kv_head * stride_kvbl_h + qi_global * stride_kvbl_m
        )

        for _sel in range(valid_sel_count):
            block_index = tl.load(sel_ptr).to(tl.int32)
            token_begin = tl.load(
                cu_block_include_tokens + block_start + block_index
            ).to(tl.int32)
            block_len = (
                tl.load(cu_block_include_tokens + block_start + block_index + 1).to(
                    tl.int32
                )
                - token_begin
            )
            c = token_begin - k_start
            sel_ptr = sel_ptr + stride_kvbl_k

            # process this block in fixed-size tiles
            for tile in range(0, block_len, BLOCK):
                tile_mask = (c + tile + offs_n < seqlen_k) & (tile + offs_n < block_len)

                # load k^T tile: [BLOCK_HEADDIM, BLOCK] — transposed indexing
                kt_tile = tl.load(
                    k_base
                    + offs_d[:, None] * stride_k_headdim
                    + (c + tile + offs_n[None, :]) * stride_kn,
                    mask=(tile_mask[None, :] & (offs_d[:, None] < headdim)),
                    other=0.0,
                )

                # compute qk: [QUERY_HEAD_GROUPS, BLOCK] — no tl.trans needed
                qk = tl.dot(q, kt_tile)

                # mask out-of-bounds positions
                qk += tl.where(
                    tile_mask[None, :],
                    0.0,
                    float("-inf"),
                )

                # online softmax update (exp2/log2 based, faster single PTX instruction)
                m_ij = tl.maximum(m_i, tl.max(qk * qk_scale, axis=1))
                p = tl.exp2(qk * qk_scale - m_ij[:, None])
                l_ij = tl.sum(p, axis=1)

                # rescale running accumulator
                acc_o = acc_o * tl.exp2(m_i - m_ij)[:, None]

                # load v tile: [BLOCK, BLOCK_HEADDIM]
                v_tile = tl.load(
                    v_base
                    + (c + tile + offs_n[:, None]) * stride_vn
                    + offs_d[None, :] * stride_v_headdim,
                    mask=(tile_mask[:, None] & (offs_d[None, :] < headdim)),
                    other=0.0,
                )

                p = p.to(v_tile.dtype)
                acc_o += tl.dot(p, v_tile)

                # update statistics
                m_i = m_ij
                l_i_new = tl.exp2(lse_i - m_ij) + l_ij
                lse_i = m_ij + tl.math.log2(l_i_new)

        # final normalization
        acc_o = acc_o * tl.exp2(m_i - lse_i)[:, None]

        # store output: [QUERY_HEAD_GROUPS, BLOCK_HEADDIM]
        o_block_ptr = tl.make_block_ptr(
            base=Out + qi_global * stride_om + off_qh * stride_oh,
            shape=(query_head_groups, headdim),
            strides=(stride_oh, stride_od),
            offsets=(0, 0),
            block_shape=(QUERY_HEAD_GROUPS, BLOCK_HEADDIM),
            order=(1, 0),
        )
        tl.store(o_block_ptr, acc_o.to(Out.dtype.element_ty), boundary_check=(0, 1))

        # store lse: [QUERY_HEAD_GROUPS]
        tl.store(
            Lse + (off_qh + offs_h) * stride_lse_h + qi_global * stride_lse_m,
            lse_i,
            mask=offs_h < query_head_groups,
        )


# ============================================================================
# Backward dQ kernel (with fused delta computation)
#
# Same query-centric iteration as forward: for each query token, iterate
# over its topk selected blocks, recompute attention weights, and
# accumulate dQ = ds @ K. Uses exp2/log2 softmax for faster math.
#
# Optimized: K^T and V^T loaded via transposed explicit pointers to
# avoid tl.trans() in QK and dp computations. Q/dO/dQ use block pointers.
# ============================================================================


@triton.jit
def backward_kernel_dq(
    Q,
    K,
    V,
    Out,
    kv_block_indices,
    Lse,
    Delta,
    DO,
    DQ,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqblocks,
    cu_block_include_tokens,
    softmax_scale,
    stride_qm,
    stride_qh,
    stride_q_headdim,
    stride_kn,
    stride_kh,
    stride_k_headdim,
    stride_vn,
    stride_vh,
    stride_v_headdim,
    stride_om,
    stride_oh,
    stride_od,
    stride_kvbl_h,
    stride_kvbl_m,
    stride_kvbl_k,
    stride_lse_h,
    stride_lse_m,
    stride_delta_h,
    stride_delta_m,
    stride_dom,
    stride_doh,
    stride_do_headdim,
    stride_dqm,
    stride_dqh,
    stride_dq_headdim,
    kv_heads,
    query_head_groups,
    headdim,
    num_sel_blocks,
    queries_per_step,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    BLOCK_SEL: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_q = tl.program_id(2)

    off_qh = pid_kv_head * query_head_groups

    q_start = tl.load(cu_seqlens_q + pid_batch)
    seqlen_q = tl.load(cu_seqlens_q + pid_batch + 1) - q_start
    k_start = tl.load(cu_seqlens_k + pid_batch)
    block_start = tl.load(cu_seqblocks + pid_batch)
    num_blocks = tl.load(cu_seqblocks + pid_batch + 1) - block_start

    if pid_q * queries_per_step >= seqlen_q:
        return

    actual_q = min(queries_per_step, seqlen_q - pid_q * queries_per_step)

    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_n = tl.arange(0, BLOCK)
    offs_h = tl.arange(0, QUERY_HEAD_GROUPS)
    offs_sel = tl.arange(0, BLOCK_SEL)

    # precompute K/V base pointers
    k_base = K + k_start * stride_kn + pid_kv_head * stride_kh
    v_base = V + k_start * stride_vn + pid_kv_head * stride_vh

    # exp2/log2 scale: log2(e) * softmax_scale
    qk_scale = softmax_scale * 1.44269504

    for j in range(actual_q):
        qi = pid_q * queries_per_step + j
        qi_global = q_start + qi

        # load topk
        topk_vals = tl.load(
            kv_block_indices
            + pid_kv_head * stride_kvbl_h
            + qi_global * stride_kvbl_m
            + offs_sel * stride_kvbl_k,
            mask=offs_sel < num_sel_blocks,
            other=-1,
        )
        valid_sel_count = tl.sum(
            tl.where((topk_vals >= 0) & (topk_vals < num_blocks), 1, 0),
            axis=0,
        )

        # load q and do using block pointers
        q_block_ptr = tl.make_block_ptr(
            base=Q + qi_global * stride_qm + off_qh * stride_qh,
            shape=(query_head_groups, headdim),
            strides=(stride_qh, stride_q_headdim),
            offsets=(0, 0),
            block_shape=(QUERY_HEAD_GROUPS, BLOCK_HEADDIM),
            order=(1, 0),
        )
        q = tl.load(q_block_ptr, boundary_check=(0, 1), padding_option="zero")

        do_block_ptr = tl.make_block_ptr(
            base=DO + qi_global * stride_dom + off_qh * stride_doh,
            shape=(query_head_groups, headdim),
            strides=(stride_doh, stride_do_headdim),
            offsets=(0, 0),
            block_shape=(QUERY_HEAD_GROUPS, BLOCK_HEADDIM),
            order=(1, 0),
        )
        do = tl.load(do_block_ptr, boundary_check=(0, 1), padding_option="zero")

        lse_val = tl.load(
            Lse + (off_qh + offs_h) * stride_lse_h + qi_global * stride_lse_m,
            mask=offs_h < query_head_groups,
            other=0.0,
        )

        # Compute delta inline: delta = sum_d(O * dO) — fused, no separate kernel
        o_block_ptr = tl.make_block_ptr(
            base=Out + qi_global * stride_om + off_qh * stride_oh,
            shape=(query_head_groups, headdim),
            strides=(stride_oh, stride_od),
            offsets=(0, 0),
            block_shape=(QUERY_HEAD_GROUPS, BLOCK_HEADDIM),
            order=(1, 0),
        )
        o_val = tl.load(o_block_ptr, boundary_check=(0, 1), padding_option="zero")
        delta_val = tl.sum(
            o_val.to(tl.float32) * do.to(tl.float32), axis=1
        )  # [QUERY_HEAD_GROUPS]

        # Store delta for dK/dV kernel to read later
        tl.store(
            Delta + (off_qh + offs_h) * stride_delta_h + qi_global * stride_delta_m,
            delta_val,
            mask=offs_h < query_head_groups,
        )

        dq = tl.zeros((QUERY_HEAD_GROUPS, BLOCK_HEADDIM), dtype=tl.float32)

        sel_ptr = (
            kv_block_indices + pid_kv_head * stride_kvbl_h + qi_global * stride_kvbl_m
        )

        for _sel in range(valid_sel_count):
            block_index = tl.load(sel_ptr).to(tl.int32)
            token_begin = tl.load(
                cu_block_include_tokens + block_start + block_index
            ).to(tl.int32)
            block_len = (
                tl.load(cu_block_include_tokens + block_start + block_index + 1).to(
                    tl.int32
                )
                - token_begin
            )
            c = token_begin - k_start
            sel_ptr = sel_ptr + stride_kvbl_k

            for tile in range(0, block_len, BLOCK):
                tile_mask = tile + offs_n < block_len

                # load k^T tile: [BLOCK_HEADDIM, BLOCK] — transposed
                kt_tile = tl.load(
                    k_base
                    + offs_d[:, None] * stride_k_headdim
                    + (c + tile + offs_n[None, :]) * stride_kn,
                    mask=(tile_mask[None, :] & (offs_d[:, None] < headdim)),
                    other=0.0,
                )

                # load v^T tile: [BLOCK_HEADDIM, BLOCK] — transposed
                vt_tile = tl.load(
                    v_base
                    + offs_d[:, None] * stride_v_headdim
                    + (c + tile + offs_n[None, :]) * stride_vn,
                    mask=(tile_mask[None, :] & (offs_d[:, None] < headdim)),
                    other=0.0,
                )

                # recompute attention weights — no tl.trans needed
                qk = tl.dot(q, kt_tile)
                qk += tl.where(
                    tile_mask[None, :],
                    0.0,
                    float("-inf"),
                )

                p = tl.exp2(qk * qk_scale - lse_val[:, None])

                # dp = do @ V^T — no tl.trans needed
                dp = tl.dot(do, vt_tile)

                # ds = softmax_scale * P * (dp - delta)
                ds = softmax_scale * p * (dp - delta_val[:, None])
                ds = ds.to(q.dtype)

                # dQ += ds @ K (transpose K^T back to K)
                dq += tl.dot(ds, tl.trans(kt_tile))

        # store dQ using block pointer
        dq_block_ptr = tl.make_block_ptr(
            base=DQ + qi_global * stride_dqm + off_qh * stride_dqh,
            shape=(query_head_groups, headdim),
            strides=(stride_dqh, stride_dq_headdim),
            offsets=(0, 0),
            block_shape=(QUERY_HEAD_GROUPS, BLOCK_HEADDIM),
            order=(1, 0),
        )
        tl.store(dq_block_ptr, dq.to(DQ.dtype.element_ty), boundary_check=(0, 1))


# ============================================================================
# Backward dK/dV kernel (fast, KV-centric)
#
# Instead of query-centric iteration with atomic_add, this kernel uses
# KV-centric iteration: each program owns one KV block tile and iterates
# over the queries that selected it (via reverse query mapping).
# dK/dV are accumulated in float32 registers and written once (no atomics).
#
# Optimized: uses tl.make_block_ptr for K^T and V^T loads (avoids tl.trans),
# block pointers for Q/dO loads and dK/dV stores.
# ============================================================================


@triton.jit
def _count_queries_kernel(
    topk_idx,  # [kv_heads, total_q, topk]
    counts,  # [kv_heads, total_blocks] output
    block_starts,  # [total_q] int32 — per-query global block offset
    n_blocks_per_q,  # [total_q] int32 — per-query block count
    total_q,
    topk,
    stride_th,
    stride_tn,
    stride_tk,
    stride_ch,
    stride_cn,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Count how many queries select each KV block (replaces bincount)."""
    pid_h = tl.program_id(0)  # kv head
    pid_n = tl.program_id(1)  # query chunk

    for ni in range(BLOCK_N):
        qi = pid_n * BLOCK_N + ni
        if qi < total_q:
            b_start = tl.load(block_starts + qi).to(tl.int32)
            n_blk = tl.load(n_blocks_per_q + qi).to(tl.int32)

            for ti in range(TOPK):
                if ti < topk:
                    blk = tl.load(
                        topk_idx + pid_h * stride_th + qi * stride_tn + ti * stride_tk
                    ).to(tl.int32)
                    if blk >= 0 and blk < n_blk:
                        global_blk = b_start + blk
                        tl.atomic_add(
                            counts + pid_h * stride_ch + global_blk * stride_cn, 1
                        )


@triton.jit
def _scatter_queries_kernel(
    topk_idx,  # [kv_heads, total_q, topk]
    result,  # [kv_heads, max_total_entries] output
    write_offsets,  # [kv_heads, total_blocks] — atomic write positions
    block_starts,  # [total_q] int32
    n_blocks_per_q,  # [total_q] int32
    total_q,
    topk,
    stride_th,
    stride_tn,
    stride_tk,
    stride_rh,
    stride_rm,
    stride_wh,
    stride_wn,
    BLOCK_N: tl.constexpr,
    TOPK: tl.constexpr,
):
    """Scatter query indices into sorted positions (replaces argsort)."""
    pid_h = tl.program_id(0)  # kv head
    pid_n = tl.program_id(1)  # query chunk

    for ni in range(BLOCK_N):
        qi = pid_n * BLOCK_N + ni
        if qi < total_q:
            b_start = tl.load(block_starts + qi).to(tl.int32)
            n_blk = tl.load(n_blocks_per_q + qi).to(tl.int32)

            for ti in range(TOPK):
                if ti < topk:
                    blk = tl.load(
                        topk_idx + pid_h * stride_th + qi * stride_tn + ti * stride_tk
                    ).to(tl.int32)
                    if blk >= 0 and blk < n_blk:
                        global_blk = b_start + blk
                        pos = tl.atomic_add(
                            write_offsets + pid_h * stride_wh + global_blk * stride_wn,
                            1,
                        )
                        tl.store(result + pid_h * stride_rh + pos * stride_rm, qi)


def _build_reverse_query_mapping(topk_idx, cu_seqlens_q, cu_seqblocks, total_blocks):
    """
    Build reverse mapping from KV blocks to query tokens using Triton kernels.

    Replaces bincount + argsort with atomic counting + scatter Triton kernels
    for faster preprocessing on GPU.

    Args:
        topk_idx: [kv_heads, total_q, topk] int32 — selected block indices
        cu_seqlens_q: [batch+1] int32
        cu_seqblocks: [batch+1] int32
        total_blocks: total number of blocks across all batches

    Returns:
        cu_reverse_count: [kv_heads, total_blocks + 1] int32 — cumulative count
        reverse_query_indices: [kv_heads, max_total_entries] int32 — query indices
    """
    kv_heads, total_q, topk = topk_idx.shape
    device = topk_idx.device

    # Build per-query metadata vectorized
    seq_lens = (cu_seqlens_q[1:] - cu_seqlens_q[:-1]).long()
    b_starts = cu_seqblocks[:-1]
    b_counts = cu_seqblocks[1:] - cu_seqblocks[:-1]
    block_starts = torch.repeat_interleave(b_starts, seq_lens).to(torch.int32)
    n_blocks_per_q = torch.repeat_interleave(b_counts, seq_lens).to(torch.int32)

    # Step 1: Count queries per block using Triton kernel
    counts = torch.zeros(kv_heads, total_blocks, dtype=torch.int32, device=device)
    BLOCK_N = 64
    grid_count = (kv_heads, triton.cdiv(total_q, BLOCK_N))

    TOPK_CONST = triton.next_power_of_2(topk)

    _count_queries_kernel[grid_count](
        topk_idx,
        counts,
        block_starts,
        n_blocks_per_q,
        total_q,
        topk,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        counts.stride(0),
        counts.stride(1),
        BLOCK_N=BLOCK_N,
        TOPK=TOPK_CONST,
        num_warps=4,
    )

    # Step 2: Build cumulative counts (CSR offsets)
    cu_reverse_count = torch.zeros(
        kv_heads, total_blocks + 1, dtype=torch.int32, device=device
    )
    cu_reverse_count[:, 1:] = counts.cumsum(dim=1).to(torch.int32)

    # Step 3: Scatter query indices using Triton kernel
    max_total_entries = counts.sum(dim=1).max().item()
    if max_total_entries == 0:
        max_total_entries = 1

    reverse_query_indices = torch.zeros(
        kv_heads, max_total_entries, dtype=torch.int32, device=device
    )

    # write_offsets starts as copy of cu_reverse_count[:, :-1], atomically incremented
    write_offsets = cu_reverse_count[:, :-1].clone().contiguous()

    grid_scatter = (kv_heads, triton.cdiv(total_q, BLOCK_N))
    _scatter_queries_kernel[grid_scatter](
        topk_idx,
        reverse_query_indices,
        write_offsets,
        block_starts,
        n_blocks_per_q,
        total_q,
        topk,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        reverse_query_indices.stride(0),
        reverse_query_indices.stride(1),
        write_offsets.stride(0),
        write_offsets.stride(1),
        BLOCK_N=BLOCK_N,
        TOPK=TOPK_CONST,
        num_warps=4,
    )

    # Opt 5: Sort query indices within each block for coalesced Q/dO loads.
    # After atomic scatter, indices within a block segment are unordered.
    # Sorting them ascending makes the gathered Q/dO loads more sequential,
    # improving L2 cache hit rate at larger scales.
    # Uses vectorized segmented sort (1 .item() per head, not per block).
    for h in range(kv_heads):
        total_entries = cu_reverse_count[h, total_blocks].item()
        if total_entries == 0:
            continue
        entries = reverse_query_indices[h, :total_entries]
        # Build segment IDs via searchsorted (vectorized on GPU)
        boundaries = cu_reverse_count[h, 1 : total_blocks + 1].to(torch.int64)
        positions = torch.arange(total_entries, device=device, dtype=torch.int64)
        seg_ids = torch.searchsorted(boundaries, positions, right=True)
        # Composite key for stable sort: seg_id * (total_q+1) + entry_value
        keys = seg_ids * (total_q + 1) + entries.to(torch.int64)
        _, sorted_idx = keys.sort()
        reverse_query_indices[h, :total_entries] = entries[sorted_idx]

    return cu_reverse_count, reverse_query_indices


@triton.jit
def backward_kernel_dkdv_fastest(
    Q,
    K,
    V,
    Lse,
    Delta,
    DO,
    DK,
    DV,
    cu_seqlens_k,
    cu_block_include_tokens,
    cu_reverse_count,
    reverse_query_indices,
    cu_seqblocks,
    softmax_scale,
    stride_qm,
    stride_qh,
    stride_q_headdim,
    stride_kn,
    stride_kh,
    stride_k_headdim,
    stride_vn,
    stride_vh,
    stride_v_headdim,
    stride_lse_h,
    stride_lse_m,
    stride_delta_h,
    stride_delta_m,
    stride_dom,
    stride_doh,
    stride_do_headdim,
    stride_dkn,
    stride_dkh,
    stride_dk_headdim,
    stride_dvn,
    stride_dvh,
    stride_dv_headdim,
    stride_curev_h,
    stride_curev_b,
    stride_revidx_h,
    stride_revidx_m,
    kv_heads,
    query_head_groups,
    headdim,
    max_seqblocks,
    BLOCK_HEADDIM: tl.constexpr,
    BLOCK: tl.constexpr,
    BLOCK_Q: tl.constexpr,
    BLOCK_M: tl.constexpr,
    QUERY_HEAD_GROUPS: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_kv_head = tl.program_id(1)
    pid_combined = tl.program_id(2)

    # Decode split-K from combined index
    pid_split = pid_combined % SPLIT_K
    pid_kb = pid_combined // SPLIT_K

    # Inline tile mapping: compute block and tile offset from pid_kb
    pid_k = pid_kb % max_seqblocks  # which block within this batch
    pid_tile_in_block = pid_kb // max_seqblocks  # which tile within the block

    b_start = tl.load(cu_seqblocks + pid_batch).to(tl.int32)
    b_len = tl.load(cu_seqblocks + pid_batch + 1).to(tl.int32) - b_start

    if pid_k >= b_len:
        return

    global_block = b_start + pid_k
    token_begin = tl.load(cu_block_include_tokens + global_block).to(tl.int32)
    block_len = (
        tl.load(cu_block_include_tokens + global_block + 1).to(tl.int32) - token_begin
    )
    tile_offset = pid_tile_in_block * BLOCK

    if tile_offset >= block_len:
        return

    offs_n = tl.arange(0, BLOCK)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    offs_m = tl.arange(0, BLOCK_M)
    offs_qm = offs_m // QUERY_HEAD_GROUPS
    offs_hm = offs_m % QUERY_HEAD_GROUPS
    base_qh = pid_kv_head * query_head_groups

    # Load K^T and V^T tiles using block pointers
    kt_block_ptr = tl.make_block_ptr(
        base=K + token_begin * stride_kn + pid_kv_head * stride_kh,
        shape=(headdim, block_len),
        strides=(stride_k_headdim, stride_kn),
        offsets=(0, tile_offset),
        block_shape=(BLOCK_HEADDIM, BLOCK),
        order=(0, 1),
    )
    kt_tile = tl.load(kt_block_ptr, boundary_check=(0, 1), padding_option="zero")

    vt_block_ptr = tl.make_block_ptr(
        base=V + token_begin * stride_vn + pid_kv_head * stride_vh,
        shape=(headdim, block_len),
        strides=(stride_v_headdim, stride_vn),
        offsets=(0, tile_offset),
        block_shape=(BLOCK_HEADDIM, BLOCK),
        order=(0, 1),
    )
    vt_tile = tl.load(vt_block_ptr, boundary_check=(0, 1), padding_option="zero")

    # Initialize accumulators in float32
    dk = tl.zeros((BLOCK, BLOCK_HEADDIM), dtype=tl.float32)
    dv = tl.zeros((BLOCK, BLOCK_HEADDIM), dtype=tl.float32)

    # Look up reverse mapping: which queries selected this block
    rev_start = tl.load(
        cu_reverse_count + pid_kv_head * stride_curev_h + global_block * stride_curev_b,
    ).to(tl.int32)
    rev_end = tl.load(
        cu_reverse_count
        + pid_kv_head * stride_curev_h
        + (global_block + 1) * stride_curev_b,
    ).to(tl.int32)
    total_queries = rev_end - rev_start

    # Split-K: divide queries among splits
    queries_per_split = (total_queries + SPLIT_K - 1) // SPLIT_K
    my_q_start = pid_split * queries_per_split
    my_q_end = tl.minimum(my_q_start + queries_per_split, total_queries)

    if my_q_start >= total_queries:
        return

    num_queries = my_q_end - my_q_start

    # exp2/log2 scale: log2(e) * softmax_scale
    qk_scale = softmax_scale * 1.44269504

    # Precompute KV tile bounds mask
    tile_valid = (tile_offset + offs_n) < block_len

    # Batch BLOCK_Q queries per inner-loop iteration
    for qi_start in range(0, num_queries, BLOCK_Q):
        remaining = num_queries - qi_start

        # reverse_query_indices is padded by BLOCK_Q, so no OOB risk.
        qi_indices_m = tl.load(
            reverse_query_indices
            + pid_kv_head * stride_revidx_h
            + (rev_start + my_q_start + qi_start + offs_qm) * stride_revidx_m,
            mask=offs_qm < remaining,
            other=0,
        ).to(tl.int32)
        qh_indices_m = base_qh + offs_hm
        valid_m = (offs_qm < remaining) & (offs_hm < query_head_groups)

        # Load Q for all query heads sharing this KV head:
        # [BLOCK_Q * QUERY_HEAD_GROUPS, BLOCK_HEADDIM]
        q_chunk = tl.load(
            Q
            + qi_indices_m[:, None] * stride_qm
            + qh_indices_m[:, None] * stride_qh
            + offs_d[None, :] * stride_q_headdim,
            mask=valid_m[:, None] & (offs_d[None, :] < headdim),
            other=0.0,
        )

        # Load dO for all query heads sharing this KV head:
        # [BLOCK_Q * QUERY_HEAD_GROUPS, BLOCK_HEADDIM]
        do_chunk = tl.load(
            DO
            + qi_indices_m[:, None] * stride_dom
            + qh_indices_m[:, None] * stride_doh
            + offs_d[None, :] * stride_do_headdim,
            mask=valid_m[:, None] & (offs_d[None, :] < headdim),
            other=0.0,
        )

        # Load LSE for the flattened query/head rows: [BLOCK_Q * QUERY_HEAD_GROUPS]
        lse_chunk = tl.load(
            Lse + qh_indices_m * stride_lse_h + qi_indices_m * stride_lse_m,
            mask=valid_m,
            other=0.0,
        )

        # Load Delta for the flattened query/head rows: [BLOCK_Q * QUERY_HEAD_GROUPS]
        delta_chunk = tl.load(
            Delta + qh_indices_m * stride_delta_h + qi_indices_m * stride_delta_m,
            mask=valid_m,
            other=0.0,
        )

        # QK = Q @ K^T: [BLOCK_Q * QUERY_HEAD_GROUPS, BLOCK]
        qk = tl.dot(q_chunk, kt_tile)

        # Mask out-of-bounds KV positions
        qk += tl.where(tile_valid[None, :], 0.0, float("-inf"))
        # Mask invalid flattened query/head rows
        qk += tl.where(valid_m[:, None], 0.0, float("-inf"))

        # Attention weights: p = exp2(qk * qk_scale - lse)
        p = tl.exp2(qk * qk_scale - lse_chunk[:, None])

        # dp = dO @ V^T: [BLOCK_Q * QUERY_HEAD_GROUPS, BLOCK]
        dp = tl.dot(do_chunk, vt_tile)

        # ds = softmax_scale * p * (dp - delta): [BLOCK_Q * QUERY_HEAD_GROUPS, BLOCK]
        ds = softmax_scale * p * (dp - delta_chunk[:, None])

        # dk += ds^T @ Q: [BLOCK, BLOCK_HEADDIM]
        dk += tl.dot(tl.trans(ds.to(q_chunk.dtype)), q_chunk)

        # dv += p^T @ dO: [BLOCK, BLOCK_HEADDIM]
        dv += tl.dot(tl.trans(p.to(do_chunk.dtype)), do_chunk)

    # Compute tile bounds mask for store
    store_mask = (tile_offset + offs_n) < block_len

    dk_base = DK + token_begin * stride_dkn + pid_kv_head * stride_dkh
    dv_base = DV + token_begin * stride_dvn + pid_kv_head * stride_dvh

    # This program owns one (batch, kv_head, kv_tile) output, so the grouped
    # query-head reduction is complete locally and a plain store is sufficient.
    tl.store(
        dk_base
        + (tile_offset + offs_n[:, None]) * stride_dkn
        + offs_d[None, :] * stride_dk_headdim,
        dk.to(DK.dtype.element_ty),
        mask=store_mask[:, None] & (offs_d[None, :] < headdim),
    )

    tl.store(
        dv_base
        + (tile_offset + offs_n[:, None]) * stride_dvn
        + offs_d[None, :] * stride_dv_headdim,
        dv.to(DV.dtype.element_ty),
        mask=store_mask[:, None] & (offs_d[None, :] < headdim),
    )


# ============================================================================
# Forward wrapper
# ============================================================================


def native_sparse_attn_forward(
    q,
    k,
    v,
    topk_idx,
    cu_seqblocks,
    cu_block_include_tokens,
    cu_seqlens_q,
    cu_seqlens_k,
    softmax_scale,
):
    total_q, num_q_heads, headdim = q.shape
    total_k, kv_heads, _ = k.shape
    topk = topk_idx.shape[-1]
    batch_size = cu_seqlens_q.shape[0] - 1
    query_head_groups = num_q_heads // kv_heads

    o = torch.zeros_like(q)
    lse = torch.zeros(num_q_heads, total_q, dtype=torch.float32, device=q.device)

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_q = seqlens_q.max().item()
    queries_per_step = total_q // 32768 + 1

    BLOCK = 64
    BLOCK_HEADDIM = max(16, triton.next_power_of_2(headdim))
    BLOCK_SEL = triton.next_power_of_2(topk)
    QUERY_HEAD_GROUPS = triton.next_power_of_2(query_head_groups)
    num_warps = 4 if headdim <= 64 else 8

    grid = (
        batch_size,
        kv_heads,
        triton.cdiv(max_seqlen_q, queries_per_step),
    )

    forward_kernel_varblock[grid](
        q,
        k,
        v,
        topk_idx,
        o,
        lse,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqblocks,
        cu_block_include_tokens,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        lse.stride(0),
        lse.stride(1),
        kv_heads,
        query_head_groups,
        total_q,
        headdim,
        topk,
        queries_per_step,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK=BLOCK,
        QUERY_HEAD_GROUPS=QUERY_HEAD_GROUPS,
        BLOCK_SEL=BLOCK_SEL,
        num_warps=num_warps,
        num_stages=3,
    )

    return o, lse


# ============================================================================
# Backward wrapper
# ============================================================================


def native_sparse_attn_backward(
    do,
    q,
    k,
    v,
    o,
    lse,
    topk_idx,
    cu_seqblocks,
    cu_block_include_tokens,
    cu_seqlens_q,
    cu_seqlens_k,
    softmax_scale,
    variant="c",
):
    """
    Backward pass with selectable optimization variants.

    Variants control dK/dV kernel parameters:
      "a" — K/V SRAM reuse verified + BLOCK_Q=64, num_warps=4
      "b" — bf16 output verified + BLOCK_Q=128, num_warps=4, num_stages=2
      "c" — both optimizations + BLOCK_Q=64, num_warps=4, num_stages=2

    NOTE: K/V SRAM reuse (Opt 1) and bf16 dK/dV output (Opt 2) are already
    present in the baseline ultrafastest kernel:
      - K^T/V^T loaded once via tl.make_block_ptr, reused across query loop
      - dk/dv allocated as q.dtype (bf16), kernel accumulates float32 then
        casts on store via DK.dtype.element_ty
    The variants above explore additional tuning axes (BLOCK_Q, num_warps,
    num_stages) to find further speedups.
    """
    total_q, num_q_heads, headdim = q.shape
    total_k, kv_heads, _ = k.shape
    topk = topk_idx.shape[-1]
    batch_size = cu_seqlens_q.shape[0] - 1
    query_head_groups = num_q_heads // kv_heads
    device = q.device

    BLOCK_HEADDIM = max(16, triton.next_power_of_2(headdim))

    # 1. Compute dQ (delta computed inline inside dQ kernel, stored for dK/dV)
    delta = torch.zeros(num_q_heads, total_q, device=device, dtype=torch.float32)
    dq = torch.zeros_like(q)
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    max_seqlen_q = seqlens_q.max().item()
    queries_per_step = total_q // 32768 + 1

    BLOCK = 64
    BLOCK_SEL = triton.next_power_of_2(topk)
    QUERY_HEAD_GROUPS_CONST = max(16, triton.next_power_of_2(query_head_groups))
    num_warps = 4 if headdim <= 64 else 8

    grid_dq = (
        batch_size,
        kv_heads,
        triton.cdiv(max_seqlen_q, queries_per_step),
    )

    backward_kernel_dq[grid_dq](
        q,
        k,
        v,
        o,
        topk_idx,
        lse,
        delta,
        do,
        dq,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqblocks,
        cu_block_include_tokens,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dq.stride(0),
        dq.stride(1),
        dq.stride(2),
        kv_heads,
        query_head_groups,
        headdim,
        topk,
        queries_per_step,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK=BLOCK,
        QUERY_HEAD_GROUPS=QUERY_HEAD_GROUPS_CONST,
        BLOCK_SEL=BLOCK_SEL,
        num_warps=num_warps,
        num_stages=3,
    )

    # 2. Compute dK/dV using a true KV-head-centric kernel. Each program owns
    # one (batch, kv_head, kv_tile), flattens the grouped query heads into the
    # matmul row dimension, and stores the fully reduced dK/dV tile once.
    DKDV_QUERY_HEAD_GROUPS = triton.next_power_of_2(query_head_groups)
    BLOCK_Q = max(1, 128 // DKDV_QUERY_HEAD_GROUPS)
    BLOCK_M = BLOCK_Q * DKDV_QUERY_HEAD_GROUPS

    # Select variant-specific pipeline parameters.
    variant = variant.lower()
    if variant == "a":
        # Minimal pipelining
        dkdv_num_warps = 8
        dkdv_num_stages = 1
    elif variant == "b":
        # Moderate pipelining
        dkdv_num_warps = 8
        dkdv_num_stages = 2
    elif variant == "c":
        # Aggressive pipelining
        dkdv_num_warps = 8
        dkdv_num_stages = 4
    else:
        # Default: same as original ultrafastest (num_stages=3)
        dkdv_num_warps = 8
        dkdv_num_stages = 3

    # This no-atomic variant requires a single writer per dK/dV tile.
    # Increasing SPLIT_K would need a separate reduction or atomics.
    SPLIT_K = 1

    # Accumulate/store dK/dV in float32, then cast to the input dtype.
    dk = torch.zeros(
        total_k,
        kv_heads,
        headdim,
        device=device,
        dtype=torch.float32,
    )
    dv = torch.zeros(
        total_k,
        kv_heads,
        headdim,
        device=device,
        dtype=torch.float32,
    )

    # Build reverse query mapping
    total_blocks = cu_block_include_tokens.shape[0] - 1
    cu_reverse_count, reverse_query_indices = _build_reverse_query_mapping(
        topk_idx,
        cu_seqlens_q,
        cu_seqblocks,
        total_blocks,
    )

    # Pad reverse_query_indices to prevent OOB reads in batched kernel loop.
    reverse_query_indices = torch.nn.functional.pad(
        reverse_query_indices,
        (0, BLOCK_Q),
        value=0,
    )

    # Compute grid dimensions for inline tile mapping
    seqblocks_per_batch = cu_seqblocks[1:] - cu_seqblocks[:-1]
    max_seqblocks = seqblocks_per_batch.max().item()

    block_lens = cu_block_include_tokens[1:] - cu_block_include_tokens[:-1]
    if block_lens.numel() > 0:
        max_tiles_per_block = (block_lens.max().item() + BLOCK - 1) // BLOCK
    else:
        max_tiles_per_block = 1
    max_tiles_per_block = max(max_tiles_per_block, 1)

    grid_dkdv = (batch_size, kv_heads, max_tiles_per_block * max_seqblocks * SPLIT_K)

    backward_kernel_dkdv_fastest[grid_dkdv](
        q,
        k,
        v,
        lse,
        delta,
        do,
        dk,
        dv,
        cu_seqlens_k,
        cu_block_include_tokens,
        cu_reverse_count,
        reverse_query_indices,
        cu_seqblocks,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        lse.stride(0),
        lse.stride(1),
        delta.stride(0),
        delta.stride(1),
        do.stride(0),
        do.stride(1),
        do.stride(2),
        dk.stride(0),
        dk.stride(1),
        dk.stride(2),
        dv.stride(0),
        dv.stride(1),
        dv.stride(2),
        cu_reverse_count.stride(0),
        cu_reverse_count.stride(1),
        reverse_query_indices.stride(0),
        reverse_query_indices.stride(1),
        kv_heads,
        query_head_groups,
        headdim,
        max_seqblocks,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
        BLOCK=BLOCK,
        BLOCK_Q=BLOCK_Q,
        BLOCK_M=BLOCK_M,
        QUERY_HEAD_GROUPS=DKDV_QUERY_HEAD_GROUPS,
        SPLIT_K=SPLIT_K,
        num_warps=dkdv_num_warps,
        num_stages=dkdv_num_stages,
    )

    # The kernel has already reduced across grouped query heads.
    dk = dk.to(k.dtype)
    dv = dv.to(v.dtype)

    return dq, dk, dv


# ============================================================================
# Autograd function - follows lucidrains' NSA class pattern
# ============================================================================


# Default variant for the backward pass (set by choose_variant or directly)
_default_variant = "a"


def set_variant(variant: str):
    """Set the default backward variant ('a', 'b', 'c', or 'default')."""
    global _default_variant
    _default_variant = variant


class NSA(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        topk_idx,
        cu_seqblocks,
        cu_block_include_tokens,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale=None,
    ):
        assert q.dtype in [
            torch.float16,
            torch.bfloat16,
        ], "Only fp16/bf16 supported"
        assert q.dtype == k.dtype == v.dtype
        assert topk_idx.dtype == torch.int32
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k.dtype == torch.int32

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        if softmax_scale is None:
            softmax_scale = 1.0 / math.sqrt(q.shape[-1])

        o, lse = native_sparse_attn_forward(
            q,
            k,
            v,
            topk_idx,
            cu_seqblocks,
            cu_block_include_tokens,
            cu_seqlens_q,
            cu_seqlens_k,
            softmax_scale,
        )

        ctx.save_for_backward(
            q,
            k,
            v,
            o,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            topk_idx,
            cu_seqblocks,
            cu_block_include_tokens,
        )
        ctx.softmax_scale = softmax_scale

        return o, lse

    @staticmethod
    def backward(ctx, do, _dlse):
        (
            q,
            k,
            v,
            o,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            topk_idx,
            cu_seqblocks,
            cu_block_include_tokens,
        ) = ctx.saved_tensors

        do = do.contiguous()

        dq, dk, dv = native_sparse_attn_backward(
            do,
            q,
            k,
            v,
            o,
            lse,
            topk_idx,
            cu_seqblocks,
            cu_block_include_tokens,
            cu_seqlens_q,
            cu_seqlens_k,
            ctx.softmax_scale,
            variant=_default_variant,
        )

        return dq, dk, dv, None, None, None, None, None, None


_native_sparse_attend = NSA.apply


# ============================================================================
# Public API - follows lucidrains' native_sparse_attend signature
# ============================================================================


def native_sparse_attend(
    q,  # [total_tokens, num_q_heads, dim]
    k,  # [total_tokens, num_kv_heads, dim]
    v,  # [total_tokens, num_kv_heads, dim]
    topk_idx,  # [num_kv_heads, total_tokens, topk]
    cu_seqblocks,  # [batch_size + 1]
    cu_block_include_tokens,  # [total_blocks + 1]
    cu_seqlens_q,  # [batch_size + 1]
    cu_seqlens_k,  # [batch_size + 1]
    softmax_scale: Optional[float] = None,
    return_lse: bool = False,
):
    """
    Variable-block-size native sparse attention.

    Adapted from lucidrains' native_sparse_attend for variable block sizes.
    Uses exp2/log2 based online softmax and optimized memory access patterns:
    - K^T transposed loading to avoid tl.trans() in QK computation
    - tl.make_block_ptr for Q/dO/dQ/dK/dV loads and stores
    - tl.make_block_ptr with tl.advance for dK/dV kernel K^T/V^T tiles

    Args:
        q: queries [total_tokens, num_q_heads, dim]
        k: keys [total_tokens, num_kv_heads, dim]
        v: values [total_tokens, num_kv_heads, dim]
        topk_idx: selected block indices [num_kv_heads, total_tokens, topk]
        cu_seqblocks: cumulative block count [batch_size + 1]
        cu_block_include_tokens: cumulative token count per block
            [total_blocks + 1]
        cu_seqlens_q: cumulative query lengths [batch_size + 1]
        cu_seqlens_k: cumulative key lengths [batch_size + 1]
        softmax_scale: optional scale (default: 1/sqrt(dim))
        return_lse: whether to return log-sum-exp

    Returns:
        out: attention output [total_tokens, num_q_heads, dim]
        lse: (optional) log-sum-exp [num_q_heads, total_tokens]
    """
    out, lse = _native_sparse_attend(
        q,
        k,
        v,
        topk_idx,
        cu_seqblocks,
        cu_block_include_tokens,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale,
    )

    if not return_lse:
        return out

    return out, lse
