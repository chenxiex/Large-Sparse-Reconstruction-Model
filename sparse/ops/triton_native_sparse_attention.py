# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Variable block size sparse attention — clean-room implementation.
Follows native_sparse_attention_pytorch (lucidrains) conventions."""

from typing import Optional

import torch

from .triton_native_sparse_attention_impl_kvcentric import native_sparse_attend


def spatial_selection_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_topk: torch.Tensor,
    cu_seqblocks: torch.Tensor,
    cu_block_include_tokens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    seqlens_q: torch.Tensor,
    seqlens_k: torch.Tensor,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    """Drop-in replacement maintaining the original function signature."""
    return native_sparse_attend(
        q,
        k,
        v,
        block_topk,
        cu_seqblocks,
        cu_block_include_tokens,
        cu_seqlens_q,
        cu_seqlens_k,
        softmax_scale=softmax_scale,
    )
