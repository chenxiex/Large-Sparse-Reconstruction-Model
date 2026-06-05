# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Per-block window attention using Flash Attention.
Each variable-size block is treated as an independent sequence."""

import torch
from flash_attn import flash_attn_varlen_func


def sparse_window_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_block_include_tokens: torch.Tensor,
) -> torch.Tensor:
    """Compute self-attention independently within each block."""
    block_sizes = cu_block_include_tokens[1:] - cu_block_include_tokens[:-1]
    max_block_len = block_sizes.max().item()
    return flash_attn_varlen_func(
        q,
        k,
        v,
        cu_block_include_tokens,
        cu_block_include_tokens,
        max_block_len,
        max_block_len,
    )
