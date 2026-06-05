# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class LayerNorm32(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.float()).type(x.dtype)


class SparseResBlock(nn.Module):
    def __init__(
        self,
        channels: int,
    ):
        super().__init__()
        self.channels = channels

        self.act_layers = nn.Sequential(
            nn.Linear(self.channels, self.channels),
            nn.SiLU(),
            nn.Linear(self.channels, self.channels),
        )
        self.norm = LayerNorm32(self.channels, elementwise_affine=False)

    def forward(self, x):
        h = self.act_layers(x)
        h = self.norm(x + h)
        return h
