# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from models.utils import BlockMix, BlockMixOneside, trunc_normal_


class VolUpsampler(nn.Module):
    def __init__(
        self,
        upsample_scale,
        embed_dim=1024,
        output_dim=32,
        use_weight_norm=False,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.upsample_scale = upsample_scale

        self.dconv3d = nn.Linear(
            embed_dim, output_dim * (upsample_scale**3), bias=False
        )
        if use_weight_norm:
            self.dconv3d = nn.utils.parametrizations.weight_norm(self.dconv3d)

    def forward(self, volume):
        volume = self.dconv3d(volume)
        return volume


class VolTransformer(nn.Module):
    def __init__(
        self,
        vol_res=16,
        upsample_scale=4,
        embed_dim=1024,
        output_dim=32,
        depth=24,
        num_q_heads=32,
        num_kv_heads=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        norm_layer=nn.LayerNorm,
        cp_freq=1,
        use_weight_norm=False,
        skip_links=(17, 11, 4),
        use_decomposed_embed=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.vol_res = vol_res
        self.num_token = vol_res * vol_res * vol_res

        self.use_decomposed_embed = use_decomposed_embed
        if use_decomposed_embed:
            self.embed_dim_z = self.embed_dim // 3
            self.embed_dim_x = (self.embed_dim - self.embed_dim_z) // 2
            self.embed_dim_y = self.embed_dim - self.embed_dim_z - self.embed_dim_x
            self.pos_embed_x = nn.Parameter(
                torch.randn(1, self.vol_res, self.embed_dim_x)
                * 1
                / math.sqrt(float(self.embed_dim))
            )
            self.pos_embed_y = nn.Parameter(
                torch.randn(1, self.vol_res, self.embed_dim_y)
                * 1
                / math.sqrt(float(self.embed_dim))
            )
            self.pos_embed_z = nn.Parameter(
                torch.randn(1, self.vol_res, self.embed_dim_z)
                * 1
                / math.sqrt(float(self.embed_dim))
            )
        else:
            self.pos_embed = nn.Parameter(
                torch.randn(1, self.num_token, embed_dim)
                * 1
                / math.sqrt(float(self.embed_dim))
            )

        block_list = []
        for _ in range(depth):
            block_list.append(
                BlockMix(
                    dim=embed_dim,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.cross_blocks = nn.ModuleList(block_list)

        self.skip_links = skip_links
        skip_block_list = []
        for _ in range(len(skip_links)):
            skip_block_list.append(
                BlockMixOneside(
                    dim=embed_dim,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    use_weight_norm=use_weight_norm,
                )
            )
        self.skip_blocks = nn.ModuleList(skip_block_list)

        self.norm = norm_layer(embed_dim)
        self.upsampler = VolUpsampler(
            upsample_scale=upsample_scale,
            embed_dim=embed_dim,
            output_dim=output_dim,
            use_weight_norm=use_weight_norm,
        )
        self.output_dim = output_dim
        self.upsample_scale = upsample_scale
        self.apply(self._init_weights)
        self.cp_freq = int(cp_freq)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def decode_volume(self, x):
        batch_size = x.shape[0]
        x = x.reshape(
            batch_size,
            self.vol_res,
            self.vol_res,
            self.vol_res,
            self.output_dim,
            self.upsample_scale,
            self.upsample_scale,
            self.upsample_scale,
        )
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)
        x = x.reshape(
            batch_size,
            self.output_dim,
            self.vol_res * self.upsample_scale,
            self.vol_res * self.upsample_scale,
            self.vol_res * self.upsample_scale,
        )
        return x

    def forward(self, y, image_num, return_feature=False):
        batch_size = y.shape[0]

        if self.use_decomposed_embed:
            pos_embed_x = self.pos_embed_x.reshape(
                1, 1, 1, self.vol_res, self.embed_dim_x
            )
            pos_embed_y = self.pos_embed_y.reshape(
                1, 1, self.vol_res, 1, self.embed_dim_y
            )
            pos_embed_z = self.pos_embed_z.reshape(
                1, self.vol_res, 1, 1, self.embed_dim_z
            )

            pos_embed_x = pos_embed_x.repeat(
                batch_size, self.vol_res, self.vol_res, 1, 1
            )
            pos_embed_y = pos_embed_y.repeat(
                batch_size, self.vol_res, 1, self.vol_res, 1
            )
            pos_embed_z = pos_embed_z.repeat(
                batch_size, 1, self.vol_res, self.vol_res, 1
            )
            x = torch.cat([pos_embed_z, pos_embed_y, pos_embed_x], dim=-1)
            x = x.reshape(batch_size, -1, self.embed_dim)
        else:
            x = self.pos_embed.repeat(batch_size, 1, 1)

        y_list = []
        if return_feature:
            x_list = []
        for idx in range(0, len(self.cross_blocks)):
            blk = self.cross_blocks[idx]
            if self.cp_freq > 0 and idx % self.cp_freq == 0:
                x, y = cp.checkpoint(
                    blk,
                    x,
                    y,
                    image_num,
                    use_reentrant=False,
                )
            else:
                x, y = blk(
                    x,
                    y,
                    image_num,
                )
            if idx in self.skip_links:
                y_list.insert(0, y)
        y_list.insert(0, y)

        for n, _ in enumerate(self.skip_links):
            blk = self.skip_blocks[n]
            if self.cp_freq > 0 and n % self.cp_freq == 0:
                x = cp.checkpoint(
                    blk,
                    x,
                    y_list[n + 1] if n != 0 else y_list[n + 1] + 0 * y[0],
                    use_reentrant=False,
                )
            else:
                x = blk(
                    x,
                    y_list[n + 1] if n != 0 else y_list[n + 1] + 0 * y[0],
                )

        if return_feature:
            x_list.append(x)
        x = self.norm(x)

        x = self.upsampler(x)
        volume = self.decode_volume(x)

        if return_feature:
            return volume, y_list, x_list
        else:
            return volume, y_list
