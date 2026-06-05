# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from models_sparse.utils import zero_module


def freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)


class DinoV2Encoder(nn.Module):
    def __init__(
        self, patch_size=8, dinov2_model_name="dinov2_vitl14_reg", max_resolution=512
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_resolution = max_resolution

        supported_models = [
            "dinov2_vitl14",
            "dinov2_vitl14_reg",
        ]

        assert dinov2_model_name in supported_models, (
            f"we only support loading models from {supported_models}."
        )

        self._unregistered_dinov2_vit = [
            torch.hub.load("facebookresearch/dinov2", dinov2_model_name)
        ]

        self.dinov2_vit.to("cuda", non_blocking=True)
        self.dinov2_vit.eval()
        freeze_module(self.dinov2_vit)

    @property
    def dinov2_vit(self):
        return self._unregistered_dinov2_vit[0]

    def forward(self, x):
        B, I, _, h, w = x.shape
        ph, pw = h // self.patch_size, w // self.patch_size
        dino_ph = min(self.max_resolution, h) // self.patch_size
        dino_pw = min(self.max_resolution, w) // self.patch_size

        p = self.patch_size
        if p != 14:
            f = F.adaptive_avg_pool2d(
                input=x.view(-1, 3, h, w),
                output_size=(
                    min(h, self.max_resolution) // p * 14,
                    min(w, self.max_resolution) // p * 14,
                ),
            )
        else:
            f = x.view(-1, 3, h, w)

        total_image_num = f.shape[0]
        f_arr = []
        for n in range(0, total_image_num):
            f_arr.append(
                self.dinov2_vit(f[n : n + 1, :], is_training=True)["x_norm_patchtokens"]
            )

        f = torch.cat(f_arr, dim=0)

        if dino_ph == ph and dino_pw == pw:
            f = rearrange(f, "(B I) N C -> B (I N) C", B=B, I=I)
        else:
            f = f.reshape(B * I, dino_ph, dino_pw, -1)
            f = f.permute(0, 3, 1, 2)
            f = F.interpolate(f, (ph, pw), mode="bilinear")
            f = f.permute(0, 2, 3, 1)
            f = f.reshape(B, I * ph * pw, -1)
        return f


class DinoV3Encoder(nn.Module):
    def __init__(
        self, patch_size=8, dinov3_model_name="dinov3_vith16plus", max_resolution=2048
    ):
        super().__init__()
        self.patch_size = patch_size
        self.max_resolution = max_resolution

        supported_models = [
            "dinov3_vith16plus",
        ]

        assert dinov3_model_name in supported_models, (
            f"we only support loading models from {supported_models}."
        )

        self._unregistered_dinov3_vit = [
            torch.hub.load(
                "dinov3",
                dinov3_model_name,
                source="local",
                weights=os.path.join("dinov3", dinov3_model_name + ".pth"),
            )
        ]

        self.dinov3_vit.to("cuda", non_blocking=True)
        self.dinov3_vit.eval()
        freeze_module(self.dinov3_vit)

    @property
    def dinov3_vit(self):
        return self._unregistered_dinov3_vit[0]

    def image_transform(self, x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        device, dtype = x.device, x.dtype
        mean = torch.FloatTensor(mean).to(dtype=dtype, device=device)
        std = torch.FloatTensor(std).to(dtype=dtype, device=device)

        x = 0.5 * (x + 1)
        mean = mean.reshape(1, 3, 1, 1)
        std = std.reshape(1, 3, 1, 1)
        x = (x - mean) / std
        return x

    def forward(self, x):
        B, I, _, h, w = x.shape
        ph, pw = h // self.patch_size, w // self.patch_size
        dino_ph = min(self.max_resolution, h) // self.patch_size
        dino_pw = min(self.max_resolution, w) // self.patch_size

        p = self.patch_size
        if p != 16:
            f = F.adaptive_avg_pool2d(
                input=x.view(-1, 3, h, w),
                output_size=(
                    min(h, self.max_resolution) // p * 16,
                    min(w, self.max_resolution) // p * 16,
                ),
            )
        else:
            f = x.view(-1, 3, h, w)

        f = self.image_transform(f)
        total_image_num = f.shape[0]
        f_arr = []
        for n in range(0, total_image_num):
            f_arr.append(
                self.dinov3_vit(f[n : n + 1, :], is_training=True)["x_norm_patchtokens"]
            )

        f = torch.cat(f_arr, dim=0)

        if dino_ph == ph and dino_pw == pw:
            f = rearrange(f, "(B I) N C -> B (I N) C", B=B, I=I)
        else:
            f = f.reshape(B * I, dino_ph, dino_pw, -1)
            f = f.permute(0, 3, 1, 2)
            f = F.interpolate(f, (ph, pw), mode="bilinear")
            f = f.permute(0, 2, 3, 1)
            f = f.reshape(B, I * ph * pw, -1)

        return f


class MultiviewTransformer(nn.Module):
    """Multiview patch embedding encoder with positional encoding."""

    def __init__(
        self,
        mvencoder_type="nocam",
        patch_size=8,
        embed_dim=1024,
        with_bg=False,
        feature_dim=None,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.type = mvencoder_type
        if self.type == "plucker":
            self.patch_embed = nn.Conv2d(
                9, embed_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            self.patch_embed = nn.Conv2d(
                5, embed_dim, kernel_size=patch_size, stride=patch_size
            )

        if with_bg:
            self.patch_embed_bg = nn.Conv2d(
                3, embed_dim, kernel_size=patch_size, stride=patch_size
            )
            zero_module(self.patch_embed_bg)

        if feature_dim is not None:
            self.input_feature_proj = nn.Linear(
                feature_dim + embed_dim, embed_dim, bias=False
            )
            self.feature_dim = feature_dim

        if self.type == "nocam":
            self.pos_embed = nn.Parameter(
                torch.randn(2, embed_dim) / math.sqrt(float(embed_dim))
            )

        self.norm = nn.LayerNorm(embed_dim)
        self.patch_size = patch_size

    def interpolate_pos_encoding(self, image_num):
        ref_pos_embed = self.pos_embed[0:1, :]
        view_pos_embed = self.pos_embed[1:2, :]
        view_pos_embed = view_pos_embed.repeat(image_num - 1, 1)
        pos_embed = torch.cat([ref_pos_embed, view_pos_embed], dim=0)
        return pos_embed

    def prepare_tokens(self, x, uv, x_bg=None):
        B, I, _, h, w = x.shape
        x = x.reshape(B * I, -1, h, w)
        uv = uv.reshape(B * I, -1, h, w)
        x = torch.cat([x, uv], dim=1)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        if x_bg is not None:
            x_bg = x_bg.reshape(B * I, -1, w, h)
            x_bg = self.patch_embed_bg(x_bg).flatten(2).transpose(1, 2)
            x = x + x_bg
        # add positional encoding mark reference frame
        if self.type == "nocam":
            pos_embed = self.interpolate_pos_encoding(I)
            pos_embed = pos_embed.reshape(I, -1, self.embed_dim).repeat(B, 1, 1)
            dtype = x.dtype
            x = x + pos_embed.to(dtype=dtype)
        x = x.reshape(B, -1, self.embed_dim)

        return x

    def forward(
        self,
        x,
        uv,
        mask,
        index,
        image_coords_idx,
        plucker_rays=None,
        x_bg=None,
        feature=None,
    ):
        dtype = x.dtype

        if self.type == "nocam":
            x = self.prepare_tokens(x, uv, x_bg)
        elif self.type == "plucker":
            x = self.prepare_tokens(x, plucker_rays, x_bg)
        else:
            raise NotImplementedError

        x = x.reshape(-1, self.embed_dim)
        x = x[mask, :].to(dtype=dtype)
        x = x[image_coords_idx, :]

        if feature is not None:
            feature = feature.reshape(-1, self.feature_dim)
            feature = feature[mask, :].to(dtype=dtype)
            feature = feature[image_coords_idx, :]
            x = self.input_feature_proj(torch.cat([feature, x], dim=1))
            x = self.norm(x)

        return x


def mvencoder_base(
    mvencoder_type="nocam",
    patch_size=4,
    with_bg=False,
    embed_dim=1024,
    feature_dim=None,
):
    model = MultiviewTransformer(
        mvencoder_type=mvencoder_type,
        patch_size=patch_size,
        embed_dim=embed_dim,
        with_bg=with_bg,
        feature_dim=feature_dim,
    )
    return model
