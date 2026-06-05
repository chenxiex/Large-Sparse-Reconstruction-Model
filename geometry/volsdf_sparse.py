# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn

from geometry.utils import build_pytorch_mlp, grid_sample_sparse


class VolSdf(nn.Module):
    def __init__(
        self,
        dim=32,
        input_dim=32,
        rgb_depth=3,
        sdf_depth=2,
        normal_depth=3,
        brdf_depth=3,
        rho_depth=2,
        radius=0.5,
        prediction_type="rgb",
        sdf_bias=True,
        normal_bias=True,
        chunk_size=1048576,
        compute_normal=False,
    ):
        super().__init__()
        # Define the network
        self.dim = dim
        self.input_dim = input_dim
        self.sdf_depth = sdf_depth
        self.rgb_depth = rgb_depth
        self.brdf_depth = brdf_depth
        self.rho_depth = rho_depth
        self.radius = radius
        self.prediction_type = prediction_type
        self.sdf_bias = sdf_bias
        self.normal_bias = normal_bias
        self.chunk_size = chunk_size
        self.compute_normal = compute_normal

        self.mlp_sdf = build_pytorch_mlp(
            input_dim,
            dim,
            1,
            depth=sdf_depth,
            bias=False,
        )

        if self.prediction_type == "rgb" or self.prediction_type == "both":
            self.mlp_rgb = build_pytorch_mlp(
                input_dim,
                dim,
                3,
                depth=rgb_depth,
                bias=False,
            )

        if self.prediction_type == "brdf" or self.prediction_type == "both":
            self.mlp_basecolor = build_pytorch_mlp(
                input_dim,
                dim,
                3,
                depth=brdf_depth,
                bias=False,
            )
            self.mlp_specular = build_pytorch_mlp(
                input_dim,
                dim,
                2,
                depth=brdf_depth,
                bias=False,
            )

        if self.compute_normal:
            self.mlp_normal = build_pytorch_mlp(
                input_dim,
                dim,
                3,
                depth=normal_depth,
                bias=False,
            )
        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=0.25)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def sampling_points(self, volume, index_1, index_2, index_3):
        index = torch.stack([index_3, index_2, index_1], dim=-1)  # pytorch convention
        index = index.reshape(-1, 3)
        feature = grid_sample_sparse(volume, index)
        return feature

    def get_sdf_bias(self, positions):
        with torch.no_grad():
            positions = positions.reshape(-1, 3)
            dist = torch.sqrt(torch.sum(positions * positions, dim=-1, keepdim=True))
            sdf = dist - 0.1 * self.radius
            return sdf

    def get_normal_bias(self, positions):
        with torch.no_grad():
            positions = positions.reshape(-1, 3)
            normal = nn.functional.normalize(positions, dim=-1, eps=1e-4)
            return normal

    def forward_one_set(
        self,
        positions,
        volume,
        mode="all",
    ):
        p0 = positions[..., 0] / self.radius
        p1 = positions[..., 1] / self.radius
        p2 = positions[..., 2] / self.radius

        feature = self.sampling_points(volume, p0, p1, p2)

        if mode == "sdf":
            sdf = self.mlp_sdf(feature)
            sdf = torch.tanh(sdf) * self.radius * 1.732 * 1.5
            if self.sdf_bias:
                sdf = sdf + self.get_sdf_bias(positions)
            return sdf
        elif mode == "image":
            if self.prediction_type == "rgb" or self.prediction_type == "both":
                x = torch.sigmoid(self.mlp_rgb(feature))

            if self.prediction_type == "brdf" or self.prediction_type == "both":
                basecolor = torch.sigmoid(self.mlp_basecolor(feature))
                specular = torch.sigmoid(self.mlp_specular(feature))
                if self.prediction_type == "both":
                    x = torch.cat([x, basecolor, specular], dim=-1)
                else:
                    x = torch.cat([basecolor, specular], dim=-1)

            if self.compute_normal:
                normal = self.mlp_normal(feature)
                if self.normal_bias:
                    normal = normal + self.get_normal_bias(positions)
                normal = nn.functional.normalize(normal, dim=-1)
                x = torch.cat([x, normal], dim=-1)
            return x
        elif mode == "all":
            sdf = self.mlp_sdf(feature)
            sdf = torch.tanh(sdf) * self.radius * 1.732 * 1.5
            if self.sdf_bias:
                sdf = sdf + self.get_sdf_bias(positions)

            if self.prediction_type == "rgb" or self.prediction_type == "both":
                x = torch.sigmoid(self.mlp_rgb(feature))

            if self.prediction_type == "brdf" or self.prediction_type == "both":
                basecolor = torch.sigmoid(self.mlp_basecolor(feature))
                specular = torch.sigmoid(self.mlp_specular(feature))
                if self.prediction_type == "both":
                    x = torch.cat([x, basecolor, specular], dim=-1)
                else:
                    x = torch.cat([basecolor, specular], dim=-1)

            if self.compute_normal:
                normal = self.mlp_normal(feature)
                if self.normal_bias:
                    normal = normal + self.get_normal_bias(positions)
                normal = nn.functional.normalize(normal, dim=-1)
                x = torch.cat([x, normal], dim=-1)

            return torch.cat([sdf, x], dim=-1)
        else:
            raise ValueError(f"Unrecognizable mode {mode}.")

    def forward(
        self,
        positions,
        volume,
        mode="all",
    ):
        if torch.is_grad_enabled():
            return self.forward_one_set(
                positions,
                volume,
                mode,
            )
        else:
            point_num = positions.shape[0]
            chunk_num = int(np.ceil(float(point_num) / self.chunk_size))
            pred_arr = []
            for n in range(0, chunk_num):
                xs = n * self.chunk_size
                xe = min(xs + self.chunk_size, point_num)
                pred = self.forward_one_set(
                    positions[xs:xe, :],
                    volume,
                    mode,
                )
                pred_arr.append(pred)
            pred_arr = torch.cat(pred_arr, dim=0)
            return pred_arr
