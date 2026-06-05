# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


def build_pytorch_mlp(input_dim, hidden_dim, output_dim, depth=10, bias=False):
    if depth == 0:
        return nn.Linear(input_dim, output_dim, bias=bias)
    mlp = []
    mlp.append(nn.Linear(input_dim, hidden_dim, bias=bias))
    mlp.append(nn.ReLU())
    for _ in range(depth - 1):
        mlp.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
        mlp.append(nn.ReLU())
    mlp.append(nn.Linear(hidden_dim, output_dim, bias=bias))
    mlp = nn.Sequential(*mlp)
    return mlp


def grid_sample_sparse(volume, pos):
    batch_id = volume["batch_id"]
    dense_feat = volume["dense_feat"][batch_id, :]
    index_volume = volume["index_volume"][batch_id, :]
    cu_seqlens = volume["cu_seqlens"]
    start, end = cu_seqlens[batch_id], cu_seqlens[batch_id + 1]
    sparse_feat = volume["sparse_feat"][start:end, :]

    dense_res = dense_feat.shape[-1]
    sparse_res = index_volume.shape[0]
    scale_factor = sparse_res // dense_res
    sparse_feat_num = sparse_feat.shape[0]

    pos = torch.clamp(0.5 * (pos + 1), 0, 1) * (sparse_res - 1)
    point_num = pos.shape[0]
    feat_dim = dense_feat.shape[0]
    dense_feat = dense_feat.reshape(feat_dim, -1).transpose(1, 0)
    dtype, device = dense_feat.dtype, dense_feat.device

    z_arr = [
        torch.floor(pos[:, 0]).to(torch.int32),
        torch.ceil(pos[:, 0]).to(torch.int32),
    ]
    y_arr = [
        torch.floor(pos[:, 1]).to(torch.int32),
        torch.ceil(pos[:, 1]).to(torch.int32),
    ]
    x_arr = [
        torch.floor(pos[:, 2]).to(torch.int32),
        torch.ceil(pos[:, 2]).to(torch.int32),
    ]

    feature_sum = 0
    for z in range(0, 2):
        for y in range(0, 2):
            for x in range(0, 2):
                if z == 0:
                    z_weight = 1 - (pos[:, 0] - z_arr[0])
                else:
                    z_weight = pos[:, 0] - z_arr[0]

                if y == 0:
                    y_weight = 1 - (pos[:, 1] - y_arr[0])
                else:
                    y_weight = pos[:, 1] - y_arr[0]

                if x == 0:
                    x_weight = 1 - (pos[:, 2] - x_arr[0])
                else:
                    x_weight = pos[:, 2] - x_arr[0]

                weight = x_weight * y_weight * z_weight
                index = index_volume[z_arr[z], y_arr[y], x_arr[x]]

                index_sparse = index[index >= 0]
                index_sparse = torch.clamp(index_sparse, 0, sparse_feat_num - 1)

                index_dense_z = z_arr[z][index < 0] // scale_factor
                index_dense_z = torch.clamp(index_dense_z, 0, dense_res - 1)
                index_dense_y = y_arr[y][index < 0] // scale_factor
                index_dense_y = torch.clamp(index_dense_y, 0, dense_res - 1)
                index_dense_x = x_arr[x][index < 0] // scale_factor
                index_dense_x = torch.clamp(index_dense_x, 0, dense_res - 1)
                index_dense = (
                    index_dense_z * dense_res * dense_res
                    + index_dense_y * dense_res
                    + index_dense_x
                )
                feature_sparse = sparse_feat[index_sparse, :]
                feature_dense = dense_feat[index_dense, :]

                feature = torch.zeros((point_num, feat_dim), dtype=dtype, device=device)
                feature[index < 0, :] = feature_dense
                feature[index >= 0, :] = feature_sparse

                feature_sum = feature_sum + weight[:, None] * feature

    return feature_sum
