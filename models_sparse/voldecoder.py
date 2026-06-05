# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp

from models_sparse.seqparallel_utils import seq_parallel_token_blocks
from models_sparse.utils import BlockMix, BlockMixOneside, LayerNorm32, trunc_normal_


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

        self.dconv3d = nn.Linear(embed_dim, output_dim * (upsample_scale**3))
        if use_weight_norm:
            self.dconv3d = nn.utils.parametrizations.weight_norm(self.dconv3d)

    def forward(self, volume):
        volume = self.dconv3d(volume)
        return volume


class VolTransformer(nn.Module):
    def __init__(
        self,
        vol_res=64,
        upsample_scale=4,
        embed_dim=1024,
        output_dim=32,
        depth=24,
        num_q_heads=32,
        num_kv_heads=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        cp_freq=1,
        use_weight_norm=False,
        skip_links=(17, 11, 4),
        topk_volume=8,
        topk_multiimage=2,
        use_decomposed_embed=False,
        use_sequence_parallel=False,
        no_checkpoint_all_gather=False,
        fuse_all_layers=False,
    ):
        super().__init__()
        self.topk_multiimage = topk_multiimage
        self.topk_volume = topk_volume
        self.depth = depth

        self.embed_dim = embed_dim
        self.vol_res = vol_res
        self.upsample_scale = upsample_scale
        self.output_dim = output_dim
        self.num_token = vol_res * vol_res * vol_res
        self.fuse_all_layers = fuse_all_layers

        self.use_decomposed_embed = use_decomposed_embed
        self.use_sequence_parallel = use_sequence_parallel
        self.no_checkpoint_all_gather = no_checkpoint_all_gather

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

        fuse_prior_volfeat_list = []
        fuse_prior_imfeat_list = []
        block_list = []
        fuse_imfeat_ratio = len(skip_links) + 1

        for _ in range(depth):
            if self.fuse_all_layers:
                fuse_prior_volfeat_list.append(
                    nn.Linear(embed_dim, embed_dim, bias=False)
                )
                fuse_prior_imfeat_list.append(
                    nn.Linear(embed_dim * fuse_imfeat_ratio, embed_dim, bias=False)
                )
            block_list.append(
                BlockMix(
                    dim=embed_dim,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=LayerNorm32,
                    use_weight_norm=use_weight_norm,
                    topk_volume=self.topk_volume,
                    topk_multiimage=self.topk_multiimage,
                    use_sequence_parallel=use_sequence_parallel,
                    no_checkpoint_all_gather=no_checkpoint_all_gather,
                )
            )

        self.fuse_prior_volfeats = nn.ModuleList(fuse_prior_volfeat_list)
        self.fuse_prior_imfeats = nn.ModuleList(fuse_prior_imfeat_list)
        self.cross_blocks = nn.ModuleList(block_list)

        self.skip_links = skip_links
        skip_fuse_prior_volfeat_list = []
        skip_fuse_prior_imfeat_list = []
        skip_block_list = []
        for n in range(len(skip_links)):
            if self.fuse_all_layers:
                skip_fuse_prior_volfeat_list.append(
                    nn.Linear(embed_dim, embed_dim, bias=False)
                )
                skip_fuse_prior_imfeat_list.append(
                    nn.Linear(
                        embed_dim * (fuse_imfeat_ratio + 1), embed_dim, bias=False
                    )
                )
            elif n == 0:
                skip_fuse_prior_imfeat_list.append(
                    nn.Linear(embed_dim, embed_dim, bias=False)
                )
            skip_block_list.append(
                BlockMixOneside(
                    dim=embed_dim,
                    num_q_heads=num_q_heads,
                    num_kv_heads=num_kv_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=LayerNorm32,
                    use_weight_norm=use_weight_norm,
                    topk_volume=self.topk_volume,
                    topk_multiimage=self.topk_multiimage,
                    use_sequence_parallel=use_sequence_parallel,
                    no_checkpoint_all_gather=self.no_checkpoint_all_gather,
                )
            )
        self.skip_fuse_prior_volfeats = nn.ModuleList(skip_fuse_prior_volfeat_list)
        self.skip_fuse_prior_imfeats = nn.ModuleList(skip_fuse_prior_imfeat_list)
        self.skip_blocks = nn.ModuleList(skip_block_list)

        if self.fuse_all_layers:
            self.fuse_pred_volfeat = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.norm = LayerNorm32(embed_dim)

        self.upsampler = VolUpsampler(
            upsample_scale=upsample_scale,
            embed_dim=embed_dim,
            output_dim=output_dim,
            use_weight_norm=use_weight_norm,
        )
        self.apply(self._init_weights)
        self.cp_freq = int(cp_freq)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, LayerNorm32):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def prepare_volume_output(self, volume, volume_index, prior_volume):
        # Dense and sparse volume feature
        batch_size = volume_index[:, 0].max().item() + 1
        device = prior_volume.device

        sparse_volume = volume.reshape(
            -1,
            self.output_dim,
            self.upsample_scale,
            self.upsample_scale,
            self.upsample_scale,
        ).permute(0, 2, 3, 4, 1)
        sparse_volume = sparse_volume.reshape(-1, self.output_dim)

        # Compute coordinate
        sparse_coords = volume_index.clone().reshape(-1, 4, 1, 1, 1)
        sparse_coords = sparse_coords.repeat(
            1, 1, self.upsample_scale, self.upsample_scale, self.upsample_scale
        )
        sparse_coords = sparse_coords.permute(0, 2, 3, 4, 1)
        sparse_coords = sparse_coords * self.upsample_scale
        x_offset = torch.arange(self.upsample_scale).to(
            dtype=torch.int32, device=device
        )
        y_offset = torch.arange(self.upsample_scale).to(
            dtype=torch.int32, device=device
        )
        z_offset = torch.arange(self.upsample_scale).to(
            dtype=torch.int32, device=device
        )
        z_offset, y_offset, x_offset = torch.meshgrid(
            z_offset, y_offset, x_offset, indexing="ij"
        )
        sparse_coords[:, :, :, :, 1] += z_offset[None, :]
        sparse_coords[:, :, :, :, 2] += y_offset[None, :]
        sparse_coords[:, :, :, :, 3] += x_offset[None, :]
        sparse_coords = sparse_coords.reshape(-1, 4)

        seq_id = volume_index[1:, 0] - volume_index[:-1, 0]
        seq = torch.nonzero(seq_id) + 1
        cu_seqlens_sparse = []
        cu_seqlens_sparse.append(0)
        for n in range(batch_size - 1):
            cu_seqlens_sparse.append(seq[n].item())
        cu_seqlens_sparse.append(volume_index.shape[0])
        cu_seqlens_sparse = torch.tensor(
            cu_seqlens_sparse, device=device, dtype=torch.int32
        )
        cu_seqlens_sparse = cu_seqlens_sparse * (self.upsample_scale**3)
        seqlens_sparse = cu_seqlens_sparse[1:] - cu_seqlens_sparse[:-1]

        # Generate index volume for sparse feature
        index_volume = -torch.ones(
            batch_size,
            self.vol_res * self.upsample_scale,
            self.vol_res * self.upsample_scale,
            self.vol_res * self.upsample_scale,
            dtype=torch.int32,
            device=device,
        )
        for n in range(batch_size):
            start, end = cu_seqlens_sparse[n], cu_seqlens_sparse[n + 1]
            seqlen = seqlens_sparse[n]
            coords = sparse_coords[start:end, 1:4]
            sparse_id = torch.arange(seqlen, device=device, dtype=torch.int32)
            index_volume[n, coords[:, 0], coords[:, 1], coords[:, 2]] = sparse_id

        output = {
            "sparse_feat": sparse_volume,
            "dense_feat": prior_volume,
            "cu_seqlens": cu_seqlens_sparse,
            "seqlens": seqlens_sparse,
            "index_volume": index_volume,  # query dense feature when == -1
        }
        return output

    def forward(
        self,
        y,
        prior_imfeat_list,
        image_index,
        prior_volfeat,
        mask_volume,
        volume_index,
        prior_volume,
        volume_coords_idx,
        attn_map,
        im_num,
    ):
        batch_size = image_index[:, 0].max().item() + 1
        dtype = y.dtype

        if self.use_decomposed_embed:
            pos_embed_z = self.pos_embed_z.reshape(-1, self.embed_dim_z)
            pos_embed_y = self.pos_embed_y.reshape(-1, self.embed_dim_y)
            pos_embed_x = self.pos_embed_x.reshape(-1, self.embed_dim_x)
            pos_embed_z = pos_embed_z[volume_index[:, 1], :]
            pos_embed_y = pos_embed_y[volume_index[:, 2], :]
            pos_embed_x = pos_embed_x[volume_index[:, 3], :]
            x = torch.cat([pos_embed_z, pos_embed_y, pos_embed_x], dim=-1)
        else:
            x = self.pos_embed.repeat(batch_size, 1, 1).to(dtype=dtype)
            x = x.reshape(-1, self.embed_dim)
            x = x[mask_volume, :]
            x = x[volume_coords_idx, :]

        prior_imfeat = torch.cat(prior_imfeat_list, dim=1)

        y_list, sparse_info_arr = [], []
        if self.use_sequence_parallel:
            (
                vol_tokens_dist,
                im_tokens_dist,
                cu_seqlens_volume,
                cu_seqlens_image,
                attn_map,
            ) = seq_parallel_token_blocks(
                [x, prior_volfeat], [y, prior_imfeat], attn_map, mode="distribute"
            )
            x, prior_volfeat = vol_tokens_dist
            y, prior_imfeat = im_tokens_dist

        else:
            volcoords_gap = volume_index[1:, 0] - volume_index[:-1, 0]
            seq_id = torch.nonzero(volcoords_gap) + 1
            cu_seqlens_volume = [0]
            for n in range(0, batch_size - 1):
                cu_seqlens_volume.append(seq_id[n].item())
            cu_seqlens_volume.append(volume_index.shape[0])
            cu_seqlens_volume = torch.tensor(
                cu_seqlens_volume, device=x.device, dtype=torch.int32
            )

            imcoords_gap = image_index[1:, 0] - image_index[:-1, 0]
            seq_id = torch.nonzero(imcoords_gap) + 1
            cu_seqlens_image = [0]
            for n in range(0, batch_size - 1):
                cu_seqlens_image.append(seq_id[n].item())
            cu_seqlens_image.append(image_index.shape[0])
            cu_seqlens_image = torch.tensor(
                cu_seqlens_image, device=x.device, dtype=torch.int32
            )

        for idx in range(0, self.depth):
            blk = self.cross_blocks[idx]
            if self.fuse_all_layers:
                x = x + self.fuse_prior_volfeats[idx](prior_volfeat)
                y = y + self.fuse_prior_imfeats[idx](prior_imfeat)

            if self.cp_freq > 0 and idx % self.cp_freq == 0:
                x, y, sparse_info = cp.checkpoint(
                    blk,
                    x,
                    cu_seqlens_volume,
                    y,
                    cu_seqlens_image,
                    attn_map,
                    im_num,
                    use_reentrant=False,
                )
            else:
                x, y, sparse_info = blk(
                    x,
                    cu_seqlens_volume,
                    y,
                    cu_seqlens_image,
                    attn_map,
                    im_num,
                )
            if idx in self.skip_links:
                y_list.insert(0, y)
            sparse_info_arr.append(sparse_info)
        y_list.insert(0, y)

        for n, _ in enumerate(self.skip_links):
            blk = self.skip_blocks[n]
            if self.fuse_all_layers:
                x = x + self.skip_fuse_prior_volfeats[n](prior_volfeat)
                y = y_list[n + 1] + self.skip_fuse_prior_imfeats[n](
                    torch.cat([y_list[0], prior_imfeat], dim=1)
                )
            elif n == 0:
                y = y_list[n + 1] + self.skip_fuse_prior_imfeats[n](y_list[0])
            else:
                y = y_list[n + 1]

            if self.cp_freq > 0 and n % self.cp_freq == 0:
                x, sparse_info = cp.checkpoint(
                    blk,
                    x,
                    cu_seqlens_volume,
                    y,
                    cu_seqlens_image,
                    attn_map,
                    im_num,
                    use_reentrant=False,
                )
            else:
                x, sparse_info = blk(
                    x,
                    cu_seqlens_volume,
                    y,
                    cu_seqlens_image,
                    attn_map,
                    im_num,
                )
            sparse_info_arr.append(sparse_info)

        if self.fuse_all_layers:
            x_gate = torch.sigmoid(self.fuse_pred_volfeat(x))
            x_output = (
                x * x_gate[:, : self.embed_dim]
                + prior_volfeat * x_gate[:, self.embed_dim :]
            )
            x = self.norm(x_output)
        else:
            x = self.norm(x)

        # Turn sparse volume back to dense volume
        volume = self.upsampler(x)

        if self.use_sequence_parallel:
            volume_collect, _, _, _, _ = seq_parallel_token_blocks(
                [volume], [], attn_map, mode="collect"
            )
            volume = volume_collect[0]
        output = self.prepare_volume_output(volume, volume_index, prior_volume)

        return output, sparse_info_arr
