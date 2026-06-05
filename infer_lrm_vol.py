# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from geometry.volsdf import VolSdf
from misc.dist_helper import get_fsdp_model, get_fsdp_model_frozen
from misc.io_helper import pathmgr
from misc.utils import restart_from_checkpoint
from models.multiview_encoder import DinoV2Encoder, DinoV3Encoder, mvencoder_base
from models.voldecoder import VolTransformer
from renderer.sdf_renderer import SdfRenderer


def create_vol_infer(path, mesh_resolution, gpu):
    checkpoint = "/".join([path, "checkpoints", "last.pth"])
    config = "/".join([path, "args.txt"])
    with pathmgr.open(config, "r") as f:
        config_lines = f.readlines()
    config = {}
    for line in config_lines:
        if line[0] == "#":
            continue
        line = line.strip()
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        config[key] = value

    return LRMVolInfer(
        checkpoint=checkpoint,
        transformer_depth=int(config["transformer_depth"]),
        patch_size=int(config["patch_size"]),
        mlp_depth=int(config["mlp_depth"]),
        mlp_brdf_depth=int(config["mlp_brdf_depth"]),
        mlp_geo_depth=int(config["mlp_geo_depth"]),
        mlp_dim=int(config["mlp_dim"]),
        embed_dim=int(config["embed_dim"]),
        volume_dim=int(config["volume_dim"]),
        volume_res=int(config["volume_res"]),
        volume_upsample_scale=int(config["volume_upsample_scale"]),
        mvencoder_type=config["mvencoder_type"],
        use_dino=(config["use_dino"]),
        bbox_radius=float(config["bbox_radius"]),
        use_weight_norm=(config.get("use_weight_norm", "False") == "True"),
        prediction_type=config["prediction_type"],
        encode_background=config["encode_background"],
        use_decomposed_embed=(config.get("use_decomposed_embed", "False") == "True"),
        sdf_inv_std=float(config["end_sdf_inv_std"]),
        num_samples_per_ray=256,
        mesh_resolution=mesh_resolution,
        gpu=gpu,
    )


class LRMVolInfer(nn.Module):
    def __init__(
        self,
        checkpoint,
        transformer_depth=24,
        patch_size=8,
        mlp_depth=3,
        mlp_brdf_depth=2,
        mlp_geo_depth=2,
        mlp_dim=32,
        embed_dim=1024,
        volume_dim=32,
        volume_res=16,
        volume_upsample_scale=4,
        mvencoder_type="nocam",
        use_dino="v2",
        bbox_radius=0.5,
        use_weight_norm=True,
        prediction_type="brdf",
        encode_background="none",
        use_decomposed_embed=True,
        sdf_inv_std=50.0,
        mesh_resolution=256,
        num_samples_per_ray=256,
        gpu=0,
    ) -> None:
        super().__init__()
        self.transformer_depth = transformer_depth
        self.patch_size = patch_size
        self.mlp_depth = mlp_depth
        self.mlp_brdf_depth = mlp_brdf_depth
        self.mlp_dim = mlp_dim
        self.embed_dim = embed_dim
        self.volume_dim = volume_dim
        self.volume_res = volume_res
        self.volume_upsample_scale = volume_upsample_scale
        self.mvencoder_type = mvencoder_type
        self.encode_background = encode_background
        self.use_dino = use_dino
        self.bbox_radius = bbox_radius
        self.use_weight_norm = use_weight_norm
        self.prediction_type = prediction_type
        self.encode_background = encode_background
        self.use_decomposed_embed = use_decomposed_embed
        self.sdf_inv_std = sdf_inv_std
        self.mesh_resolution = mesh_resolution
        self.render_step_size = 1.732 * 2 * self.bbox_radius / num_samples_per_ray

        if use_dino == "v2":
            feature_dim = 1024
        elif use_dino == "v3":
            feature_dim = 1280
        else:
            feature_dim = None
        self.mvencoder = mvencoder_base(
            mvencoder_type=mvencoder_type,
            with_bg=(encode_background == "augment"),
            embed_dim=embed_dim,
            patch_size=patch_size,
            feature_dim=feature_dim,
            num_heads=32,
        )
        if use_dino == "v2":
            self.dinoencoder = DinoV2Encoder(patch_size=patch_size)
        elif use_dino == "v3":
            self.dinoencoder = DinoV3Encoder(patch_size=patch_size)
        else:
            self.dinoencoder = None

        self.voldecoder = VolTransformer(
            vol_res=volume_res,
            output_dim=volume_dim,
            upsample_scale=volume_upsample_scale,
            embed_dim=embed_dim,
            depth=transformer_depth,
            cp_freq=0,
            use_weight_norm=use_weight_norm,
            use_decomposed_embed=use_decomposed_embed,
            num_q_heads=32,
            num_kv_heads=2,
        )
        self.volsdf = VolSdf(
            dim=mlp_dim,
            input_dim=volume_dim,
            rgb_depth=mlp_depth,
            brdf_depth=mlp_brdf_depth,
            sdf_depth=mlp_geo_depth,
            radius=(bbox_radius * 1.05),
            prediction_type=prediction_type,
            compute_normal=False,
        )
        self.renderer = SdfRenderer(
            pred_mode=prediction_type,
            radius=(bbox_radius * 1.05),
            num_samples_per_ray=num_samples_per_ray,
            auto_cast_dtype=torch.bfloat16,
        )
        restart_from_checkpoint(
            checkpoint,
            mvencoder=self.mvencoder,
            voldecoder=self.voldecoder,
            volsdf=self.volsdf,
            load_weights_only=True,
        )
        for para in self.mvencoder.parameters():
            para.requires_grad = False
        for para in self.voldecoder.parameters():
            para.requires_grad = False
        for para in self.volsdf.parameters():
            para.requires_grad = False

        self.dinoencoder = self.dinoencoder.cuda(gpu)
        self.dinoencoder = get_fsdp_model_frozen(self.dinoencoder, gpu)
        self.mvencoder = self.mvencoder.cuda(gpu)
        self.volsdf = self.volsdf.cuda(gpu)
        self.renderer = self.renderer.cuda(gpu)
        self.voldecoder = get_fsdp_model(self.voldecoder, gpu)

    def forward(
        self,
        batch,
        device,
        auto_cast_dtype=torch.bfloat16,
        infer_points=True,
        output_res=None,
    ):
        with torch.no_grad():
            images = batch["rgb_input"]
            batch_size, image_num, _, height, width = images.shape
            images = images.reshape(batch_size, image_num, -1, height, width)
            images = images.to(device=device, dtype=auto_cast_dtype)
            if self.encode_background != "none":
                bg = batch["bgs_input"].reshape(batch_size, image_num, 3, height, width)
                bg = bg.to(device=device, dtype=auto_cast_dtype)
            else:
                bg = None

            with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                if self.mvencoder_type == "plucker":
                    rays_o_input = batch["rays_o_input"]
                    rays_d_input = batch["rays_d_input"]

                    rays_o_input = rays_o_input.permute(0, 1, 4, 2, 3)
                    rays_d_input = rays_d_input.permute(0, 1, 4, 2, 3)

                    plucker_rays = torch.cat(
                        [rays_d_input, torch.cross(rays_o_input, rays_d_input, dim=2)],
                        dim=2,
                    )
                    plucker_rays = plucker_rays.to(device=device, dtype=auto_cast_dtype)
                else:
                    plucker_rays = None

                uv_input = batch["uv_input"].to(device=device, dtype=auto_cast_dtype)
                uv_input = uv_input.permute(0, 1, 4, 2, 3)

                if self.use_dino != "no":
                    feature = self.dinoencoder(images)
                else:
                    feature = None

                tokens = self.mvencoder(
                    images,
                    uv_input,
                    plucker_rays=plucker_rays,
                    x_bg=bg,
                    feature=feature,
                )
                volume, y, volume_feat = self.voldecoder(
                    tokens,
                    image_num,
                    return_feature=True,
                )

                if infer_points:
                    surface_point_arr = []
                    mask_arr = []
                    valid_mask_arr = []
                    rays_o_input = batch["rays_o_input"].to(device=device)
                    rays_d_input = batch["rays_d_input"].to(device=device)
                    if output_res is not None:
                        batch_size, image_num, input_size = rays_o_input.shape[:3]
                        if input_size != output_res:
                            rays_o_input = rays_o_input.permute(0, 1, 4, 2, 3)
                            rays_o_input = rays_o_input.reshape(
                                batch_size * image_num, 3, input_size, input_size
                            )
                            rays_o_input = F.interpolate(
                                rays_o_input, size=output_res, mode="bilinear"
                            )
                            rays_o_input = rays_o_input.reshape(
                                batch_size, image_num, 3, output_res, output_res
                            )
                            rays_o_input = rays_o_input.permute(0, 1, 3, 4, 2)

                            rays_d_input = rays_d_input.permute(0, 1, 4, 2, 3)
                            rays_d_input = rays_d_input.reshape(
                                batch_size * image_num, 3, input_size, input_size
                            )
                            rays_d_input = F.interpolate(
                                rays_d_input, size=output_res, mode="bilinear"
                            )
                            rays_d_input = rays_d_input.reshape(
                                batch_size, image_num, 3, output_res, output_res
                            )
                            rays_d_input = rays_d_input.permute(0, 1, 3, 4, 2)
                            rays_d_input = F.normalize(rays_d_input, dim=4)

                    for n in range(0, batch_size):
                        out = self.renderer(
                            volume[n : n + 1, :],
                            self.volsdf,
                            self.sdf_inv_std,
                            rays_o_input[n : n + 1, :],
                            rays_d_input[n : n + 1, :],
                            cams=None,
                            points_only=True,
                        )
                        surface_point_arr.append(out["points"])
                        mask_arr.append(out["mask"])
                        valid_mask_arr.append(out["valid_mask"])

                    dense_points = {}
                    dense_points["surface_points_input"] = torch.cat(
                        surface_point_arr, dim=0
                    )
                    dense_points["masks_input"] = torch.cat(mask_arr, dim=0)
                    dense_points["depth_masks_input"] = torch.cat(valid_mask_arr, dim=0)
                else:
                    dense_points = None

                sdf_volume_arr, alpha_volume_arr = [], []
                for n in range(0, batch_size):
                    sdf_volume = self.renderer.sample_point(
                        volume[n : n + 1, :],
                        self.volsdf,
                        mode="grid",
                        N=self.mesh_resolution,
                        reverse=True,
                    )
                    sdf_volume = sdf_volume["sdf"]
                    alpha_volume = self.renderer.get_alpha(
                        sdf_volume, self.render_step_size, self.sdf_inv_std
                    )
                    sdf_volume_arr.append(sdf_volume)
                    alpha_volume_arr.append(alpha_volume)

                sdf_volume_arr = torch.cat(sdf_volume_arr, dim=0)
                alpha_volume_arr = torch.cat(alpha_volume_arr, dim=0)

        return (volume_feat, y, sdf_volume_arr, alpha_volume_arr, volume, dense_points)
