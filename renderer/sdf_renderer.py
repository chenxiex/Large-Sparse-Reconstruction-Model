# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import nerfacc
import numpy as np
import torch
import torch.nn as nn

from renderer.utils import validate_empty_rays


class SdfRenderer(nn.Module):
    def __init__(
        self,
        radius=0.5,
        num_samples_per_ray=128,
        pred_mode="rgb",
        opacity_threshold=0.98,
        occgrid_res=128,
        occgrid_thr=1e-4,
        auto_cast_dtype=torch.bfloat16,
    ):
        super().__init__()
        # Define the bounding box
        self.radius = radius
        self.register_buffer(
            "bbox",
            torch.as_tensor(
                [
                    [-radius, -radius, -radius],
                    [radius, radius, radius],
                ],
                dtype=torch.float32,
            ),
        )
        self.num_samples_per_ray = num_samples_per_ray
        self.render_step_size = (
            1.732 * 2 * radius / num_samples_per_ray
        )  # sqrt(3) — diagonal of unit cube in [-radius, radius]^3
        self.eps = 4 * radius / num_samples_per_ray
        self.estimator = nerfacc.OccGridEstimator(
            roi_aabb=self.bbox.view(-1), resolution=occgrid_res, levels=1
        )
        self.estimator.occs.fill_(True)
        self.estimator.binaries.fill_(True)
        self.occgrid_res = occgrid_res
        self.occgrid_thr = occgrid_thr

        if pred_mode != "rgb" and pred_mode != "brdf" and pred_mode != "both":
            raise ValueError(f"Undefined prediction mode {pred_mode}")
        self.mode = pred_mode
        self.opacity_threshold = opacity_threshold
        self.auto_cast_dtype = auto_cast_dtype

    def update_occupancy_grid(self, volume, volsdf, inv_std):
        def occ_eval_fn(x):
            chunk_size = 2097152
            point_num = x.shape[0]
            chunk_num = int(np.ceil(float(point_num) / chunk_size))
            alpha_arr = []
            for n in range(0, chunk_num):
                xs = n * chunk_size
                xe = min(xs + chunk_size, point_num)

                with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
                    sdf = volsdf(
                        positions=x[xs:xe, :],
                        volume=volume,
                        mode="sdf",
                    )
                    sdf = sdf.to(dtype=torch.float32)
                alpha = self.get_alpha(sdf, self.render_step_size, inv_std)
                alpha_arr.append(alpha)
            alpha = torch.cat(alpha_arr, dim=0)
            return alpha

        self.estimator._update(
            step=0,
            occ_eval_fn=occ_eval_fn,
            occ_thre=self.occgrid_thr,
            ema_decay=0.0,
        )

    def reset_occupancy_grid(self):
        self.estimator.occs.fill_(True)
        self.estimator.binaries.fill_(True)

    def compose_output(
        self,
        weights,
        values,
        ray_indices,
        n_rays,
        opacity,
        batch_size,
        im_num,
        height,
        width,
        is_scale=True,
    ):
        comp_fg = nerfacc.accumulate_along_rays(
            weights, values=values, ray_indices=ray_indices, n_rays=n_rays
        )
        comp_fg = comp_fg.view(batch_size, im_num, height, width, -1)
        if opacity is not None:
            comp = comp_fg + (1 - opacity)
        else:
            comp = comp_fg
        if is_scale:
            comp = 2 * comp - 1
        return comp

    def get_alpha(self, sdf, dists, inv_std):
        density = inv_std * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() * inv_std))
        alpha = 1 - torch.exp(-dists * density)
        return alpha

    def grid_sample(self, N, reverse=False):
        device = self.bbox.device
        dtype = self.bbox.dtype
        x = torch.linspace(-self.radius, self.radius, N, device=device)
        y = torch.linspace(-self.radius, self.radius, N, device=device)
        z = torch.linspace(-self.radius, self.radius, N, device=device)
        if reverse:
            z, y, x = torch.meshgrid(z, y, x)
        else:
            x, y, z = torch.meshgrid(x, y, z)
        points = torch.stack([x, y, z], dim=-1)
        points = points.reshape(-1, 3).to(device=device, dtype=dtype)
        return points

    def uniform_sample(self, N):
        device = self.bbox.device
        dtype = self.bbox.dtype
        points = torch.rand(N, 3, device=device) * 2 * self.radius - self.radius
        points = points.to(device=device, dtype=dtype)
        return points

    def evaluate_points_sdf(self, points, volume, volsdf, chunk=1048576):
        point_num = points.shape[1]
        p_interval = int(np.ceil(float(point_num) / chunk))

        sdf_arr = []
        for n in range(0, p_interval):
            rs = n * chunk
            re = min(rs + chunk, point_num)
            with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
                sdf = volsdf(
                    points[0, rs:re, :],
                    volume,
                    mode="sdf",
                )
                sdf = sdf.to(dtype=torch.float32)
            sdf_arr.append(sdf.reshape(re - rs, 1))
        sdf = torch.cat(sdf_arr, dim=0).reshape(1, point_num, 1)
        return sdf

    def sample_point(self, volume, volsdf, mode, N, reverse=False):
        # Only support when batch size is equal to 1
        if mode == "grid":
            points = self.grid_sample(N, reverse)
        elif mode == "uniform":
            points = self.uniform_sample(N)
        else:
            raise ValueError("Unrecognizable point sampling mode")
        points = points[None, :, :]
        out = {}
        out["sdf"] = self.evaluate_points_sdf(points, volume, volsdf)
        out["points"] = points
        return out

    def compute_numerical_normal(
        self,
        sdf,
        volsdf,
        points,
        volume,
        reverse=False,
    ):
        device = volume.device
        point_num = points.shape[0]

        eps = [[self.eps, 0, 0], [0, self.eps, 0], [0, 0, self.eps]]
        eps = torch.Tensor(eps).to(device=device)
        eps = eps.reshape(3, 1, 3)
        eps = -eps if reverse else eps
        points = points[None, :, :]
        points_eps = (points + eps).reshape(-1, 3)

        with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
            sdf_epsilon = volsdf(
                points_eps,
                volume,
                mode="sdf",
            )
            sdf_epsilon = sdf_epsilon.to(dtype=torch.float32)
        sdf_x, sdf_y, sdf_z = torch.split(sdf_epsilon, point_num, dim=0)
        grad_x = (sdf_x - sdf) / self.eps
        grad_y = (sdf_y - sdf) / self.eps
        grad_z = (sdf_z - sdf) / self.eps

        gradient = torch.cat([grad_x, grad_y, grad_z], dim=-1)
        gradient = -gradient if reverse else gradient
        normal = nn.functional.normalize(gradient, dim=-1, eps=1e-4)

        return normal, gradient

    def forward(
        self,
        volume,
        volsdf,
        inv_std,
        rays_o,
        rays_d,
        cams,
        compute_normal=False,
        compute_numerical_normal=False,
        return_intermediate=False,
        sampled_points_per_ray=None,
        points_only=False,
    ):
        batch_size, im_num, height, width = rays_o.shape[:4]
        rays_o_flatten = rays_o.reshape(-1, 3)
        rays_d_flatten = rays_d.reshape(-1, 3)
        n_rays = rays_o_flatten.shape[0]

        if sampled_points_per_ray is None:
            with torch.no_grad():
                ray_indices, t_starts_, t_ends_ = self.estimator.sampling(
                    rays_o_flatten,
                    rays_d_flatten,
                    sigma_fn=None,
                    near_plane=0,
                    far_plane=1e10,
                    render_step_size=self.render_step_size,
                    alpha_thre=0.0,
                    stratified=False,
                    cone_angle=0.0,
                    early_stop_eps=0,
                )
                ray_indices, t_starts_, t_ends_ = validate_empty_rays(
                    ray_indices, t_starts_, t_ends_
                )
                ray_indices = ray_indices.long()
                t_starts, t_ends = t_starts_[..., None], t_ends_[..., None]
        else:
            ray_indices = sampled_points_per_ray["ray_indices"]
            t_starts = sampled_points_per_ray["t_starts"]
            t_ends = sampled_points_per_ray["t_ends"]

        t_origins = rays_o_flatten[ray_indices]
        t_dirs = rays_d_flatten[ray_indices]
        t_positions = (t_starts + t_ends) / 2.0
        positions = t_origins + t_dirs * t_positions
        t_intervals = t_ends - t_starts

        if points_only:
            with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
                sdf = volsdf(
                    positions,
                    volume,
                    mode="sdf",
                )
                sdf = sdf.to(torch.float32)
        else:
            with torch.amp.autocast("cuda", dtype=self.auto_cast_dtype):
                pred = volsdf(
                    positions,
                    volume,
                    mode="all",
                )
                pred = pred.to(torch.float32)
                sdf, pred = pred[:, :1], pred[:, 1:]

        alpha = self.get_alpha(sdf, t_intervals, inv_std)
        weights, transmittance = nerfacc.render_weight_from_alpha(
            alpha[..., 0],
            ray_indices=ray_indices,
            n_rays=n_rays,
        )

        opacity = nerfacc.accumulate_along_rays(
            weights, values=None, ray_indices=ray_indices, n_rays=n_rays
        )
        points = nerfacc.accumulate_along_rays(
            weights, values=positions, ray_indices=ray_indices, n_rays=n_rays
        )
        points = points / (torch.clamp(opacity, min=self.opacity_threshold)).detach()

        opacity = opacity.view(batch_size, im_num, height, width, 1)
        points = points.view(batch_size, im_num, height, width, 3)
        valid_mask = (opacity > self.opacity_threshold).to(dtype=points.dtype).detach()

        out = {}
        if not points_only:
            if compute_numerical_normal:
                numerical_normal, gradient = self.compute_numerical_normal(
                    sdf,
                    volsdf,
                    positions,
                    volume,
                    reverse=False,
                )
            comp_pred = self.compose_output(
                weights,
                pred,
                ray_indices,
                n_rays,
                opacity,
                batch_size,
                im_num,
                height,
                width,
            )
            if self.mode == "rgb":
                if compute_normal:
                    comp_rgb, comp_normal = torch.split(comp_pred, [3, 3], dim=-1)
                    comp_normal = 0.5 * (comp_normal + 1)
                    out["rgb"] = comp_rgb
                    out["normal"] = comp_normal
                else:
                    out["rgb"] = comp_pred
            elif self.mode == "brdf":
                if compute_normal:
                    comp_albedo, comp_roughness, comp_metallic, comp_normal = (
                        torch.split(comp_pred, [3, 1, 1, 3], dim=-1)
                    )
                    comp_normal = 0.5 * (comp_normal + 1)
                    out["albedo"] = comp_albedo
                    out["roughness"] = comp_roughness
                    out["metallic"] = comp_metallic
                    out["normal"] = comp_normal
                else:
                    comp_albedo, comp_roughness, comp_metallic = torch.split(
                        comp_pred, [3, 1, 1], dim=-1
                    )
                    out["albedo"] = comp_albedo
                    out["roughness"] = comp_roughness
                    out["metallic"] = comp_metallic
            elif self.mode == "both":
                if compute_normal:
                    (
                        comp_rgb,
                        comp_albedo,
                        comp_roughness,
                        comp_metallic,
                        comp_normal,
                    ) = torch.split(comp_pred, [3, 3, 1, 1, 3], dim=-1)
                    comp_normal = 0.5 * (comp_normal + 1)
                    out["rgb"] = comp_rgb
                    out["albedo"] = comp_albedo
                    out["roughness"] = comp_roughness
                    out["metallic"] = comp_metallic
                    out["normal"] = comp_normal
                else:
                    comp_rgb, comp_albedo, comp_roughness, comp_metallic = torch.split(
                        comp_pred, [3, 3, 1, 1], dim=-1
                    )
                    out["rgb"] = comp_rgb
                    out["albedo"] = comp_albedo
                    out["roughness"] = comp_roughness
                    out["metallic"] = comp_metallic

            if compute_numerical_normal:
                comp_numerical_normal = self.compose_output(
                    weights,
                    numerical_normal,
                    ray_indices,
                    n_rays,
                    None,
                    batch_size,
                    im_num,
                    height,
                    width,
                    is_scale=False,
                )
                comp_numerical_normal = comp_numerical_normal.view(
                    batch_size, im_num, height, width, 3
                )
                comp_numerical_normal = nn.functional.normalize(
                    comp_numerical_normal, dim=-1, eps=1e-4
                )
                comp_numerical_normal = comp_numerical_normal * valid_mask + (
                    1 - valid_mask
                )
        out["mask"] = opacity
        out["valid_mask"] = valid_mask
        out["points"] = points

        if not points_only and cams is not None:
            cams = cams[:, :, 0:16].reshape(batch_size, im_num, 4, 4)
            z_axis = cams[:, :, :3, 2].unsqueeze(-2).unsqueeze(-2)
            depth = torch.sum((points - rays_o) * -z_axis, dim=-1, keepdim=True)
            depth = depth * valid_mask
            out["depth"] = depth

            if compute_normal:
                out["normal"] = comp_normal
            if compute_numerical_normal:
                out["numerical_normal"] = comp_numerical_normal
                out["gradient"] = gradient

        if return_intermediate:
            intermediate = {
                "ray_indices": ray_indices,
                "t_starts": t_starts,
                "t_ends": t_ends,
                "alpha": alpha,
                "transmittance": transmittance,
            }
            return out, intermediate
        else:
            return out
