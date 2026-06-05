# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import json

import numpy as np
from torch.utils.data import Dataset

from misc.io_helper import pathmgr
from data_loader.utils import (
    compute_cropping_from_mask,
    compute_rays,
    crop_and_resize,
    linear_to_srgb,
    load_one_image,
    transform_cams,
    transform_rays,
)


class RealDataset(Dataset):
    def __init__(
        self,
        root_dir,
        model_list=None,
        mode="TEST",
        input_image_res=512,
        input_image_num=4,
        output_image_res=128,
        env_height: int = 32,
        env_width: int = 64,
        relative_cam_pose: bool = False,
        test_model_num=-1,
        centralized_cropping=False,
        normalized_input=False,
        eva_input_views=None,
        start_id=0,
        end_id=-1,
    ):
        super().__init__()

        assert mode == "TEST"
        if model_list is None:
            model_list = "train.txt" if mode.upper() == "TRAIN" else "test.txt"
            model_list = pathmgr.get_local_path(os.path.join(root_dir, model_list))

        with pathmgr.open(model_list, "r") as fIn:
            models = [line.strip() for line in fIn.readlines()]
            models = [
                m if os.path.isabs(m) or m.startswith(root_dir) else os.path.join(root_dir, m)
                for m in models
                if m
            ]
            start_id = min(max(start_id, 0), len(models) - 1)
            if end_id > start_id:
                end_id = min(end_id, len(models))
                models = models[start_id:end_id]
            else:
                models = models[start_id:]

        if mode.upper() == "TEST" and test_model_num > 0:
            models = models[:test_model_num]

        self.models = models
        self.mode = mode
        self.input_image_res = input_image_res
        self.input_image_num = input_image_num
        self.output_image_res = output_image_res
        self.env_height = env_height
        self.env_width = env_width
        self.eva_input_views = eva_input_views
        self.relative_cam_pose = relative_cam_pose
        self.centralized_cropping = centralized_cropping
        self.normalized_input = normalized_input

    def __len__(self):
        return len(self.models)

    def load_camera_poses(self, json_file):
        with pathmgr.open(json_file, "r") as fIn:
            data = json.load(fIn)
            if "camera_angle_x" in data:
                fov = data["camera_angle_x"]
            else:
                fov = [float(rec["camera_angle_x"]) for rec in data["frames"]]

            camera_poses = np.stack(
                [np.asarray(rec["transform_matrix"]) for rec in data["frames"]],
                axis=0,
            ).astype(np.float32)
            frames = [rec["file_path"] for rec in data["frames"]]
            if "scale" in data:
                cam_scale = data["scale"]
            else:
                cam_scale = 1
        return frames, camera_poses, fov, cam_scale

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        model = self.models[idx]
        model_id = model.split("/")[-1]

        input_file = os.path.join(model, "transforms_input.json")
        input_frames, input_poses, input_fov, cam_scale = self.load_camera_poses(
            input_file
        )
        if self.eva_input_views is not None:
            input_frames = [input_frames[x] for x in self.eva_input_views]
            input_poses = [input_poses[x] for x in self.eva_input_views]
            if isinstance(input_fov, list):
                input_fov = [input_fov[x] for x in self.eva_input_views]
        input_image_num = min(len(input_frames), self.input_image_num)
        input_frames = input_frames[0:input_image_num]
        input_frames = [os.path.join(model, x) for x in input_frames]

        for n, input_frame in enumerate(input_frames):
            if not input_frame.endswith(".jpg") and not input_frame.endswith(".png"):
                input_frame = input_frame + ".png"
                input_frames[n] = input_frame
        mask_frames = [x.replace("input", "mask") for x in input_frames]
        (
            images_input,
            image_names_input,
            bgs_input,
            masks_input,
            rays_o_input,
            rays_d_input,
            rays_d_un_input,
            uv_input,
            K_input,
            cameras_input,
        ) = ([], [], [], [], [], [], [], [], [], [])
        if self.centralized_cropping:
            crop_input = []
            origin_size = None
        for n in range(0, len(input_frames)):
            image_names_input.append(input_frames[n])
            if self.centralized_cropping:
                input_image_res = None
            else:
                input_image_res = self.input_image_res
            bg, _ = load_one_image(
                input_frames[n],
                image_res=input_image_res,
                normalize=False,
                ldr_to_hdr=True,
            )
            mask_im, mask = load_one_image(
                mask_frames[n],
                image_res=input_image_res,
                normalize=False,
            )
            if mask is None:
                mask = mask_im[0:1, :, :]
            image = bg * mask + (1 - mask)
            input_poses[n][:3, 3] = input_poses[n][:3, 3] * cam_scale

            if input_image_res is None:
                input_image_res = bg.shape[1:3]

            if isinstance(input_fov, list):
                rays_o, rays_d, rays_d_un, uv, K = compute_rays(
                    fov=input_fov[n], matrix=input_poses[n], res=input_image_res
                )
                camera_ext = input_poses[n].reshape(16)
                camera_int = np.array([input_fov[n], input_fov[n], 0.5, 0.5])
                camera = np.concatenate([camera_ext, camera_int])
            else:
                rays_o, rays_d, rays_d_un, uv, K = compute_rays(
                    fov=input_fov, matrix=input_poses[n], res=input_image_res
                )
                camera_ext = input_poses[n].reshape(16)
                camera_int = np.array([input_fov, input_fov, 0.5, 0.5])
                camera = np.concatenate([camera_ext, camera_int])

            if self.relative_cam_pose:
                if n == 0:
                    cam_ext = camera[:16].reshape(4, 4)
                    cam_rot = cam_ext[0:3, 0:3].copy()
                camera = transform_cams(camera, cam_rot)
                rays_o = transform_rays(rays_o, cam_rot)
                rays_d = transform_rays(rays_d, cam_rot)
                rays_d_un = transform_rays(rays_d_un, cam_rot)

            if self.centralized_cropping:
                origin_height, origin_width = bg.shape[1], bg.shape[2]
                if origin_size is None:
                    origin_size = np.array(
                        [origin_height, origin_width], dtype=np.float32
                    )

                hs, he, ws, we = compute_cropping_from_mask(mask.squeeze())
                bg = bg.transpose(1, 2, 0)
                bg = crop_and_resize(
                    bg, hs, he, ws, we, self.input_image_res, self.input_image_res
                )
                bg = bg.transpose(2, 0, 1)

                mask = mask.squeeze()
                mask = crop_and_resize(
                    mask, hs, he, ws, we, self.input_image_res, self.input_image_res
                )
                mask = mask[None, :, :]

                rays_o = crop_and_resize(
                    rays_o, hs, he, ws, we, self.input_image_res, self.input_image_res
                )
                rays_d = crop_and_resize(
                    rays_d, hs, he, ws, we, self.input_image_res, self.input_image_res
                )
                rays_d = rays_d / np.linalg.norm(rays_d, axis=2)[:, :, None]

                rays_d_un = crop_and_resize(
                    rays_d_un,
                    hs,
                    he,
                    ws,
                    we,
                    self.input_image_res,
                    self.input_image_res,
                )

                uv = crop_and_resize(
                    uv, hs, he, ws, we, self.input_image_res, self.input_image_res
                )

                image = image.transpose(1, 2, 0)
                image = crop_and_resize(
                    image, hs, he, ws, we, self.input_image_res, self.input_image_res
                )
                image = image.transpose(2, 0, 1)

                scale_h = self.input_image_res / float(he - hs)
                scale_w = self.input_image_res / float(we - ws)
                assert scale_h == scale_w
                K = K * scale_h

                crop_input.append(np.array([hs, he, ws, we], dtype=np.float32))

            bgs_input.append(bg)
            masks_input.append(mask)
            rays_o_input.append(rays_o)
            rays_d_input.append(rays_d)
            rays_d_un_input.append(rays_d_un)
            uv_input.append(uv)
            K_input.append(K)
            cameras_input.append(camera)
            images_input.append(image)

        images_input = np.stack(images_input, axis=0).astype(np.float32)
        masks_input = np.stack(masks_input, axis=0).astype(np.float32)
        rays_o_input = np.stack(rays_o_input, axis=0).astype(np.float32)
        rays_d_input = np.stack(rays_d_input, axis=0).astype(np.float32)
        rays_d_un_input = np.stack(rays_d_un_input, axis=0).astype(np.float32)
        uv_input = np.stack(uv_input, axis=0).astype(np.float32)
        K_input = np.stack(K_input, axis=0).astype(np.float32)
        cameras_input = np.stack(cameras_input, axis=0).astype(np.float32)
        bgs_input = np.stack(bgs_input, axis=0).astype(np.float32)

        output_file = os.path.join(model, "transforms_output.json")
        output_frames, output_poses, output_fov, _ = self.load_camera_poses(output_file)
        output_ids = []
        for x in output_frames:
            if x.endswith(".jpg") or x.endswith(".png"):
                output_ids.append(int(x.split("/")[-1].split(".")[0]))
            else:
                output_ids.append(int(x.split("/")[-1]))
        output_ids = np.array(output_ids)

        output_frames = [os.path.join(model, x) for x in output_frames]
        for n, output_frame in enumerate(output_frames):
            if not output_frame.endswith(".jpg") and not output_frame.endswith(".png"):
                output_frame = output_frame + ".png"
                output_frames[n] = output_frame

        (
            image_names_output,
            rays_o_output,
            rays_d_output,
            cameras_output,
        ) = ([], [], [], [])
        for n in range(0, len(output_frames)):
            image_names_output.append(output_frames[n])
            output_poses[n][:3, 3] = output_poses[n][:3, 3] * cam_scale

            if isinstance(output_fov, list):
                rays_o, rays_d, _, _, _ = compute_rays(
                    fov=output_fov[n],
                    matrix=output_poses[n],
                    res=self.output_image_res,
                )
                camera_ext = output_poses[n].reshape(16)
                camera_int = np.array([output_fov[n], output_fov[n], 0.5, 0.5])
                camera = np.concatenate([camera_ext, camera_int])
            else:
                rays_o, rays_d, _, _, _ = compute_rays(
                    fov=output_fov,
                    matrix=output_poses[n],
                    res=self.output_image_res,
                )
                camera_ext = output_poses[n].reshape(16)
                camera_int = np.array([output_fov, output_fov, 0.5, 0.5])
                camera = np.concatenate([camera_ext, camera_int])

            if self.relative_cam_pose:
                camera = transform_cams(camera, cam_rot)
                rays_o = transform_rays(rays_o, cam_rot)
                rays_d = transform_rays(rays_d, cam_rot)

            rays_o_output.append(rays_o)
            rays_d_output.append(rays_d)
            cameras_output.append(camera)

        rays_o_output = np.stack(rays_o_output, axis=0).astype(np.float32)
        rays_d_output = np.stack(rays_d_output, axis=0).astype(np.float32)
        cameras_output = np.stack(cameras_output, axis=0).astype(np.float32)

        if self.normalized_input:
            valid_pixels = np.mean(images_input, axis=1, keepdims=True)[
                masks_input > 0.9999
            ]
            valid_pixels_num = valid_pixels.shape[0]
            if valid_pixels_num == 0:
                scale = 1
            else:
                split = int(0.9 * valid_pixels_num)
                partition = np.partition(valid_pixels, split)[split]
                scale = np.clip(0.75 / np.maximum(partition, 1e-6), 0.625, 1.6)
        else:
            scale = 1

        images_input = images_input * masks_input * scale + images_input * (
            1 - masks_input
        )
        bgs_input = bgs_input * scale

        images_input = linear_to_srgb(np.clip(images_input, 0, 1))
        bgs_input = linear_to_srgb(np.clip(bgs_input, 0, 1))

        images_input = 2 * images_input - 1
        bgs_input = 2 * bgs_input - 1

        out = {
            "name": model_id,
            "rgb_input": np.clip(images_input, -1, 1),
            "rgb_names_input": image_names_input,
            "mask_input": np.clip(masks_input, 0, 1),
            "rays_o_input": rays_o_input,
            "rays_d_input": rays_d_input,
            "rays_d_un_input": rays_d_un_input,
            "uv_input": uv_input,
            "K_input": K_input,
            "cameras_input": cameras_input,
            "bgs_input": np.clip(bgs_input, -1, 1),
            "eva_output_views": output_ids,
            "scale": cam_scale,
            "fov": input_fov / np.pi * 180,
        }
        if self.centralized_cropping:
            out["crop_input"] = np.stack(crop_input, axis=0)
            out["origin_size"] = origin_size
        if self.relative_cam_pose:
            out["init_cam_rot"] = cam_rot

        rays_o_output = np.stack(rays_o_output, axis=0)
        rays_d_output = np.stack(rays_d_output, axis=0)
        cameras_output = np.stack(cameras_output, axis=0)
        out["rays_o_output"] = rays_o_output.astype(np.float32)
        out["rays_d_output"] = rays_d_output.astype(np.float32)
        out["cameras_output"] = cameras_output.astype(np.float32)
        out["rgb_names_output"] = image_names_output

        return out
