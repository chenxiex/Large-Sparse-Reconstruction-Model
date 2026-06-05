# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import io
import json
import random
import tarfile

import cv2
import Imath
import numpy as np
import OpenEXR
import scipy.ndimage as ndimage
import trimesh
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from misc.io_helper import pathmgr
from data_loader.utils import (
    compute_cropping_from_mask,
    crop_and_resize,
    importance_selection,
    linear_to_srgb,
    load_depth,
    load_one_frame,
    load_one_image,
    load_specular,
    srgb_to_linear,
    transform_cams,
    transform_normal,
    transform_points,
    transform_rays,
)


def custom_collate_fn(batch):
    # Separate the batch into different elements
    batch_dict = {key: [d[key] for d in batch] for key in batch[0]}

    # Use default collate for other elements
    for key in batch_dict:
        if key != "mesh":
            batch_dict[key] = default_collate(batch_dict[key])

    return batch_dict


class DtcDataset(Dataset):
    def __init__(
        self,
        root_dir,
        sep_gt_dir=None,
        model_list=None,
        mode="TRAIN",
        input_image_res=512,
        input_image_num=4,
        output_image_res=128,
        output_image_num=4,
        output_res_range=None,
        env_height: int = 32,
        env_width: int = 64,
        load_bg: bool = False,
        load_depth: bool = False,
        load_normal: bool = False,
        load_brdf: bool = False,
        load_mesh: bool = False,
        rotate_mesh_y_to_z: bool = True,
        relative_cam_pose: bool = False,
        perturb_range: float = 0.3,
        radius=0.5,
        cam_loc_scale=2,
        eva_input_views=None,
        eva_output_views=None,
        cropping_mode="importance",
        perturb_color=False,
        use_adobe_view_selection=False,
        test_model_num=-1,
        centralized_cropping=False,
        white_env=False,
        start_id=0,
        end_id=-1,
    ):
        super().__init__()

        if output_res_range is None:
            output_res_range = [128, 384]
        if eva_input_views is None:
            eva_input_views = [0]
        if eva_output_views is None:
            eva_output_views = [4, 5, 6, 7]

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

        if sep_gt_dir is not None:
            self.sep_gt_dir = sep_gt_dir
        else:
            self.sep_gt_dir = None

        if mode == "TEST" and test_model_num > 0:
            models = models[:test_model_num]

        self.models = models
        self.mode = mode
        self.input_image_res = input_image_res

        if isinstance(input_image_num, int):
            self.input_image_num_max = input_image_num
            self.input_image_num_min = input_image_num
        elif isinstance(input_image_num, list):
            self.input_image_num_max = input_image_num[1]
            self.input_image_num_min = input_image_num[0]
        else:
            raise ValueError("input_image_num must be an integer or list.")

        self.output_image_res = output_image_res
        self.output_image_num = output_image_num
        self.output_res_range = output_res_range
        self.env_height = env_height
        self.env_width = env_width
        self.load_bg = load_bg
        self.load_depth = load_depth
        self.load_normal = load_normal
        self.load_brdf = load_brdf
        self.load_mesh = load_mesh
        self.relative_cam_pose = relative_cam_pose
        self.rotate_mesh_y_to_z = rotate_mesh_y_to_z
        self.radius = radius
        self.cam_loc_scale = cam_loc_scale
        self.perturb_range = perturb_range
        self.eva_input_views = eva_input_views
        self.eva_output_views = eva_output_views
        self.cropping_mode = cropping_mode
        self.perturb_color = perturb_color
        self.use_adobe_view_selection = use_adobe_view_selection
        self.centralized_cropping = centralized_cropping
        self.white_env = white_env

    def __len__(self):
        return len(self.models)

    def decode_exr(self, data):
        exr = OpenEXR.InputFile(io.BytesIO(data))
        header = exr.header()
        dw = header["dataWindow"]
        channels = list(header["channels"].keys())
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        image = np.empty((size[1], size[0], len(channels)), dtype=np.float32)
        for i, c in enumerate(["R", "G", "B", "A"]):
            if c in channels:
                channel = exr.channel(c, Imath.PixelType(Imath.PixelType.FLOAT))
                image[:, :, i] = np.frombuffer(channel, dtype=np.float32).reshape(
                    (size[1], size[0])
                )
        if len(channels) == 1:
            image = image[:, :, 0]
        return image

    def decode_ldr(self, data):
        buffer = np.asarray(bytearray(data), np.uint8)
        image = cv2.imdecode(buffer, -1)
        if len(image.shape) == 3:
            image[:, :, 0:3] = image[:, :, 0:3][:, :, ::-1]
            image = np.ascontiguousarray(image)
        return image

    def _load_tar_file(self, img_dir, file_name, im_num=32, resize=True):
        images, names = [], []
        # surreal data format
        if pathmgr.exists(os.path.join(img_dir, file_name)):
            tar_data_path = os.path.join(img_dir, file_name)
            with pathmgr.open(tar_data_path, "rb") as fp:
                with tarfile.open(fileobj=fp) as tar:
                    for _, member in enumerate(tar.getmembers()):
                        data = tar.extractfile(member)
                        if ".exr" in member.name:
                            image = self.decode_exr(data.read())
                        elif (
                            ".png" in member.name
                            or ".hdr" in member.name
                            or ".jpg" in member.name
                        ):
                            image = self.decode_ldr(data.read())
                        else:
                            raise ValueError("Unrecognizable name %s." % member.name)

                        # Use INTER_AREA for downscaling, INTER_LINEAR for upscaling
                        if resize:
                            orig_h, orig_w = image.shape[:2]
                            if (
                                orig_h > self.input_image_res
                                or orig_w > self.input_image_res
                            ):
                                interpolation = cv2.INTER_AREA
                            else:
                                interpolation = cv2.INTER_LINEAR
                            image = cv2.resize(
                                image,
                                (self.input_image_res, self.input_image_res),
                                interpolation=interpolation,
                            )
                        images.append(image)
                        names.append(member.name)
        elif pathmgr.isdir(img_dir):
            # Fallback: load loose files when no tar archive is present.
            # Match files whose name starts with the tar's stem (e.g. "rgb.tar" -> "rgb"
            # matches "rgb0000000.png", "mask.tar" -> "mask*.png", etc.).
            stem = os.path.splitext(file_name)[0]
            valid_exts = (".exr", ".png", ".hdr", ".jpg")
            entries = sorted(
                f
                for f in pathmgr.ls(img_dir)
                if f.startswith(stem) and f.lower().endswith(valid_exts)
            )
            for entry in entries:
                file_path = os.path.join(img_dir, entry)
                with pathmgr.open(file_path, "rb") as fp:
                    raw = fp.read()
                if entry.lower().endswith(".exr"):
                    image = self.decode_exr(raw)
                else:
                    image = self.decode_ldr(raw)

                if resize:
                    orig_h, orig_w = image.shape[:2]
                    if (
                        orig_h > self.input_image_res
                        or orig_w > self.input_image_res
                    ):
                        interpolation = cv2.INTER_AREA
                    else:
                        interpolation = cv2.INTER_LINEAR
                    image = cv2.resize(
                        image,
                        (self.input_image_res, self.input_image_res),
                        interpolation=interpolation,
                    )
                images.append(image)
                names.append(entry)

        return np.stack(images, axis=0), names

    def _load_camera_poses(self, img_dir):
        if pathmgr.exists(os.path.join(img_dir, "CameraRig.json")):
            camera_info_path = os.path.join(img_dir, "CameraRig.json")
            with pathmgr.open(camera_info_path, "r") as fp:
                camera = json.load(fp)
                fov = camera["intrinsic"]["fov"] / 180.0 * np.pi
                origin_height = camera["intrinsic"]["height"]
                origin_width = camera["intrinsic"]["width"]
                camera_poses = camera["to_world"]
                for n in range(0, len(camera_poses)):
                    camera_poses[n] = np.array(camera_poses[n])
                camera_poses = np.stack(camera_poses, axis=0)
        else:
            raise ValueError(
                "%s does not exists." % os.path.join(img_dir, "CameraRig.json")
            )
        return camera_poses, fov, origin_height, origin_width

    def _load_exposures(self, img_dir):
        if pathmgr.exists(os.path.join(img_dir, "image_process_info.json")):
            exposure_info_path = os.path.join(img_dir, "image_process_info.json")
            with pathmgr.open(exposure_info_path, "r") as fp:
                camera = json.load(fp)
            exposures = camera["rgb"]["autoexposure_scale"]
            return exposures
        else:
            return None

    def _scale_image(self, im, exposure, exposures_mean):
        im = 0.5 * (im + 1)
        im = srgb_to_linear(im)
        im = im / exposure * exposures_mean
        im = linear_to_srgb(im)
        im = np.clip(im, 0, 1)
        im = 2 * im - 1
        return im

    def _rescale_image(self, im, mask=None, scale=None):
        im = 0.5 * (im + 1)
        im = srgb_to_linear(im)

        if scale is not None:
            im = im * scale
            im = linear_to_srgb(im)
            if mask is not None:
                im = im * mask + (1 - mask)
            im = np.clip(im, 0, 1)
            im = 2 * im - 1
            return im
        else:
            if mask is not None:
                mean_int = np.sum(im * mask) / max(np.sum(mask) * 3, 1)
            else:
                mean_int = np.sum(im) / im.size()

            scale = 0.08 / max(mean_int, 1e-6)
            scale = max(min(scale, 10), 1)
            im = im * scale
            im = linear_to_srgb(im)
            if mask is not None:
                im = im * mask + (1 - mask)
            im = np.clip(im, 0, 1)
            im = 2 * im - 1
            return im, scale

    def compute_depth_mask_and_surface(self, depth, rays_o, rays_d_un, mask):
        surface_points = rays_o + rays_d_un * depth[0, :, :, None]
        depth_mask = np.logical_and(mask[0, :, :] == 1, depth[0, :] < 10)
        depth_mask = np.logical_and(
            np.logical_and(depth_mask, surface_points[:, :, 0] > -self.radius),
            surface_points[:, :, 0] < self.radius,
        )
        depth_mask = np.logical_and(
            np.logical_and(depth_mask, surface_points[:, :, 1] > -self.radius),
            surface_points[:, :, 1] < self.radius,
        )
        depth_mask = np.logical_and(
            np.logical_and(depth_mask, surface_points[:, :, 2] > -self.radius),
            surface_points[:, :, 2] < self.radius,
        )
        depth_mask = ndimage.binary_erosion(depth_mask, structure=np.ones((3, 3)))
        depth_mask = depth_mask.astype(np.float32)[None, :, :]
        depth = depth * depth_mask
        surface_points = (rays_o + rays_d_un * depth[0, :, :, None]) * depth_mask[
            0, :, :, None
        ]
        return depth, depth_mask, surface_points

    def __getitem__(self, idx):
        idx = idx % len(self.models)

        model = self.models[idx]
        model_id = model.split("/")[-1]

        scene_file = os.path.join(model, "scene", "scene_info.json")
        with pathmgr.open(scene_file, "r") as fIn:
            scene_data = json.load(fIn)
        obj_location = np.array(scene_data["object_location"], dtype=np.float32)

        if self.load_mesh:
            glb_path = scene_data["model_file"]
            with pathmgr.open(glb_path, "rb") as fIn:
                scene = trimesh.load(fIn, file_type="glb")
            mesh = list(scene.geometry.values())[0]
            vertices = mesh.vertices
            v_min = np.min(vertices, axis=0)
            v_max = np.max(vertices, axis=0)
            center = (v_min + v_max) * 0.5
            vertices = vertices - center[None, :]
            dist = np.sqrt(np.sum(vertices * vertices, axis=1))
            max_dist = dist.max()
            mesh_scale = self.radius / max_dist
            vertices = vertices * mesh_scale

            if self.rotate_mesh_y_to_z:
                rot_mat = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32)
                vertices = vertices.transpose(1, 0)
                vertices = np.matmul(rot_mat, vertices)
                vertices = vertices.transpose(1, 0)

            mesh.vertices = vertices

        camera_poses, fov, origin_height, origin_width = self._load_camera_poses(
            os.path.join(model, "images")
        )
        camera_poses[:, :3, 3] = (
            camera_poses[:, :3, 3] - obj_location[None, :]
        ) * self.cam_loc_scale
        total_image_num = camera_poses.shape[0]
        if self.sep_gt_dir:
            camera_poses_out, fov_out, origin_height_out, origin_width_out = (
                self._load_camera_poses(
                    os.path.join(self.sep_gt_dir, model_id, "images")
                )
            )
            camera_poses_out[:, :3, 3] = (
                camera_poses_out[:, :3, 3] - obj_location[None, :]
            )

        exposures = self._load_exposures(os.path.join(model, "images"))
        if exposures is None:
            exposures = np.ones(camera_poses.shape[0], dtype=np.float32)
        exposures_mean = sum(exposures) / len(exposures)
        if self.sep_gt_dir:
            exposures_out = self._load_exposures(
                os.path.join(self.sep_gt_dir, model_id, "images")
            )
            if exposures_out is None:
                exposures_out = np.ones(camera_poses_out.shape[0], dtype=np.float32)

        if pathmgr.isfile(os.path.join(model, "images", "CropArea.json")):
            with pathmgr.open(
                os.path.join(model, "images", "CropArea.json"), "r"
            ) as fIn:
                crop_info = json.load(fIn)["rgb"]["crop_areas"]
        else:
            crop_info = None
        if self.sep_gt_dir is not None:
            if pathmgr.isfile(os.path.join(model, "images", "CropArea.json")):
                with pathmgr.open(
                    os.path.join(self.sep_gt_dir, model_id, "images", "CropArea.json"),
                    "r",
                ) as fIn:
                    crop_info_out = json.load(fIn)["rgb"]["crop_areas"]
            else:
                crop_info_out = None

        tared_images, image_names = self._load_tar_file(
            os.path.join(model, "images", "rgb"),
            "rgb.tar",
            im_num=total_image_num,
            resize=not (self.centralized_cropping and crop_info is None),
        )
        hdr_images = tared_images.dtype == np.float32
        if self.sep_gt_dir is not None:
            tared_images_out, _ = self._load_tar_file(
                os.path.join(self.sep_gt_dir, model_id, "images", "rgb"),
                "rgb.tar",
                im_num=total_image_num,
                resize=not (self.centralized_cropping and crop_info_out is None),
            )

        tared_masks, mask_names = self._load_tar_file(
            os.path.join(model, "images", "rgb"),
            "mask.tar",
            im_num=total_image_num,
            resize=not (self.centralized_cropping and crop_info is None),
        )
        if self.sep_gt_dir is not None:
            tared_masks_out, _ = self._load_tar_file(
                os.path.join(self.sep_gt_dir, model_id, "images", "rgb"),
                "mask.tar",
                im_num=total_image_num,
                resize=not (self.centralized_cropping and crop_info_out is None),
            )

        if self.load_depth:
            tared_depths, depth_names = self._load_tar_file(
                os.path.join(model, "images", "rgb"),
                "depth.tar",
                im_num=total_image_num,
                resize=not (self.centralized_cropping and crop_info is None),
            )
            if self.sep_gt_dir is not None:
                tared_depths_out, _ = self._load_tar_file(
                    os.path.join(self.sep_gt_dir, model_id, "images", "rgb"),
                    "depth.tar",
                    im_num=total_image_num,
                    resize=not (self.centralized_cropping and crop_info_out is None),
                )

        if self.load_normal:
            tared_normal, normal_names = self._load_tar_file(
                os.path.join(model, "images", "rgb"),
                "normal.tar",
                im_num=total_image_num,
                resize=not (self.centralized_cropping and crop_info is None),
            )
            if self.sep_gt_dir is not None:
                tared_normal_out, _ = self._load_tar_file(
                    os.path.join(self.sep_gt_dir, model_id, "images", "rgb"),
                    "normal.tar",
                    im_num=total_image_num,
                    resize=not (self.centralized_cropping and crop_info_out is None),
                )

        if self.load_brdf:
            tared_albedo, albedo_names = self._load_tar_file(
                os.path.join(model, "images", "rgb"),
                "albedo.tar",
                im_num=total_image_num,
                resize=not (self.centralized_cropping and crop_info is None),
            )
            if self.sep_gt_dir is not None:
                tared_albedo_out, _ = self._load_tar_file(
                    os.path.join(self.sep_gt_dir, model_id, "images", "rgb"),
                    "albedo.tar",
                    im_num=total_image_num,
                    resize=not (self.centralized_cropping and crop_info_out is None),
                )
        if self.load_brdf:
            tared_specular, specular_names = self._load_tar_file(
                os.path.join(model, "images", "rgb"),
                "metallic_roughness.tar",
                im_num=total_image_num,
                resize=not (self.centralized_cropping and crop_info is None),
            )
            if self.sep_gt_dir is not None:
                tared_specular_out, _ = self._load_tar_file(
                    os.path.join(self.sep_gt_dir, model_id, "images", "rgb"),
                    "metallic_roughness.tar",
                    im_num=total_image_num,
                    resize=not (self.centralized_cropping and crop_info_out is None),
                )

        frames = list(range(len(tared_images)))
        if self.mode == "TRAIN":
            if self.use_adobe_view_selection:
                random.shuffle(frames)
                input_frames = frames[0 : self.input_image_num_min]
                record = np.zeros(len(frames), dtype=np.uint8)
                for n in range(0, len(input_frames)):
                    input_pose = camera_poses[input_frames[n]]
                    input_trans = input_pose[:3, 3].reshape(1, 3)
                    other_trans = (
                        camera_poses[:, :3, 3]
                        if self.sep_gt_dir is None
                        else camera_poses_out[:, :3, 3]
                    )

                    input_dist = np.linalg.norm(input_trans, axis=-1)
                    other_dist = np.linalg.norm(other_trans, axis=-1)

                    cosine = (
                        np.sum(input_trans * other_trans, axis=-1)
                        / input_dist
                        / other_dist
                    )
                    cosine_check = cosine > 0.717
                    if self.sep_gt_dir is None:
                        cosine_check[input_frames[n]] = 0
                    record = np.logical_or(record, cosine_check)

                if np.sum(record.astype(np.int32)) < self.output_image_num:
                    gap = self.output_image_num - np.sum(record.astype(np.int32))
                    cnt = 0
                    for m in range(0, len(frames)):
                        if self.sep_gt_dir is None and m in input_frames:
                            continue
                        if record[m] == 0:
                            record[m] = 1
                            cnt += 1
                        if cnt == gap:
                            break
                output_frames = [n for n in range(0, len(frames)) if record[n]]
                random.shuffle(output_frames)
                output_frames = output_frames[0 : self.output_image_num]
                if self.sep_gt_dir is None:
                    output_frames = input_frames + output_frames
                    random.shuffle(output_frames)
                    output_frames = output_frames[0 : self.output_image_num]
                frames = frames[0 : self.input_image_num_max] + output_frames
            else:
                random.shuffle(frames)

        if self.mode == "TRAIN":
            input_frames = frames[0 : self.input_image_num_max]
        else:
            input_frames = [frames[x] for x in self.eva_input_views]
            input_frames = input_frames[0 : self.input_image_num_max]

        model_orig = model.replace("memcache_", "")
        image_names_input = [
            os.path.join(model_orig, "images", "rgb", image_names[x].split("/")[-1])
            for x in input_frames
        ]
        if self.load_depth:
            depth_names_input = [
                os.path.join(model_orig, "images", "rgb", depth_names[x].split("/")[-1])
                for x in input_frames
            ]

        (
            images_input,
            masks_input,
            rays_o_input,
            rays_d_input,
            rays_d_un_input,
            uv_input,
            K_input,
            cameras_input,
        ) = ([], [], [], [], [], [], [], [])
        if crop_info is not None or self.centralized_cropping:
            crop_input = []
        if self.load_depth:
            depths_input = []
            depth_masks_input = []
            surface_points_input = []
        if self.load_bg:
            bgs_input = []

        bg_color_cnt, bg_pixel_cnt = 0, 0
        for n in range(0, len(input_frames)):
            frame = input_frames[n]
            if self.centralized_cropping and crop_info is None:
                input_image_res = (origin_height, origin_width)
            else:
                input_image_res = self.input_image_res

            image_origin, rays_o, rays_d, camera, _, rays_d_un, uv, K = load_one_frame(
                fov=fov,
                im=tared_images[frame],
                c2w=camera_poses[frame],
                image_res=input_image_res,
                hdr_to_ldr=hdr_images,
            )
            exposure = exposures[frame]
            image_scaled = self._scale_image(image_origin, exposure, exposures_mean)

            if crop_info is not None:
                height, width = image_origin.shape[1:3]
                hs, he = crop_info[frame]["top"], crop_info[frame]["bottom"]
                ws, we = crop_info[frame]["left"], crop_info[frame]["right"]
                scale_h = origin_height / float(he - hs)
                scale_w = origin_width / float(we - ws)
                assert scale_h == scale_w
                K = K * scale_h

                crop_input.append(np.array([hs, he, ws, we]))

                rays_o = cv2.resize(
                    rays_o,
                    (origin_width, origin_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                rays_o = crop_and_resize(rays_o, hs, he, ws, we, height, width)
                rays_d = cv2.resize(
                    rays_d,
                    (origin_width, origin_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                rays_d = crop_and_resize(rays_d, hs, he, ws, we, height, width)
                rays_d_un = cv2.resize(
                    rays_d_un,
                    (origin_width, origin_height),
                    interpolation=cv2.INTER_LINEAR,
                )
                rays_d_un = crop_and_resize(rays_d_un, hs, he, ws, we, height, width)
                uv = cv2.resize(
                    uv, (origin_width, origin_height), interpolation=cv2.INTER_LINEAR
                )
                uv = crop_and_resize(uv, hs, he, ws, we, height, width)

            _, mask = load_one_image(
                tared_masks[frame],
                image_res=input_image_res,
                normalize=False,
            )

            if self.white_env:
                bg_color_cnt += np.sum((1 - mask) * 0.5 * (image_scaled + 1))
                bg_pixel_cnt += np.sum(1 - mask) * 3

            image = 0.5 * (image_scaled + 1) * mask + (1 - mask)
            image = np.clip(image, 0, 1)
            image = 2 * image - 1

            if self.relative_cam_pose:
                if n == 0:
                    cam_ext = camera[:16].reshape(4, 4)
                    cam_rot = cam_ext[0:3, 0:3].copy()
                camera = transform_cams(camera, cam_rot)
                rays_o = transform_rays(rays_o, cam_rot)
                rays_d = transform_rays(rays_d, cam_rot)
                rays_d_un = transform_rays(rays_d_un, cam_rot)

            if self.load_depth:
                depth = (
                    load_depth(tared_depths[frame], input_image_res)
                    * self.cam_loc_scale
                )
                depth, depth_mask, surface_points = self.compute_depth_mask_and_surface(
                    depth, rays_o, rays_d_un, mask
                )

            if self.load_bg:
                bg = image_scaled

            if crop_info is None and self.centralized_cropping:
                hs, he, ws, we = compute_cropping_from_mask(mask.squeeze())
                if isinstance(self.input_image_res, int):
                    height, width = self.input_image_res, self.input_image_res
                else:
                    height, width = self.input_image_res

                mask = mask.squeeze()
                mask = crop_and_resize(mask, hs, he, ws, we, height, width)
                mask = mask[None, :, :]

                rays_o = crop_and_resize(rays_o, hs, he, ws, we, height, width)

                rays_d = crop_and_resize(rays_d, hs, he, ws, we, height, width)
                rays_d = rays_d / np.linalg.norm(rays_d, axis=2)[:, :, None]

                rays_d_un = crop_and_resize(rays_d_un, hs, he, ws, we, height, width)
                uv = crop_and_resize(uv, hs, he, ws, we, height, width)

                image = image.transpose(1, 2, 0)
                image = crop_and_resize(image, hs, he, ws, we, height, width)
                image = image.transpose(2, 0, 1)

                scale_h = height / float(he - hs)
                scale_w = width / float(we - ws)
                assert scale_h == scale_w
                K = K * scale_h

                crop_input.append(np.array([hs, he, ws, we]))

                if self.load_depth:
                    depth = depth.squeeze()
                    depth = crop_and_resize(depth, hs, he, ws, we, height, width)
                    depth = depth[None, :, :]

                    depth_mask = depth_mask.squeeze()
                    depth_mask = crop_and_resize(
                        depth_mask, hs, he, ws, we, height, width
                    )
                    depth_mask = depth_mask[None, :, :]

                    surface_points = surface_points
                    surface_points = crop_and_resize(
                        surface_points, hs, he, ws, we, height, width
                    )

                if self.load_bg:
                    bg = bg.transpose(1, 2, 0)
                    bg = crop_and_resize(bg, hs, he, ws, we, height, width)
                    bg = bg.transpose(2, 0, 1)

            images_input.append(image)
            masks_input.append(mask)
            rays_o_input.append(rays_o)
            rays_d_input.append(rays_d)
            rays_d_un_input.append(rays_d_un)
            uv_input.append(uv)
            K_input.append(K)
            cameras_input.append(camera)

            if self.load_depth:
                depths_input.append(depth)
                depth_masks_input.append(depth_mask)
                surface_points_input.append(surface_points)

            if self.load_bg:
                bgs_input.append(bg)

        if self.relative_cam_pose:
            if self.load_mesh:
                vertices = mesh.vertices
                vertices = transform_points(vertices, cam_rot)
                mesh.vertices = vertices

        images_input = np.stack(images_input, axis=0)
        masks_input = np.stack(masks_input, axis=0)
        rays_o_input = np.stack(rays_o_input, axis=0)
        rays_d_input = np.stack(rays_d_input, axis=0)
        rays_d_un_input = np.stack(rays_d_un_input, axis=0)
        K_input = np.stack(K_input, axis=0)
        uv_input = np.stack(uv_input, axis=0)
        cameras_input = np.stack(cameras_input, axis=0)
        if crop_info is not None or self.centralized_cropping:
            crop_input = np.stack(crop_input, axis=0)
        if self.mode == "TRAIN":
            images_input, rescale = self._rescale_image(images_input, masks_input)
        if self.load_depth:
            depths_input = np.stack(depths_input, axis=0)
            depth_masks_input = np.stack(depth_masks_input, axis=0)
            surface_points_input = np.stack(surface_points_input, axis=0)
        if self.load_bg:
            bgs_input = np.stack(bgs_input, axis=0)
            if self.mode == "TRAIN":
                bgs_input = self._rescale_image(bgs_input, scale=rescale)

        if self.mode == "TRAIN":
            output_frames = frames[
                self.input_image_num_max : self.input_image_num_max
                + self.output_image_num
            ]
        else:
            output_frames = [frames[x] for x in self.eva_output_views]
            output_frames = output_frames[0 : self.output_image_num]

        if self.sep_gt_dir is None:
            image_names_output = [
                os.path.join(model_orig, "images", "rgb", image_names[x].split("/")[-1])
                for x in output_frames
            ]

            if self.load_depth:
                depth_names_output = [
                    os.path.join(
                        model_orig, "images", "rgb", depth_names[x].split("/")[-1]
                    )
                    for x in output_frames
                ]
            if self.load_normal:
                normal_names_output = [
                    os.path.join(
                        model_orig, "images", "rgb", normal_names[x].split("/")[-1]
                    )
                    for x in output_frames
                ]
            if self.load_brdf:
                albedo_names_output = [
                    os.path.join(
                        model_orig, "images", "rgb", albedo_names[x].split("/")[-1]
                    )
                    for x in output_frames
                ]
                specular_names_output = [
                    os.path.join(
                        model_orig, "images", "rgb", specular_names[x].split("/")[-1]
                    )
                    for x in output_frames
                ]
        else:
            image_names_output = [
                os.path.join(
                    self.sep_gt_dir.replace("memcache_", ""),
                    "images",
                    "rgb",
                    image_names[x].split("/")[-1],
                )
                for x in output_frames
            ]

            if self.load_depth:
                depth_names_output = [
                    os.path.join(
                        self.sep_gt_dir.replace("memcache_", ""),
                        "images",
                        "rgb",
                        depth_names[x].split("/")[-1],
                    )
                    for x in output_frames
                ]
            if self.load_normal:
                normal_names_output = [
                    os.path.join(
                        self.sep_gt_dir.replace("memcache_", ""),
                        "images",
                        "rgb",
                        normal_names[x].split("/")[-1],
                    )
                    for x in output_frames
                ]
            if self.load_brdf:
                albedo_names_output = [
                    os.path.join(
                        self.sep_gt_dir.replace("memcache_", ""),
                        "images",
                        "rgb",
                        albedo_names[x].split("/")[-1],
                    )
                    for x in output_frames
                ]
                specular_names_output = [
                    os.path.join(
                        self.sep_gt_dir.replace("memcache_", ""),
                        "images",
                        "rgb",
                        specular_names[x].split("/")[-1],
                    )
                    for x in output_frames
                ]

        (
            images_output,
            masks_output,
            rays_o_output,
            rays_d_output,
            rays_d_un_output,
            cameras_output,
        ) = ([], [], [], [], [], [])

        if self.load_depth:
            depths_output = []
            depth_masks_output = []
            surface_points_output = []
        if self.load_brdf:
            albedo_output = []
            roughness_output = []
            metallic_output = []
        if self.load_normal:
            normal_output = []
            normal_masks_output = []

        for frame in output_frames:
            if self.mode.upper() == "TRAIN":
                res = np.random.randint(
                    self.output_res_range[0], self.output_res_range[1] + 1
                )
            else:
                res = self.output_image_res

            image_origin, rays_o, rays_d, camera, _, rays_d_un, _, _ = load_one_frame(
                fov=fov if self.sep_gt_dir is None else fov_out,
                im=(
                    tared_images[frame]
                    if self.sep_gt_dir is None
                    else tared_images_out[frame]
                ),
                c2w=(
                    camera_poses[frame]
                    if self.sep_gt_dir is None
                    else camera_poses_out[frame]
                ),
                image_res=res,
                hdr_to_ldr=hdr_images,
            )
            exposure = (
                exposures[frame] if self.sep_gt_dir is None else exposures_out[frame]
            )
            image_scaled = self._scale_image(image_origin, exposure, exposures_mean)

            if self.sep_gt_dir is None:
                crop_info_out = crop_info
                origin_height_out = origin_height
                origin_width_out = origin_width
            if crop_info_out is not None:
                height, width = image_origin.shape[1:3]
                hs, he = crop_info_out[frame]["top"], crop_info_out[frame]["bottom"]
                ws, we = crop_info_out[frame]["left"], crop_info_out[frame]["right"]
                rays_o = cv2.resize(
                    rays_o,
                    (origin_width_out, origin_height_out),
                    interpolation=cv2.INTER_LINEAR,
                )
                rays_o = crop_and_resize(rays_o, hs, he, ws, we, height, width)
                rays_d = cv2.resize(
                    rays_d,
                    (origin_width_out, origin_height_out),
                    interpolation=cv2.INTER_LINEAR,
                )
                rays_d = crop_and_resize(rays_d, hs, he, ws, we, height, width)
                rays_d_un = cv2.resize(
                    rays_d_un,
                    (origin_width_out, origin_height_out),
                    interpolation=cv2.INTER_LINEAR,
                )
                rays_d_un = crop_and_resize(rays_d_un, hs, he, ws, we, height, width)
                uv = cv2.resize(
                    uv,
                    (origin_width_out, origin_height_out),
                    interpolation=cv2.INTER_LINEAR,
                )
                uv = crop_and_resize(uv, hs, he, ws, we, height, width)

            _, mask = load_one_image(
                im=(
                    tared_masks[frame]
                    if self.sep_gt_dir is None
                    else tared_masks_out[frame]
                ),
                image_res=res,
                normalize=False,
            )
            mask = mask[0:1, :, :]
            image = 0.5 * (image_scaled + 1) * mask + (1 - mask)
            image = np.clip(image, 0, 1)
            image = 2 * image - 1

            if self.relative_cam_pose:
                camera = transform_cams(camera, cam_rot)
                rays_o = transform_rays(rays_o, cam_rot)
                rays_d = transform_rays(rays_d, cam_rot)
                rays_d_un = transform_rays(rays_d_un, cam_rot)

            if self.load_depth:
                depth = (
                    load_depth(
                        (
                            tared_depths[frame]
                            if self.sep_gt_dir is None
                            else tared_depths_out[frame]
                        ),
                        res,
                    )
                    * self.cam_loc_scale
                )
                depth, depth_mask, surface_points = self.compute_depth_mask_and_surface(
                    depth, rays_o, rays_d_un, mask
                )

            if self.load_brdf:
                albedo, _ = load_one_image(
                    (
                        tared_albedo[frame]
                        if self.sep_gt_dir is None
                        else tared_albedo_out[frame]
                    ),
                    res,
                    normalize=True,
                    ldr_to_hdr=False,
                )
                albedo = 0.5 * (albedo + 1) * mask + (1 - mask)
                albedo = np.clip(albedo, 0, 1)
                albedo = 2 * albedo - 1

                roughness, metallic = load_specular(
                    (
                        tared_specular[frame]
                        if self.sep_gt_dir is None
                        else tared_specular_out[frame]
                    ),
                    res,
                )
                roughness = 0.5 * (roughness + 1) * mask + (1 - mask)
                roughness = np.clip(roughness, 0, 1)
                roughness = 2 * roughness - 1

                metallic = 0.5 * (metallic + 1) * mask + (1 - mask)
                metallic = np.clip(metallic, 0, 1)
                metallic = 2 * metallic - 1

            if self.load_normal:
                normal, _ = load_one_image(
                    (
                        tared_normal[frame]
                        if self.sep_gt_dir is None
                        else tared_normal_out[frame]
                    ),
                    res,
                )
                normal = normal * mask + (1 - mask)
                normal = np.clip(normal, -1, 1)
                if self.relative_cam_pose:
                    normal = transform_normal(normal, mask, cam_rot)

                # Filter out normal with reversed orientation
                rays_d_permuted = rays_d.transpose(2, 0, 1)
                dot_prod = np.sum(rays_d_permuted * normal, axis=0)
                normal_mask = np.logical_and(dot_prod < 0, mask[0, :] == 1)
                normal_mask = ndimage.binary_erosion(
                    normal_mask, structure=np.ones((3, 3))
                )
                normal_mask = normal_mask.astype(np.float32)[None, :, :]

            if self.mode.upper() == "TRAIN":
                height, width = image.shape[1:]
                crop_size = self.output_image_res
                if height < crop_size or width < crop_size:
                    raise ValueError(
                        f"Image size {height}, {width} is smaller than {crop_size}"
                    )

                if self.cropping_mode == "uniform":
                    hs = np.random.randint(0, height - crop_size + 1)
                    he = hs + crop_size
                    ws = np.random.randint(0, width - crop_size + 1)
                    we = ws + crop_size
                elif self.cropping_mode == "importance":
                    hs, he, ws, we = importance_selection(mask, crop_size)
                else:
                    raise ValueError(
                        f"Unrecognizable cropping mode {self.cropping_mode}"
                    )

                image = image[:, hs:he, ws:we]
                mask = mask[:, hs:he, ws:we]
                rays_o = rays_o[hs:he, ws:we, :]
                rays_d = rays_d[hs:he, ws:we, :]
                rays_d_un = rays_d_un[hs:he, ws:we, :]

                if self.load_depth:
                    depth = depth[:, hs:he, ws:we]
                    depth_mask = depth_mask[:, hs:he, ws:we]
                    surface_points = surface_points[hs:he, ws:we, :]

                if self.load_brdf:
                    albedo = albedo[:, hs:he, ws:we]
                    roughness = roughness[:, hs:he, ws:we]
                    metallic = metallic[:, hs:he, ws:we]

                if self.load_normal:
                    normal = normal[:, hs:he, ws:we]
                    normal_mask = normal_mask[:, hs:he, ws:we]

            images_output.append(image)
            masks_output.append(mask)
            rays_o_output.append(rays_o)
            rays_d_output.append(rays_d)
            rays_d_un_output.append(rays_d_un)
            cameras_output.append(camera)

            if self.load_depth:
                depths_output.append(depth)
                depth_masks_output.append(depth_mask)
                surface_points_output.append(surface_points)

            if self.load_brdf:
                albedo_output.append(albedo)
                roughness_output.append(roughness)
                metallic_output.append(metallic)

            if self.load_normal:
                normal = normal / np.maximum(
                    np.sqrt(np.sum(normal * normal, axis=0, keepdims=True)), 1e-6
                )
                normal = normal * normal_mask + (1 - normal_mask)
                normal_output.append(normal)
                normal_masks_output.append(normal_mask)

        if len(output_frames) > 0:
            images_output = np.stack(images_output, axis=0)
            masks_output = np.stack(masks_output, axis=0)
            if self.mode == "TRAIN":
                images_output = self._rescale_image(
                    images_output, masks_output, rescale
                )
            rays_o_output = np.stack(rays_o_output, axis=0)
            rays_d_output = np.stack(rays_d_output, axis=0)
            rays_d_un_output = np.stack(rays_d_un_output, axis=0)
            cameras_output = np.stack(cameras_output, axis=0)

        if self.load_depth:
            depths_output = np.stack(depths_output, axis=0)
            depth_masks_output = np.stack(depth_masks_output, axis=0)
            surface_points_output = np.stack(surface_points_output, axis=0)

        if self.load_brdf:
            albedo_output = np.stack(albedo_output, axis=0)
            roughness_output = np.stack(roughness_output, axis=0)
            metallic_output = np.stack(metallic_output, axis=0)

        if self.load_normal:
            normal_output = np.stack(normal_output, axis=0)
            normal_masks_output = np.stack(normal_masks_output, axis=0)

        if self.white_env and self.mode == "TRAIN":
            if bg_pixel_cnt == 0:
                raise ValueError("Input images has no bg pixels")
            white_scale = bg_pixel_cnt / bg_color_cnt

            images_input = 0.5 * (images_input + 1)
            images_output = 0.5 * (images_output + 1)
            if self.load_bg:
                bgs_input = 0.5 * (bgs_input + 1)

            images_input = images_input * white_scale
            images_output = images_output * white_scale
            if self.load_bg:
                bgs_input = bgs_input * white_scale

            images_input = images_input * masks_input + images_input * (1 - masks_input)
            images_output = images_output * masks_output + images_output * (
                1 - masks_output
            )

            images_input = np.clip(images_input, 0, 1)
            images_output = np.clip(images_output, 0, 1)
            if self.load_bg:
                bgs_input = np.clip(bgs_input, 0, 1)

            images_input = 2 * images_input - 1
            images_output = 2 * images_output - 1
            if self.load_bg:
                bgs_input = 2 * bgs_input - 1

        if self.perturb_color and self.mode == "TRAIN":
            images_input = 0.5 * (images_input + 1)
            images_output = 0.5 * (images_output + 1)
            if self.load_bg:
                bgs_input = 0.5 * (bgs_input + 1)

            if self.perturb_range > 0 and self.mode == "TRAIN":
                if self.white_env:
                    # Must turn darker to avoid totally white image.
                    perturb_scale = (np.random.random() - 1) * self.perturb_range
                else:
                    perturb_scale = (2 * np.random.random() - 1) * self.perturb_range
                perturb_scale = 2**perturb_scale
                scale = perturb_scale
            else:
                scale = 1

            images_input = images_input * masks_input * scale + images_input * (
                1 - masks_input
            )
            images_output = images_output * masks_output * scale + images_output * (
                1 - masks_output
            )
            if self.load_bg:
                bgs_input = bgs_input * scale

            images_input = np.clip(images_input, 0, 1)
            images_output = np.clip(images_output, 0, 1)
            if self.load_bg:
                bgs_input = np.clip(bgs_input, 0, 1)

            images_input = 2 * images_input - 1
            images_output = 2 * images_output - 1
            if self.load_bg:
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
            "origin_size": np.array([origin_height, origin_width]),
        }
        if crop_info is not None or self.centralized_cropping:
            out["crop_input"] = crop_input
        if self.relative_cam_pose:
            out["init_cam_rot"] = cam_rot
        if fov is not None:
            out["fov"] = fov / np.pi * 180
        if self.load_depth:
            out["depth_input"] = depths_input.astype(np.float32)
            out["depth_masks_input"] = depth_masks_input.astype(np.float32)
            out["surface_points_input"] = surface_points_input.astype(np.float32)
            out["depth_names_input"] = depth_names_input
        if self.load_bg:
            out["bgs_input"] = np.clip(bgs_input, -1, 1)

        if len(output_frames) > 0:
            out["rgb_output"] = np.clip(images_output.astype(np.float32), -1, 1)
            out["rgb_names_output"] = image_names_output
            out["mask_output"] = np.clip(masks_output.astype(np.float32), 0, 1)
            out["mask_names_output"] = image_names_output
            out["rays_o_output"] = rays_o_output.astype(np.float32)
            out["rays_d_output"] = rays_d_output.astype(np.float32)
            out["cameras_output"] = cameras_output.astype(np.float32)

            if self.load_depth:
                out["depth_output"] = depths_output.astype(np.float32)
                out["depth_names_output"] = depth_names_output
                out["depth_masks_output"] = depth_masks_output.astype(np.float32)
                out["surface_points_output"] = surface_points_output.astype(np.float32)

            if self.load_brdf:
                out["albedo_output"] = np.clip(albedo_output.astype(np.float32), -1, 1)
                out["roughness_output"] = np.clip(
                    roughness_output.astype(np.float32), -1, 1
                )
                out["metallic_output"] = np.clip(
                    metallic_output.astype(np.float32), -1, 1
                )

                out["albedo_names_output"] = albedo_names_output
                out["roughness_names_output"] = specular_names_output
                out["metallic_names_output"] = specular_names_output

            if self.load_normal:
                out["normal_output"] = normal_output.astype(np.float32)
                out["normal_masks_output"] = normal_masks_output.astype(np.float32)
                out["normal_names_output"] = normal_names_output

        if self.load_mesh:
            out["mesh"] = mesh
            out["mesh_name"] = glb_path

        return out
