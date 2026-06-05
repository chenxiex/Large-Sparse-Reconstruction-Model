# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import os
import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch

import misc.utils as utils
from data_loader.dtc_dataset import DtcDataset
from data_loader.real_dataset import RealDataset
from misc.env_utils import fix_random_seeds, init_distributed_mode
from misc.io_helper import mkdirs, pathmgr


def create_output_dir(args):
    args.user = os.environ["USER"] or "default"
    args.output_dir = os.path.join(args.exp_root, args.exp_name)

    mkdirs(args.output_dir)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("LRM Mesh Rendering", add_help=False)

    parser.add_argument(
        "--exp_root",
        required=True,
        type=str,
        help="Path to save mesh rendering results",
    )
    parser.add_argument(
        "--exp_name",
        default="default_exp_name",
        type=str,
        help="Experiment name used as subdirectory under exp_root.",
    )

    # Model parameters (kept for compatibility with dataset)
    parser.add_argument(
        "--output_image_res", default=512, type=int, help="output image resolution"
    )
    parser.add_argument(
        "--bbox_radius",
        default=0.5,
        type=float,
        help="the size of the bounding box; 1.05 for sdf",
    )
    parser.add_argument(
        "--prediction_type",
        default="rgb",
        choices=["rgb", "brdf", "both"],
        type=str,
        help="The output of LRM, either rgb or brdf",
    )

    # Dataset parameters
    parser.add_argument(
        "--dataset_type",
        default="dtc_dataset",
        choices=["real_dataset", "dtc_dataset"],
        type=str,
        help="The input dataset format",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="Please specify path to the evaluation/test data.",
    )
    parser.add_argument(
        "--relative_cam_pose",
        action="store_true",
        help="whether to use relative camera poses.",
    )
    parser.add_argument(
        "--centralized_cropping",
        action="store_true",
        help="crop the center of the image.",
    )
    parser.add_argument(
        "--input_image_res",
        default=[256, 512],
        type=int,
        nargs=2,
        help="input image resolution",
    )
    parser.add_argument(
        "--image_num_per_batch",
        default=4,
        type=int,
        help="Number of input images per 3D model.",
    )
    parser.add_argument(
        "--output_image_num",
        default=10,
        type=int,
        help="output image num, only for dtc_dataset",
    )
    parser.add_argument(
        "--dtc_white_envmap",
        action="store_true",
        help="whether to load data with white environment map",
    )

    # Evaluate
    parser.add_argument(
        "--eva_input_views",
        nargs="*",
        type=int,
        default=None,
        help="Input view ids",
    )
    parser.add_argument(
        "--eva_output_views",
        nargs="*",
        type=int,
        default=None,
        help="Output view ids",
    )

    # Misc
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=1,
        type=int,
        help="Per-GPU batch-size: number of distinct 3D models loaded on one GPU.",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="the starting model id of the data list.",
    )
    parser.add_argument(
        "--end_id",
        type=int,
        default=-1,
        help="the ending model id of the data list.",
    )

    # Blender path
    parser.add_argument(
        "--blender_bin",
        type=str,
        default=None,
        help="Path to the Blender binary. Required when not on FB infra.",
    )

    # Mesh rendering options
    parser.add_argument(
        "--render_mesh",
        action="store_true",
        help="whether to render the textured mesh with environment maps",
    )
    parser.add_argument(
        "--render_mesh_mode",
        default="texture",
        choices=["texture", "relighting", "geometry"],
        type=str,
        help="rendering mode: texture (show textured mesh), relighting (PBR with env map), or geometry (render normal and depth in .exr)",
    )
    parser.add_argument(
        "--render_white_bg",
        action="store_true",
        help="render mesh with white background instead of environment map background",
    )
    parser.add_argument(
        "--camera_center_coord",
        action="store_true",
        help="whether to use camera center coordinate system for mesh rendering",
    )
    parser.add_argument(
        "--save_mesh_video",
        action="store_true",
        help="whether to save a mesh rendering video with rotating camera",
    )
    parser.add_argument(
        "--output_texture_res",
        type=int,
        default=1024,
        help="output texture resolution",
    )
    parser.add_argument(
        "--env_video_path",
        type=str,
        default=None,
        help="Path to a single environment map (.exr/.hdr) used for mesh video rendering. "
        "When set, overrides env_files so that one consistent env map is used.",
    )
    parser.add_argument(
        "--env_video_output_path",
        type=str,
        default="video",
        help="Path to save the mesh video.",
    )
    parser.add_argument(
        "--env_video_mean",
        type=float,
        default=0.5,
        help="mean of the environment map",
    )
    parser.add_argument(
        "--rotate_y_to_z",
        action="store_true",
        help="Rotate the mesh from Y-up to Z-up (Blender convention). "
        "Applies a 90-degree rotation around the X axis to mesh objects in Blender.",
    )

    return parser


def get_blender_bin(args):
    assert args.blender_bin is not None, "--blender_bin is required"
    return args.blender_bin


def test_mesh_rendering(local_rank, ngpus_per_node, args):
    init_distributed_mode(local_rank, ngpus_per_node, args)
    args = create_output_dir(args)
    fix_random_seeds(args.seed)

    print("*" * 80)
    print(
        f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for rendering"
    )
    print(args)
    print("*" * 80)

    # Setup dataset
    if args.eva_output_views is not None:
        args.output_image_num = min(len(args.eva_output_views), args.output_image_num)
        args.eva_output_views = args.eva_output_views[: args.output_image_num]

    if args.dataset_type == "dtc_dataset":
        dataset = DtcDataset(
            mode="TEST",
            root_dir=args.data_path,
            input_image_num=args.image_num_per_batch,
            input_image_res=args.input_image_res[-1],
            output_image_num=args.output_image_num,
            output_image_res=args.output_image_res,
            eva_input_views=args.eva_input_views,
            eva_output_views=args.eva_output_views,
            radius=args.bbox_radius,
            load_normal=False,
            load_depth=False,
            load_brdf=(
                args.prediction_type == "brdf" or args.prediction_type == "both"
            ),
            load_bg=False,
            relative_cam_pose=args.relative_cam_pose,
            centralized_cropping=args.centralized_cropping,
            start_id=args.start_id,
            end_id=args.end_id,
            white_env=args.dtc_white_envmap,
        )
    elif args.dataset_type == "real_dataset":
        dataset = RealDataset(
            mode="TEST",
            root_dir=args.data_path,
            input_image_num=args.image_num_per_batch,
            input_image_res=args.input_image_res[-1],
            output_image_res=args.output_image_res,
            relative_cam_pose=args.relative_cam_pose,
            centralized_cropping=args.centralized_cropping,
            start_id=args.start_id,
            end_id=args.end_id,
        )
    else:
        raise NotImplementedError(f"Unknown dataset type {args.dataset_type}")

    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,
    )
    print(f"Data loaded: there are {len(dataset)} 3D models.")

    blender_bin = get_blender_bin(args)

    start_time = time.time()
    print("Starting mesh rendering!")
    data_loader.sampler.set_epoch(0)

    render_meshes_from_dataset(
        data_loader,
        args.eva_output_views,
        args,
        blender_bin,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Rendering time {}".format(total_time_str))


def render_meshes_from_dataset(
    data_loader,
    eva_output_views,
    args,
    blender_bin,
):
    """Render meshes using camera poses and environment maps from the dataset."""

    for _, batch in enumerate(data_loader):
        model_ids = batch["name"]
        batch_size = len(model_ids)
        assert batch_size == 1

        model_id = model_ids[0]
        print("Model Id: %s" % model_id)

        if "eva_output_views" in batch:
            batch_eva_output_views = batch["eva_output_views"][0, :]
            batch_eva_output_views = batch_eva_output_views.numpy().tolist()
        else:
            batch_eva_output_views = eva_output_views

        # Get the mesh directory for this model (mesh is in the experiment folder)
        model_dir = os.path.join(args.output_dir, model_id)
        mesh_dir = os.path.join(model_dir, "mesh")

        # Check if mesh exists
        mesh_uv_path = os.path.join(mesh_dir, "mesh_uv.obj")
        if not pathmgr.exists(mesh_uv_path):
            print(f"Warning: Mesh not found at {mesh_uv_path}, skipping {model_id}")
            continue

        # Create output directory
        model_dir = os.path.join(args.output_dir, model_id)
        mkdirs(model_dir, is_main_process_only=False)

        # Get initial camera rotation for relative pose
        if args.relative_cam_pose:
            init_cam_arr = batch["init_cam_rot"]
        else:
            init_cam_arr = None

        if args.render_mesh:
            env_files = get_mesh_env_files(batch, 0)

            if args.render_mesh_mode == "geometry":
                render_out_dir = os.path.join(mesh_dir, "mesh_rendering_geometry")
            else:
                render_out_dir = os.path.join(mesh_dir, "mesh_rendering")
            mkdirs(render_out_dir)
            cameras_output_np = batch["cameras_output"][0, :].numpy()

            if args.relative_cam_pose:
                cam_rot = init_cam_arr[0, :].numpy()  # R_0 (3x3)
                rot_4x4 = np.eye(4, dtype=cameras_output_np.dtype)
                rot_4x4[:3, :3] = cam_rot
                n_cams = cameras_output_np.shape[0]
                exts = cameras_output_np[:, :16].reshape(n_cams, 4, 4)
                exts = rot_4x4 @ exts
                cameras_output_np[:, :16] = exts.reshape(n_cams, 16)

            if len(env_files) > 0 and len(env_files) == len(batch_eva_output_views):
                utils.render_images_from_mesh_multienv(
                    mesh_dir,
                    args.prediction_type,
                    blender_bin,
                    cameras_output_np,
                    batch_eva_output_views,
                    render_out_dir,
                    env_paths=env_files,
                    mode=args.render_mesh_mode,
                    show_bg=not args.render_white_bg,
                    camera_center_coord=args.camera_center_coord,
                    image_resolution=args.output_image_res,
                    rotate_y_to_z=args.rotate_y_to_z,
                    white_bg=args.render_white_bg,
                )
            else:
                utils.render_images_from_mesh(
                    mesh_dir,
                    args.prediction_type,
                    blender_bin,
                    cameras_output_np,
                    batch_eva_output_views,
                    render_out_dir,
                    env_path=env_files[0] if len(env_files) > 0 else None,
                    mode=(
                        args.render_mesh_mode
                        if args.render_mesh_mode == "geometry"
                        else (
                            "texture" if len(env_files) == 0 else args.render_mesh_mode
                        )
                    ),
                    show_bg=(len(env_files) > 0),
                    image_resolution=args.output_image_res,
                    rotate_y_to_z=args.rotate_y_to_z,
                    white_bg=args.render_white_bg,
                )

        if args.save_mesh_video:
            save_mesh_video(
                mesh_dir,
                args,
                blender_bin,
                get_mesh_video_env_path(batch, 0, args),
                fov=60,
                frame=120,
                init_cam=init_cam_arr[0, :] if args.relative_cam_pose else None,
                output_dir=model_dir,
            )


def get_data_model_dir(batch, batch_id):
    rgb_name = batch["rgb_names_input"][0][batch_id]
    return os.path.dirname(os.path.dirname(rgb_name))


def get_mesh_env_files(batch, batch_id):
    env_folder = os.path.join(get_data_model_dir(batch, batch_id), "env")
    if not pathmgr.isdir(env_folder):
        return []
    return sorted(
        [
            os.path.join(env_folder, f)
            for f in pathmgr.ls(env_folder)
            if f.endswith(".exr") or f.endswith(".hdr")
        ]
    )


def get_mesh_video_env_path(batch, batch_id, args):
    if args.env_video_path is not None:
        return args.env_video_path
    env_files = get_mesh_env_files(batch, batch_id)
    if len(env_files) > 0:
        return env_files[0]
    return None


def save_mesh_video(
    mesh_dir,
    args,
    blender_bin,
    envs,
    fov=60,
    frame=120,
    init_cam=None,
    output_dir=None,
):
    """Save a mesh rendering video with rotating camera."""
    cams_output, _, _, _ = utils.create_video_cameras(
        args.bbox_radius,
        frame,
        args.output_image_res,
        fov=fov,
        init_cam=init_cam,
    )
    cams_output = cams_output[0, :, :]
    eva_output_views = list(range(frame))

    video_dir = os.path.join(output_dir, args.env_video_output_path)
    mkdirs(video_dir)

    if envs is not None:
        utils.render_images_from_mesh(
            mesh_dir,
            args.prediction_type,
            blender_bin,
            cams_output,
            eva_output_views,
            video_dir,
            env_path=envs,
            env_mean=args.env_video_mean,
            mode=args.render_mesh_mode,
            save_video=True,
            show_bg=True,
            image_resolution=args.output_image_res,
            rotate_y_to_z=args.rotate_y_to_z,
        )
    else:
        utils.render_images_from_mesh(
            mesh_dir,
            args.prediction_type,
            blender_bin,
            cams_output,
            eva_output_views,
            video_dir,
            mode="texture",
            save_video=True,
            image_resolution=args.output_image_res,
            rotate_y_to_z=args.rotate_y_to_z,
        )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ.keys():
        local_rank = 0
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
    ngpus_per_node = torch.cuda.device_count()

    test_mesh_rendering(local_rank, ngpus_per_node, args)
