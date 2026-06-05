# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import os
import tempfile
import time

import cv2
import ffmpeg

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import yaml

import misc.utils as utils
from data_loader.dtc_dataset import DtcDataset
from data_loader.real_dataset import RealDataset
from geometry.volsdf import VolSdf
from loss.volloss import VolLoss
from misc.env_utils import fix_random_seeds, init_distributed_mode
from misc.io_helper import mkdirs, pathmgr
from models.multiview_encoder import (
    DinoV2Encoder,
    DinoV3Encoder,
    mvencoder_base,
)
from models.pointscams_decoder import PointsCamsDptDecoder
from models.voldecoder import VolTransformer
from renderer.sdf_renderer import SdfRenderer


def create_output_dir(args):
    args.user = os.environ["USER"] or "default"
    args.output_dir = os.path.join(args.exp_root, args.exp_name)

    mkdirs(args.output_dir)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("LRM Dense Volume", add_help=False)

    parser.add_argument(
        "--exp_root",
        required=True,
        type=str,
        help="Path to save logs and checkpoints.",
    )
    parser.add_argument(
        "--exp_name",
        default="default_exp_name",
        type=str,
        help="Experiment name used as subdirectory under exp_root.",
    )

    # Model parameters
    parser.add_argument(
        "--transformer_depth",
        default=24,
        type=int,
        help="Depth of the volume transformer",
    )
    parser.add_argument(
        "--patch_size",
        default=8,
        type=int,
        help="patch_size of the encoder",
    )
    parser.add_argument(
        "--mlp_depth", default=3, type=int, help="Depth of the VolSDF Mlp"
    )
    parser.add_argument(
        "--mlp_brdf_depth",
        default=3,
        type=int,
        help="Depth of the VolSDF BRDF MLP branch",
    )
    parser.add_argument(
        "--mlp_geo_depth",
        default=2,
        type=int,
        help="Depth of the VolSDF geometry/SDF MLP branch",
    )
    parser.add_argument(
        "--mlp_dim", default=32, type=int, help="Number of channels of the VolSDF MLP"
    )
    parser.add_argument(
        "--embed_dim",
        default=1024,
        type=int,
        help="The hidden dimension of transformer",
    )
    parser.add_argument(
        "--volume_dim",
        default=32,
        type=int,
        help="Number of channels of the volume output",
    )
    parser.add_argument(
        "--volume_res", default=16, type=int, help="volume token resolution"
    )
    parser.add_argument(
        "--volume_upsample_scale",
        default=4,
        type=int,
        help="volume output upsample factor",
    )
    parser.add_argument(
        "--num_samples_per_ray",
        default=512,
        type=int,
        help="the number of samples per ray",
    )
    parser.add_argument(
        "--mvencoder_type",
        default="nocam",
        choices=["plucker", "nocam"],
        help="Mvencoder type",
    )
    parser.add_argument(
        "--use_dino",
        default="v3",
        choices=["v2", "v3", "no"],
        help="whether to use dino feature or not",
    )
    parser.add_argument(
        "--use_decomposed_embed",
        action="store_true",
        help="whether to use decomposed embed to save memory",
    )

    parser.add_argument(
        "--input_image_res", default=512, type=int, help="input image resolution"
    )
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
        "--dataset_type",
        default="dtc_dataset",
        choices=["real_dataset", "dtc_dataset"],
        type=str,
        help="The input dataset format",
    )
    parser.add_argument(
        "--dtc_white_envmap",
        action="store_true",
        help="whether to load data with white environment map",
    )
    parser.add_argument(
        "--prediction_type",
        default="rgb",
        choices=["rgb", "brdf", "both"],
        type=str,
        help="The output of LRM, either rgb or brdf",
    )
    parser.add_argument(
        "--encode_background",
        type=str,
        default="none",
        choices=["none", "augment"],
        help="whether to encode the background information",
    )

    # Dataset parameters
    parser.add_argument(
        "--relative_cam_pose",
        action="store_true",
        help="whether to use relative camera poses.",
    )
    parser.add_argument(
        "--max_dataset_size",
        default=-1,
        type=int,
        help="maximum number of models to be used for evaluation",
    )

    parser.add_argument(
        "--batch_size_per_gpu",
        default=1,
        type=int,
        help="Per-GPU batch-size: number of distinct 3D models loaded on one GPU.",
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
        "--sdf_inv_std",
        type=float,
        default=100,
        help="the SDF inverse standard deviation for inference (higher = sharper surfaces)",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="path to the checkpoint"
    )
    parser.add_argument(
        "--weights_dir",
        default="config",
        help="the folder that contains yaml files for weights",
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
    parser.add_argument(
        "--data_path",
        type=str,
        help="Please specify path to the evaluation/test data.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--save_mesh",
        action="store_true",
        help="save mesh",
    )
    parser.add_argument(
        "--save_texture",
        action="store_true",
        help="save texture",
    )
    parser.add_argument(
        "--output_texture_res",
        type=int,
        default=1024,
        help="output texture resolution",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="save video",
    )
    parser.add_argument(
        "--mesh_resolution",
        type=int,
        default=256,
        help="the density/sdf grid resolution when outputting mesh",
    )
    parser.add_argument(
        "--blender_bin",
        type=str,
        default=None,
        help="Path to the Blender binary. Required when not on FB infra.",
    )
    parser.add_argument(
        "--output_eval_json",
        action="store_true",
        help="whether to save the evaluation json file.",
    )
    parser.add_argument(
        "--centralized_cropping",
        action="store_true",
        help="crop the center of the image.",
    )
    parser.add_argument(
        "--loss_weights_file",
        default=None,
        type=str,
        help="the weights for losses.",
    )
    parser.add_argument(
        "--use_weight_norm",
        action="store_true",
        help="whether to use weight normalization or not",
    )
    parser.add_argument(
        "--start_id",
        type=int,
        default=0,
        help="the starting model id of the data list.",
    )
    parser.add_argument(
        "--end_id", type=int, default=-1, help="the ending model id of the data list."
    )
    parser.add_argument(
        "--resave_gt_images",
        action="store_true",
        help="whether to resave gt images or not",
    )
    parser.add_argument(
        "--render_mesh",
        action="store_true",
        help="whether to render the textured mesh with environment maps",
    )
    parser.add_argument(
        "--render_mesh_mode",
        default="texture",
        choices=["texture", "relighting"],
        type=str,
        help="rendering mode: texture (show textured mesh) or relighting (PBR with env map)",
    )
    parser.add_argument(
        "--save_mesh_video",
        action="store_true",
        help="whether to save a mesh rendering video with rotating camera",
    )
    parser.add_argument(
        "--camera_center_coord",
        action="store_true",
        help="whether to use camera center coordinate system for mesh rendering",
    )
    parser.add_argument(
        "--render_white_bg",
        action="store_true",
        help="render mesh with white background instead of environment map background",
    )
    parser.add_argument(
        "--env_video_path",
        type=str,
        default=None,
        help="Path to a single environment map (.exr/.hdr) used for mesh video rendering. "
        "When set, overrides env_files so that one consistent env map is used.",
    )
    return parser


def test_lrm(local_rank, ngpus_per_node, args):
    init_distributed_mode(local_rank, ngpus_per_node, args)
    args = create_output_dir(args)
    fix_random_seeds(args.seed)

    print("*" * 80)
    print(
        f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for training"
    )
    print(args)
    print("*" * 80)

    if utils.is_main_process():
        with pathmgr.open(os.path.join(args.output_dir, "args.txt"), "w") as fOut:
            fOut.write(
                "\n".join(
                    "%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())
                )
            )

    loss_weights_file = os.path.join(
        os.path.dirname(__file__), "config", f"{args.loss_weights_file}.yaml"
    )

    if utils.is_main_process():
        pathmgr.copy_from_local(
            loss_weights_file,
            os.path.join(args.output_dir, "weights.yaml"),
            overwrite=True,
        )

    with pathmgr.open(loss_weights_file, "r") as fIn:
        loss_weights_dict = yaml.safe_load(fIn)

    normal_weight_max = max(
        loss_weights_dict["mse"]["normal"],
        loss_weights_dict["perceptual"]["normal"],
    )
    numerical_normal_weight_max = max(
        loss_weights_dict["mse"]["numerical_normal"],
        loss_weights_dict["perceptual"]["numerical_normal"],
    )

    if args.use_dino == "v2":
        dinoencoder = DinoV2Encoder(patch_size=args.patch_size)
    elif args.use_dino == "v3":
        dinoencoder = DinoV3Encoder(patch_size=args.patch_size)
    else:
        dinoencoder = None

    if args.use_dino == "v2":
        feature_dim = 1024
    elif args.use_dino == "v3":
        feature_dim = 1280
    else:
        feature_dim = None
    mvencoder = mvencoder_base(
        mvencoder_type=args.mvencoder_type,
        with_bg=(args.encode_background == "augment"),
        embed_dim=args.embed_dim,
        patch_size=args.patch_size,
        feature_dim=feature_dim,
        num_heads=32,
    )
    voldecoder = VolTransformer(
        vol_res=args.volume_res,
        output_dim=args.volume_dim,
        upsample_scale=args.volume_upsample_scale,
        embed_dim=args.embed_dim,
        depth=args.transformer_depth,
        cp_freq=0,
        use_weight_norm=args.use_weight_norm,
        num_q_heads=32,
        num_kv_heads=2,
        use_decomposed_embed=args.use_decomposed_embed,
    )
    pdecoder = PointsCamsDptDecoder(
        token_dim=args.embed_dim,
        patch_size=args.patch_size,
        radius=(args.bbox_radius * 1.05),
    )

    volsdf = VolSdf(
        dim=args.mlp_dim,
        input_dim=args.volume_dim,
        rgb_depth=args.mlp_depth,
        brdf_depth=args.mlp_brdf_depth,
        sdf_depth=args.mlp_geo_depth,
        radius=(args.bbox_radius * 1.05),
        prediction_type=args.prediction_type,
        compute_normal=(normal_weight_max > 0),
    )
    renderer = SdfRenderer(
        pred_mode=args.prediction_type,
        radius=(args.bbox_radius * 1.05),
        num_samples_per_ray=args.num_samples_per_ray,
        occgrid_res=512,
        auto_cast_dtype=torch.bfloat16,
    )

    loss = VolLoss()

    if args.eva_output_views is not None:
        args.output_image_num = min(len(args.eva_output_views), args.output_image_num)
        args.eva_output_views = args.eva_output_views[: args.output_image_num]
    if args.dataset_type == "dtc_dataset":
        dataset = DtcDataset(
            mode="TEST",
            root_dir=args.data_path,
            input_image_num=args.image_num_per_batch,
            input_image_res=args.input_image_res,
            output_image_num=args.output_image_num,
            output_image_res=args.output_image_res,
            eva_input_views=args.eva_input_views,
            eva_output_views=args.eva_output_views,
            radius=args.bbox_radius,
            load_normal=(max(normal_weight_max, numerical_normal_weight_max) > 0),
            load_depth=False,
            load_brdf=(
                args.prediction_type == "brdf" or args.prediction_type == "both"
            ),
            load_bg=(args.encode_background != "none"),
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
            input_image_res=args.input_image_res,
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

    if args.use_dino != "no":
        dinoencoder = dinoencoder.cuda()
    mvencoder = mvencoder.cuda().eval()
    for para in mvencoder.parameters():
        para.requires_grad = False
    voldecoder = voldecoder.cuda()
    for para in voldecoder.parameters():
        para.requires_grad = False
    volsdf = volsdf.cuda()
    for para in volsdf.parameters():
        para.requires_grad = False
    renderer = renderer.cuda().eval()
    pdecoder = pdecoder.cuda()
    for para in pdecoder.parameters():
        para.requires_grad = False

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        args.checkpoint,
        run_variables=to_restore,
        mvencoder=mvencoder,
        voldecoder=voldecoder,
        volsdf=volsdf,
        pdecoder=pdecoder,
        load_weights_only=False,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting LRM inference !")
    data_loader.sampler.set_epoch(start_epoch)
    test_one_epoch(
        dinoencoder,
        mvencoder,
        voldecoder,
        volsdf,
        pdecoder,
        renderer,
        loss,
        loss_weights_dict,
        data_loader,
        args.eva_output_views,
        args,
    )
    log_stats = {"epoch": start_epoch}
    if utils.is_main_process():
        with pathmgr.open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Testing time {}".format(total_time_str))

    return


def save_imgs_one_sample(preds, batch_id, im_ids, folder):
    eval_dict = [{"frame": n} for n in range(0, len(im_ids))]
    pred_keys = []
    for key in preds.keys():
        if key == "gradient":
            continue  # to be finished.
        pred = preds[key]
        if key == "rgb":
            name = os.path.join(folder, "%03d_rgb.png")
            utils.save_single_png(pred[batch_id, :].permute(0, 3, 1, 2), name, im_ids)

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_image_path"] = os.path.join(
                    folder, "%03d_rgb.png" % im_ids[n]
                )
            pred_keys.append("rgb")

        elif key == "albedo":
            name = os.path.join(folder, "%03d_albedo.png")
            utils.save_single_png(
                pred[batch_id, :].permute(0, 3, 1, 2),
                name,
                im_ids,
                is_gamma=False,
            )

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_albedo_path"] = os.path.join(
                    folder, "%03d_albedo.png" % im_ids[n]
                )
            pred_keys.append("albedo")

        elif key == "roughness":
            name = os.path.join(folder, "%03d_roughness.png")
            utils.save_single_png(
                pred[batch_id, :].permute(0, 3, 1, 2),
                name,
                im_ids,
                is_gamma=False,
            )

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_roughness_path"] = os.path.join(
                    folder, "%03d_roughness.png" % im_ids[n]
                )
            pred_keys.append("roughness")

        elif key == "normal":
            name = os.path.join(folder, "%03d_normal.exr")
            utils.save_single_exr(pred[batch_id, :].permute(0, 3, 1, 2), name, im_ids)

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_normal_path"] = os.path.join(
                    folder, "%03d_normal.exr" % im_ids[n]
                )
            pred_keys.append("normal")

        elif key == "metallic":
            name = os.path.join(folder, "%03d_metallic.png")
            utils.save_single_png(
                pred[batch_id, :].permute(0, 3, 1, 2),
                name,
                im_ids,
                is_gamma=False,
            )

            for n in range(0, len(im_ids)):
                eval_dict[n]["pred_metallic_path"] = os.path.join(
                    folder, "%03d_metallic.png" % im_ids[n]
                )
            pred_keys.append("metallic")

    return eval_dict, pred_keys


def test_one_epoch(
    dinoencoder,
    mvencoder,
    voldecoder,
    volsdf,
    pdecoder,
    renderer,
    loss,
    loss_weights_dict,
    data_loader,
    eva_output_views,
    args,
):
    if utils.is_main_process():
        volsdf_path = os.path.join(args.output_dir, "volsdf.pth")
        with pathmgr.open(volsdf_path, "wb") as fp:
            torch.save(volsdf.state_dict(), fp)

    if args.output_eval_json:
        eval_dict_array = []

    for _, batch in enumerate(data_loader):
        auto_cast_dtype = torch.bfloat16
        preds, volume = test_one_iteration(
            batch,
            dinoencoder,
            mvencoder,
            voldecoder,
            volsdf,
            pdecoder,
            renderer,
            loss,
            loss_weights_dict,
            args,
            auto_cast_dtype=auto_cast_dtype,
        )

        model_ids = batch["name"]
        if args.relative_cam_pose:
            init_cam_arr = batch["init_cam_rot"]
        else:
            init_cam_arr = None
        if args.save_video:
            save_video(
                volume,
                volsdf,
                renderer,
                loss,
                model_ids,
                args,
                fov=60,
                frame=120,
                init_cam_arr=init_cam_arr,
            )

        batch_size = len(model_ids)
        for b in range(0, batch_size):
            print("Model Id: %s" % model_ids[b])
            if "fov" in batch:
                args.fov = batch["fov"][b].item()
            if "eva_output_views" in batch:
                eva_output_views = batch["eva_output_views"][b, :]
                eva_output_views = eva_output_views.numpy().tolist()

            model_id = model_ids[b]
            model_dir = os.path.join(args.output_dir, model_id)
            mkdirs(model_dir, is_main_process_only=False)

            volume_path = os.path.join(model_dir, "volume.pth")
            volume_one_sample = {}
            volume_one_sample["volume"] = volume["volume"][b : b + 1, :]
            volume_one_sample["inv_std"] = volume["inv_std"]

            with pathmgr.open(volume_path, "wb") as fp:
                torch.save(volume_one_sample, fp)

            eval_dict, pred_keys = save_imgs_one_sample(
                preds, b, eva_output_views, model_dir
            )

            # Save input images and input rendering
            input_image_out = os.path.join(model_dir, "inputs_gt.png")
            inputs = batch["rgb_input"].detach().cpu()
            utils.save_image(inputs, input_image_out)
            input_pred_mask_out = os.path.join(model_dir, "inputs_mask.png")
            masks = batch["mask_input"][b : b + 1, :]
            utils.save_image(masks, input_pred_mask_out)

            if args.output_eval_json:
                for key in pred_keys:
                    if key == "rgb":
                        if args.resave_gt_images and "rgb_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_rgb.png"
                            )
                            gt_rgbs = batch["rgb_output"][b, :]
                            utils.save_single_png(
                                gt_rgbs, name_template, eva_output_views
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_image_path"] = names[n]
                        elif "rgb_names_output" in batch:
                            names = batch["rgb_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_image_path"] = names[n][b]
                    elif key == "albedo":
                        if args.resave_gt_images and "albedo_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_albedo.png"
                            )
                            gt_albedos = batch["albedo_output"][b, :]
                            utils.save_single_png(
                                gt_albedos,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_albedo_path"] = names[n]
                        elif "albedo_names_output" in batch:
                            names = batch["albedo_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_albedo_path"] = names[n][b]
                    elif key == "roughness":
                        if args.resave_gt_images and "roughness_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_roughness.png"
                            )
                            gt_roughnesss = batch["roughness_output"][b, :]
                            utils.save_single_png(
                                gt_roughnesss,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_roughness_path"] = names[n]
                        elif "roughness_names_output" in batch:
                            names = batch["roughness_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_roughness_path"] = names[n][b]
                    elif key == "normal":
                        if args.resave_gt_images and "normal_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_normal.png"
                            )
                            gt_normals = batch["normal_output"][b, :]
                            utils.save_single_png(
                                gt_normals,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_normal_path"] = names[n]
                        elif "normal_names_output" in batch:
                            names = batch["normal_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_normal_path"] = names[n][b]
                    elif key == "metallic":
                        if args.resave_gt_images and "metallic_output" in batch:
                            name_template = os.path.join(
                                model_dir, "%03d_" + "gt_metallic.png"
                            )
                            gt_metallics = batch["metallic_output"][b, :]
                            utils.save_single_png(
                                gt_metallics,
                                name_template,
                                eva_output_views,
                            )
                            names = [name_template % n for n in eva_output_views]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_metallic_path"] = names[n]
                        if "metallic_names_output" in batch:
                            names = batch["metallic_names_output"]
                            for n in range(0, len(names)):
                                eval_dict[n]["gt_metallic_path"] = names[n][b]
                eval_dict_array.append(eval_dict)

            if args.save_mesh:
                with torch.no_grad():
                    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                        mesh_out = renderer.sample_point(
                            volume["volume"][b : b + 1, :],
                            volsdf,
                            mode="grid",
                            N=args.mesh_resolution,
                        )
                    utils.save_mesh(
                        path=os.path.join(model_dir, "mesh.obj"),
                        values=mesh_out["sdf"][0, :].float(),
                        N=args.mesh_resolution,
                        threshold=0,
                        radius=(args.bbox_radius * 1.05),
                        init_cam_rot=init_cam_arr[b, :]
                        if args.relative_cam_pose
                        else None,
                    )

                    # Smooth the mesh and output textures
                    assert args.blender_bin is not None, "--blender_bin is required"
                    blender_bin = args.blender_bin
                    mesh_dir = os.path.join(model_dir, "mesh")
                    mkdirs(mesh_dir)

                    if args.save_texture:
                        utils.compute_mesh_textures(
                            path=os.path.join(model_dir, "mesh.obj"),
                            blender_bin=blender_bin,
                            volume=volume["volume"][b : b + 1, :],
                            volsdf=volsdf,
                            output_path=os.path.join(mesh_dir, "mesh_uv.obj"),
                            output_texture_res=args.output_texture_res,
                            prediction_type=args.prediction_type,
                            save_texture=args.save_texture,
                            init_cam_rot=init_cam_arr[b, :]
                            if args.relative_cam_pose
                            else None,
                        )

                        if args.render_mesh:
                            render_mesh_images(
                                mesh_dir,
                                batch,
                                b,
                                init_cam_arr,
                                eva_output_views,
                                blender_bin,
                                args,
                            )

                        if args.save_mesh_video:
                            save_mesh_video(
                                mesh_dir,
                                args,
                                blender_bin,
                                get_mesh_video_env_path(batch, b, args),
                                fov=60,
                                frame=120,
                                init_cam=init_cam_arr[b, :]
                                if args.relative_cam_pose
                                else None,
                            )

        if args.output_eval_json:
            with pathmgr.open(
                os.path.join(args.output_dir, "eval_%d.json" % args.gpu), "w"
            ) as fp:
                json.dump(eval_dict_array, fp, indent=4)

    # gather the stats from all processes
    return


def get_mesh_env_files(batch, batch_id):
    data_model_dir = get_data_model_dir(batch, batch_id)
    env_folder = os.path.join(data_model_dir, "env")
    if not pathmgr.isdir(env_folder):
        return []
    return sorted(
        [
            os.path.join(env_folder, f)
            for f in pathmgr.ls(env_folder)
            if f.endswith(".exr") or f.endswith(".hdr")
        ]
    )


def get_data_model_dir(batch, batch_id):
    rgb_name = batch["rgb_names_input"][0][batch_id]
    return os.path.dirname(os.path.dirname(rgb_name))


def get_mesh_video_env_path(batch, batch_id, args):
    if args.env_video_path is not None:
        return args.env_video_path
    env_files = get_mesh_env_files(batch, batch_id)
    if len(env_files) > 0:
        return env_files[0]
    return None


def get_mesh_render_cameras(batch, batch_id, init_cam_arr, args):
    cameras_output_np = batch["cameras_output"][batch_id, :].numpy()
    if not args.relative_cam_pose:
        return cameras_output_np

    cam_rot = init_cam_arr[batch_id, :].numpy()  # R_0 (3x3)
    rot_4x4 = np.eye(4, dtype=cameras_output_np.dtype)
    rot_4x4[:3, :3] = cam_rot
    n_cams = cameras_output_np.shape[0]
    exts = cameras_output_np[:, :16].reshape(n_cams, 4, 4)
    exts = rot_4x4 @ exts
    cameras_output_np[:, :16] = exts.reshape(n_cams, 16)
    return cameras_output_np


def render_mesh_images(
    mesh_dir,
    batch,
    batch_id,
    init_cam_arr,
    eva_output_views,
    blender_bin,
    args,
):
    env_files = get_mesh_env_files(batch, batch_id)
    render_out_dir = os.path.join(mesh_dir, "mesh_rendering")
    mkdirs(render_out_dir)
    cameras_output_np = get_mesh_render_cameras(batch, batch_id, init_cam_arr, args)

    if len(env_files) > 0 and len(env_files) == len(eva_output_views):
        utils.render_images_from_mesh_multienv(
            mesh_dir,
            args.prediction_type,
            blender_bin,
            cameras_output_np,
            eva_output_views,
            render_out_dir,
            env_paths=env_files,
            mode=args.render_mesh_mode,
            show_bg=not args.render_white_bg,
            camera_center_coord=args.camera_center_coord,
            image_resolution=args.output_image_res,
        )
        return

    utils.render_images_from_mesh(
        mesh_dir,
        args.prediction_type,
        blender_bin,
        cameras_output_np,
        eva_output_views,
        render_out_dir,
        env_path=env_files[0] if len(env_files) > 0 else None,
        mode="texture" if len(env_files) == 0 else args.render_mesh_mode,
        show_bg=(len(env_files) > 0 and not args.render_white_bg),
        image_resolution=args.output_image_res,
    )


def save_video(
    volume,
    volsdf,
    renderer,
    loss,
    model_ids,
    args,
    fov=60,
    frame=120,
    init_cam_arr=None,
):
    batch_size = len(model_ids)
    for b in range(0, batch_size):
        model_id = model_ids[b]
        video_dir = os.path.join(args.output_dir, model_id, "video")
        mkdirs(video_dir, is_main_process_only=False)

        volfeat = volume["volume"][b : b + 1, :]
        inv_std = volume["inv_std"]

        if init_cam_arr is not None:
            init_cam = init_cam_arr[b, :]
        else:
            init_cam = None
        cams_output, rays_o, rays_d, _ = utils.create_video_cameras(
            args.bbox_radius, frame, args.output_image_res, init_cam=init_cam
        )
        with pathmgr.open(os.path.join(video_dir, "extrinsic.npy"), "wb") as fOut:
            extr = cams_output[:, :, 0:16].reshape(-1, 4, 4)
            np.save(fOut, extr)

        with pathmgr.open(os.path.join(video_dir, "fov.txt"), "w") as fOut:
            fOut.write("%.3f\n" % fov)

        rays_o = torch.from_numpy(rays_o).to(device=args.gpu)
        rays_d = torch.from_numpy(rays_d).to(device=args.gpu)
        cams_output = torch.from_numpy(cams_output).to(device=args.gpu)

        preds = loss.forward(
            volsdf,
            renderer,
            volfeat,
            inv_std,
            rays_o,
            rays_d,
            cams_output,
            compute_normal=False,
        )
        im_ids = list(range(frame))
        name = os.path.join(video_dir, "%03d_rgb.png")
        utils.save_single_png(preds["rgb"][0, :].permute(0, 3, 1, 2), name, im_ids)

        with tempfile.TemporaryDirectory() as temp_dir:
            for n in range(0, frame):
                image = preds["rgb"][0, n, :, :, :]
                image = image.detach().to(torch.float32).cpu().numpy()
                image = np.clip(0.5 * (image + 1), 0, 1)
                image = (255 * image).astype(np.uint8)
                cv2.imwrite(os.path.join(temp_dir, "%03d.png" % n), image[:, :, ::-1])
            local_video_path = os.path.join(temp_dir, "video.mp4")
            video_path = os.path.join(video_dir, "video.mp4")
            ffmpeg.input(os.path.join(temp_dir, "%03d.png"), framerate=24).output(
                local_video_path,
                pix_fmt="yuv420p",
                vcodec="libx264",
                crf=18,
                preset="slow",
            ).run()
            pathmgr.copy_from_local(local_video_path, video_path, overwrite=True)

        return


def save_mesh_video(
    model_dir,
    args,
    blender_bin,
    envs,
    fov=60,
    frame=6,
    init_cam=None,
):
    cams_output, _, _, _ = utils.create_video_cameras(
        args.bbox_radius,
        frame,
        args.output_image_res,
        fov=fov,
        init_cam=init_cam,
    )
    cams_output = cams_output[0, :, :]
    eva_output_views = list(range(frame))

    video_dir = os.path.join(model_dir, "video")
    mkdirs(video_dir)

    if envs is not None:
        utils.render_images_from_mesh(
            model_dir,
            args.prediction_type,
            blender_bin,
            cams_output,
            eva_output_views,
            video_dir,
            env_path=envs,
            env_mean=0.5,
            mode=args.render_mesh_mode,
            save_video=True,
            show_bg=not args.render_white_bg,
            image_resolution=args.output_image_res,
        )
    else:
        utils.render_images_from_mesh(
            model_dir,
            args.prediction_type,
            blender_bin,
            cams_output,
            eva_output_views,
            video_dir,
            mode="texture",
            save_video=True,
            image_resolution=args.output_image_res,
        )
    return


def test_one_iteration(
    batch,
    dinoencoder,
    mvencoder,
    voldecoder,
    volsdf,
    pdecoder,
    renderer,
    loss,
    loss_weights_dict,
    args,
    auto_cast_dtype=torch.bfloat16,
):
    images = batch["rgb_input"]
    rays_o = batch["rays_o_output"]
    rays_d = batch["rays_d_output"]
    cams_output = batch["cameras_output"]

    batch_size, image_num, _, height, width = images.shape
    images = images.reshape(batch_size, image_num, -1, height, width)
    images = images.to(device=args.gpu, dtype=auto_cast_dtype)
    if args.encode_background != "none":
        bg = batch["bgs_input"].reshape(batch_size, image_num, 3, height, width)
        bg = bg.to(device=args.gpu, dtype=auto_cast_dtype)
    else:
        bg = None

    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
        if args.mvencoder_type == "plucker":
            rays_o_input = batch["rays_o_input"]
            rays_d_input = batch["rays_d_input"]

            rays_o_input = rays_o_input.permute(0, 1, 4, 2, 3)
            rays_d_input = rays_d_input.permute(0, 1, 4, 2, 3)

            plucker_rays = torch.cat(
                [rays_d_input, torch.cross(rays_o_input, rays_d_input, dim=2)], dim=2
            )
            plucker_rays = plucker_rays.to(device=args.gpu, dtype=auto_cast_dtype)
        else:
            plucker_rays = None

        uv_input = batch["uv_input"].to(device=args.gpu, dtype=auto_cast_dtype)
        uv_input = uv_input.permute(0, 1, 4, 2, 3)

        if args.use_dino != "no":
            feature = dinoencoder(images)
        else:
            feature = None

        tokens = mvencoder(
            images, uv_input, plucker_rays=plucker_rays, x_bg=bg, feature=feature
        )

        volume, y = voldecoder(
            tokens,
            image_num,
        )
        points = pdecoder(y, images)
        points = points.to(dtype=torch.float32)

    rays_o = rays_o.to(device=args.gpu)
    rays_d = rays_d.to(device=args.gpu)
    cams_output = cams_output.to(device=args.gpu)

    normal_weight_max = max(
        loss_weights_dict["mse"]["normal"], loss_weights_dict["perceptual"]["normal"]
    )
    numerical_normal_weight_max = max(
        loss_weights_dict["mse"]["numerical_normal"],
        loss_weights_dict["perceptual"]["numerical_normal"],
    )

    preds = loss.forward(
        volsdf,
        renderer,
        volume,
        args.sdf_inv_std,
        rays_o,
        rays_d,
        cams_output,
        compute_normal=(normal_weight_max > 0),
        compute_numerical_normal=(numerical_normal_weight_max > 0),
    )
    preds["surface_points"] = points
    volume = {
        "volume": volume,
        "inv_std": args.sdf_inv_std,
    }
    return preds, volume


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ.keys():
        local_rank = 0
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
    ngpus_per_node = torch.cuda.device_count()

    test_lrm(local_rank, ngpus_per_node, args)
