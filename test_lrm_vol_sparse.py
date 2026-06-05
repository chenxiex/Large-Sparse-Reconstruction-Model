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
import torch.nn.functional as F
import yaml

import misc.dist_helper as dist_helper
import misc.utils as utils
from data_loader.dtc_dataset import DtcDataset
from data_loader.real_dataset import RealDataset
from geometry.volsdf_sparse import VolSdf
from infer_lrm_vol import create_vol_infer
from loss.utils import scale_ls
from loss.volloss_sparse import VolLoss
from misc.env_utils import fix_random_seeds, init_distributed_mode
from misc.io_helper import mkdirs, pathmgr
from misc.sparse_utils import (
    build_3D_aware_attn,
    sparsify_imfeat,
    sparsify_volfeat,
)
from models_sparse.multiview_encoder import (
    DinoV2Encoder,
    DinoV3Encoder,
    mvencoder_base,
)
from models_sparse.voldecoder import VolTransformer
from renderer.sdf_renderer_sparse import SdfRenderer


def create_output_dir(args):
    args.user = os.environ["USER"] or "default"
    args.output_dir = os.path.join(args.exp_root, args.exp_name)

    mkdirs(args.output_dir)
    return args


def get_args_parser():
    parser = argparse.ArgumentParser("LRM Sparse Volume", add_help=False)
    # Sparse setting
    parser.add_argument(
        "--block_size",
        default=8,
        type=int,
        help="the size of compression, window and selection block",
    )
    parser.add_argument(
        "--topk_volume",
        default=8,
        type=int,
        help="the number of selected blocks for volume",
    )
    parser.add_argument(
        "--topk_multiimage",
        default=2,
        type=int,
        help="the number of selected blocks from volume to view",
    )
    parser.add_argument(
        "--max_volume_token_num",
        default=None,
        type=int,
        help="the maximum number of tokens in the volume",
    )
    parser.add_argument(
        "--max_image_token_num",
        default=None,
        type=int,
        help="the maximum number of image tokens",
    )

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
        "--volume_res", default=64, type=int, help="volume token resolution"
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
        "--use_3d_aware_attn",
        action="store_true",
        help="whether to use attention map computed from 3D info",
    )
    parser.add_argument(
        "--attn_winsize",
        type=int,
        default=2,
        help="the window size for 2D image when 3D aware attention is used",
    )
    parser.add_argument(
        "--volume_dilation_radius",
        type=int,
        default=0,
        help="the dilation radius for volume",
    )
    parser.add_argument(
        "--use_decomposed_embed",
        action="store_true",
        help="whether to use decomposed embed to save memory",
    )
    parser.add_argument(
        "--fuse_all_layers",
        action="store_true",
        help="whether to fuse all layers or not",
    )

    parser.add_argument(
        "--input_image_res",
        default=[256, 512],
        type=int,
        nargs=2,
        help="input image resolution",
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
        "--dense_pretrained_folder",
        default=None,
        required=True,
        help="path to the dense pretrained folder",
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
        "--ocgrid_acc",
        action="store_true",
        help="whether to use occupancy grid to accelerate rendering",
    )
    parser.add_argument(
        "--ocgrid_dilation_radius",
        type=int,
        default=0,
        help="dilation radius (in voxels) to grow the occupancy grid during rendering, "
        "avoiding missing surface details. e.g. 1 or 2. Only used with --ocgrid_acc.",
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
    )
    voldecoder = VolTransformer(
        vol_res=args.volume_res,
        output_dim=args.volume_dim,
        upsample_scale=args.volume_upsample_scale,
        embed_dim=args.embed_dim,
        depth=args.transformer_depth,
        cp_freq=0,
        use_weight_norm=args.use_weight_norm,
        topk_volume=args.topk_volume,
        topk_multiimage=args.topk_multiimage,
        num_q_heads=32,
        num_kv_heads=2,
        use_decomposed_embed=args.use_decomposed_embed,
        fuse_all_layers=args.fuse_all_layers,
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
        occgrid_res=(512 if args.ocgrid_acc else 1),
        auto_cast_dtype=torch.bfloat16,
    )
    volinfer = create_vol_infer(
        args.dense_pretrained_folder,
        mesh_resolution=args.volume_res * 4,
        gpu=args.gpu,
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
            input_image_res=args.input_image_res[-1],
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

    voldecoder = dist_helper.get_fsdp_model(voldecoder, args.gpu)

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        args.checkpoint,
        run_variables=to_restore,
        mvencoder=mvencoder,
        voldecoder=voldecoder,
        volsdf=volsdf,
        load_weights_only=False,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting LRM inference !")
    data_loader.sampler.set_epoch(start_epoch)
    test_one_epoch(
        volinfer,
        dinoencoder,
        mvencoder,
        voldecoder,
        volsdf,
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
    volinfer,
    dinoencoder,
    mvencoder,
    voldecoder,
    volsdf,
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
            volinfer,
            dinoencoder,
            mvencoder,
            voldecoder,
            volsdf,
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
                volinfer,
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
        assert batch_size == 1

        print("Model Id: %s" % model_ids[0])
        if "fov" in batch:
            args.fov = batch["fov"][0].item()
        if "eva_output_views" in batch:
            eva_output_views = batch["eva_output_views"][0, :]
            eva_output_views = eva_output_views.numpy().tolist()

        model_id = model_ids[0]
        model_dir = os.path.join(args.output_dir, model_id)
        mkdirs(model_dir, is_main_process_only=False)

        eval_dict, pred_keys = save_imgs_one_sample(
            preds, 0, eva_output_views, model_dir
        )

        # Save input images and input rendering
        input_image_out = os.path.join(model_dir, "inputs_gt.png")
        inputs = batch["rgb_input"].detach().cpu()
        utils.save_image(inputs, input_image_out)
        input_pred_mask_out = os.path.join(model_dir, "inputs_mask.png")
        masks = batch["mask_input"][0:1, :]
        utils.save_image(masks, input_pred_mask_out)

        if args.output_eval_json:
            for key in pred_keys:
                if key == "rgb":
                    if args.resave_gt_images and "rgb_output" in batch:
                        name_template = os.path.join(model_dir, "%03d_" + "gt_rgb.png")
                        gt_rgbs = batch["rgb_output"][0, :]
                        utils.save_single_png(gt_rgbs, name_template, eva_output_views)
                        names = [name_template % n for n in eva_output_views]
                        for n in range(0, len(names)):
                            eval_dict[n]["gt_image_path"] = names[n]
                    elif "rgb_names_output" in batch:
                        names = batch["rgb_names_output"]
                        for n in range(0, len(names)):
                            eval_dict[n]["gt_image_path"] = names[n][0]
                elif key == "albedo":
                    if args.resave_gt_images and "albedo_output" in batch:
                        name_template = os.path.join(
                            model_dir, "%03d_" + "gt_albedo.png"
                        )
                        gt_albedos = batch["albedo_output"][0, :]
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
                            eval_dict[n]["gt_albedo_path"] = names[n][0]
                elif key == "roughness":
                    if args.resave_gt_images and "roughness_output" in batch:
                        name_template = os.path.join(
                            model_dir, "%03d_" + "gt_roughness.png"
                        )
                        gt_roughnesss = batch["roughness_output"][0, :]
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
                            eval_dict[n]["gt_roughness_path"] = names[n][0]
                elif key == "normal":
                    if args.resave_gt_images and "normal_output" in batch:
                        name_template = os.path.join(
                            model_dir, "%03d_" + "gt_normal.png"
                        )
                        gt_normals = batch["normal_output"][0, :]
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
                            eval_dict[n]["gt_normal_path"] = names[n][0]
                elif key == "metallic":
                    if args.resave_gt_images and "metallic_output" in batch:
                        name_template = os.path.join(
                            model_dir, "%03d_" + "gt_metallic.png"
                        )
                        gt_metallics = batch["metallic_output"][0, :]
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
                            eval_dict[n]["gt_metallic_path"] = names[n][0]
            eval_dict_array.append(eval_dict)

        if args.save_mesh:
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                    # Update occupancy grid before mesh extraction to filter floaters
                    if args.ocgrid_acc:
                        renderer.update_occupancy_grid(
                            volume["volume"]["dense_feat"][0:1, :],
                            volinfer.volsdf,
                            volinfer.sdf_inv_std,
                        )
                    mesh_out = renderer.sample_point(
                        volume["volume"],
                        volsdf,
                        volinfer.volsdf,
                        mode="grid",
                        N=args.mesh_resolution,
                        ocgrid_acc=args.ocgrid_acc,
                    )
                    # Reset occupancy grid after mesh extraction if it was used
                    if args.ocgrid_acc:
                        renderer.reset_occupancy_grid()
                utils.save_mesh(
                    path=os.path.join(model_dir, "mesh.obj"),
                    values=mesh_out["sdf"][0, :].float(),
                    N=args.mesh_resolution,
                    threshold=0,
                    radius=(args.bbox_radius * 1.05),
                    init_cam_rot=init_cam_arr[0, :] if args.relative_cam_pose else None,
                )

                # Smooth the mesh and output textures
                assert args.blender_bin is not None, "--blender_bin is required"
                blender_bin = args.blender_bin
                mesh_dir = os.path.join(model_dir, "mesh")
                mkdirs(mesh_dir)

                if args.save_texture:
                    utils.compute_mesh_textures_sparse(
                        path=os.path.join(model_dir, "mesh.obj"),
                        blender_bin=blender_bin,
                        volume=volume["volume"],
                        volsdf=volsdf,
                        volsdf_dense=volinfer.volsdf,
                        output_path=os.path.join(mesh_dir, "mesh_uv.obj"),
                        output_texture_res=args.output_texture_res,
                        prediction_type=args.prediction_type,
                        save_texture=args.save_texture,
                        init_cam_rot=init_cam_arr[0, :]
                        if args.relative_cam_pose
                        else None,
                    )

                    if args.render_mesh:
                        env_files = get_mesh_env_files(batch, 0)

                        render_out_dir = os.path.join(
                            mesh_dir,
                            "mesh_rendering",
                        )
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

                        if len(env_files) > 0 and len(env_files) == len(
                            eva_output_views
                        ):
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
                        else:
                            utils.render_images_from_mesh(
                                mesh_dir,
                                args.prediction_type,
                                blender_bin,
                                cameras_output_np,
                                eva_output_views,
                                render_out_dir,
                                env_path=env_files[0] if len(env_files) > 0 else None,
                                mode="texture"
                                if len(env_files) == 0
                                else args.render_mesh_mode,
                                show_bg=(len(env_files) > 0),
                                image_resolution=args.output_image_res,
                            )

                    if args.save_mesh_video:
                        save_mesh_video(
                            mesh_dir,
                            args,
                            blender_bin,
                            get_mesh_video_env_path(batch, 0, args),
                            fov=60,
                            frame=120,
                            init_cam=init_cam_arr[0, :]
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


def save_video(
    volinfer,
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
    assert batch_size == 1

    model_id = model_ids[0]
    video_dir = os.path.join(args.output_dir, model_id, "video")
    mkdirs(video_dir, is_main_process_only=False)

    inv_std = volume["inv_std"]

    if init_cam_arr is not None:
        init_cam = init_cam_arr[0, :]
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
        volinfer.volsdf,
        renderer,
        volume["volume"],
        inv_std,
        volinfer.sdf_inv_std,
        rays_o,
        rays_d,
        cams_output,
        compute_normal=False,
        use_occ_grid=args.ocgrid_acc,
        ocgrid_dilation_radius=args.ocgrid_dilation_radius,
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
            show_bg=True,
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


def extract_inputs(batch, image_num):
    batch_input = {}
    batch_input["rgb_input"] = batch["rgb_input"][:, :image_num, :]
    batch_input["mask_input"] = batch["mask_input"][:, :image_num, :]
    batch_input["rays_o_input"] = batch["rays_o_input"][:, :image_num, :]
    batch_input["rays_d_input"] = batch["rays_d_input"][:, :image_num, :]
    batch_input["uv_input"] = batch["uv_input"][:, :image_num, :]
    if "depth_input" in batch:
        batch_input["depth_input"] = batch["depth_input"][:, :image_num, :, :, :]
    if "depth_masks_input" in batch:
        batch_input["depth_masks_input"] = batch["depth_masks_input"][
            :, :image_num, :, :, :
        ]
    if "surface_points_input" in batch:
        batch_input["surface_points_input"] = batch["surface_points_input"][
            :, :image_num, :, :, :
        ]
    if "bgs_input" in batch:
        batch_input["bgs_input"] = batch["bgs_input"][:, :image_num, :, :, :]
    batch_input["cameras_input"] = batch["cameras_input"][:, :image_num, :]
    if "crop_input" in batch:
        batch_input["crop_input"] = batch["crop_input"][:, :image_num, :]
    if "origin_size" in batch:
        batch_input["origin_size"] = batch["origin_size"]
    return batch_input


def slice_dense_points(dense_points, image_num):
    """Slice dense_points to only include the first image_num images."""
    if dense_points is None:
        return None
    sliced = {}
    for key, value in dense_points.items():
        if isinstance(value, torch.Tensor) and value.dim() >= 2:
            # Assume image dimension is at index 1
            sliced[key] = value[:, :image_num]
        else:
            sliced[key] = value
    return sliced


def downsample_im(image, new_size, channel_last=False):
    if channel_last:
        image = image.permute(0, 1, 4, 2, 3)
    batch_size, image_num, _, height, width = image.shape
    image = image.reshape(batch_size * image_num, -1, height, width)
    image = F.adaptive_avg_pool2d(image, (new_size, new_size))
    image = image.reshape(batch_size, image_num, -1, new_size, new_size)
    if channel_last:
        image = image.permute(0, 1, 3, 4, 2)
    return image


def downsample_inputs(batch, new_size):
    batch_input = {}
    batch_input["rgb_input"] = downsample_im(batch["rgb_input"], new_size, False)
    batch_input["mask_input"] = downsample_im(batch["mask_input"], new_size, False)
    batch_input["rays_o_input"] = downsample_im(batch["rays_o_input"], new_size, True)
    batch_input["rays_d_input"] = downsample_im(batch["rays_d_input"], new_size, True)
    batch_input["uv_input"] = downsample_im(batch["uv_input"], new_size, True)
    if "depth_input" in batch:
        batch_input["depth_input"] = downsample_im(
            batch["depth_input"], new_size, False
        )
    if "depth_masks_input" in batch:
        batch_input["depth_masks_input"] = downsample_im(
            batch["depth_masks_input"], new_size, False
        )
    if "surface_points_input" in batch:
        batch_input["surface_points_input"] = downsample_im(
            batch["surface_points_input"], new_size, True
        )
    if "bgs_input" in batch:
        batch_input["bgs_input"] = downsample_im(batch["bgs_input"], new_size, False)
    batch_input["cameras_input"] = batch["cameras_input"]
    return batch_input


def test_one_iteration(
    batch,
    volinfer,
    dinoencoder,
    mvencoder,
    voldecoder,
    volsdf,
    renderer,
    loss,
    loss_weights_dict,
    args,
    auto_cast_dtype=torch.bfloat16,
):
    batch_dense = downsample_inputs(batch, args.input_image_res[0])

    volfeat, imfeat, sdf_volume, alpha_volume, volume, dense_points = volinfer(
        batch_dense,
        args.gpu,
        auto_cast_dtype=auto_cast_dtype,
        output_res=args.input_image_res[1] // 2,
    )

    rays_o = batch["rays_o_output"]
    rays_d = batch["rays_d_output"]
    cams_output = batch["cameras_output"]

    images = batch["rgb_input"]
    batch_size, image_num, _, height, width = images.shape
    images = images.reshape(batch_size, image_num, -1, height, width)
    images = images.to(device=args.gpu, dtype=auto_cast_dtype)
    if args.encode_background != "none":
        bg = batch["bgs_input"].reshape(batch_size, image_num, 3, height, width)
        bg = bg.to(device=args.gpu, dtype=auto_cast_dtype)
    else:
        bg = None

    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
        with torch.no_grad():
            if args.use_dino != "no":
                feature = dinoencoder(images)
            else:
                feature = None

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
    mask_input = batch["mask_input"].to(device=args.gpu, dtype=auto_cast_dtype)

    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
        with torch.no_grad():
            (
                imfeat_sparse,
                imfeat_mask,
                imfeat_index,
                image_coords_idx,
                actual_im_num,
            ) = sparsify_imfeat(imfeat, mask_input, args)

            # If images were removed to fit token limit, slice batch and dense_points
            if actual_im_num < image_num:
                batch_sliced = extract_inputs(batch, actual_im_num)
                # Preserve output keys from original batch (they don't need slicing by image_num)
                for key in batch:
                    if key not in batch_sliced:
                        batch_sliced[key] = batch[key]
                batch = batch_sliced
                dense_points = slice_dense_points(dense_points, actual_im_num)
                images = images[:, :actual_im_num, :, :, :]
                if bg is not None:
                    bg = bg[:, :actual_im_num, :, :, :]
                uv_input = uv_input[:, :actual_im_num, :, :, :]
                mask_input = mask_input[:, :actual_im_num, :, :, :]
                if plucker_rays is not None:
                    plucker_rays = plucker_rays[:, :actual_im_num, :, :, :]
                if feature is not None:
                    # feature shape: (B, I*patches_per_image, C)
                    # Need to slice to keep only first actual_im_num images
                    patches_per_image = feature.shape[1] // image_num
                    feature = feature[:, : actual_im_num * patches_per_image, :]

            volfeat_sparse, volfeat_mask, volfeat_index, volume_coords_idx = (
                sparsify_volfeat(
                    volfeat,
                    sdf_volume,
                    batch,
                    dense_points,
                    alpha_volume,
                    args,
                    augment_volume=False,
                )
            )
        torch.cuda.empty_cache()

    attn_map = build_3D_aware_attn(
        imfeat_index,
        volfeat_index,
        batch,
        dense_points,
        args,
    )

    torch.cuda.synchronize()
    t_forward_start = time.time()
    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
        tokens = mvencoder(
            images,
            uv_input,
            imfeat_mask,
            imfeat_index,
            image_coords_idx,
            plucker_rays=plucker_rays,
            x_bg=bg,
            feature=feature,
        )
        volume, _ = voldecoder(
            tokens,
            imfeat_sparse,
            imfeat_index,
            volfeat_sparse,
            volfeat_mask,
            volfeat_index,
            volume,
            volume_coords_idx,
            attn_map,
            args.image_num_per_batch,
        )
    torch.cuda.synchronize()
    t_forward = time.time() - t_forward_start
    print("[Timing] forward pass: %.4fs" % t_forward)

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
        volinfer.volsdf,
        renderer,
        volume,
        args.sdf_inv_std,
        volinfer.sdf_inv_std,
        rays_o,
        rays_d,
        cams_output,
        compute_normal=(normal_weight_max > 0),
        compute_numerical_normal=(numerical_normal_weight_max > 0),
        use_occ_grid=args.ocgrid_acc,
        ocgrid_dilation_radius=args.ocgrid_dilation_radius,
    )
    if "albedo" in preds and "albedo_output" in batch:
        albedo_pred = preds["albedo"].permute(0, 1, 4, 2, 3)
        albedo_gt = batch["albedo_output"].cuda()
        mask_pred = preds["mask"].permute(0, 1, 4, 2, 3)
        mask_gt = batch["mask_output"].cuda()
        mask = mask_pred * mask_gt
        albedo_pred = 0.5 * (albedo_pred + 1)
        albedo_gt = 0.5 * (albedo_gt + 1)
        albedo_pred, _ = scale_ls(albedo_pred, albedo_gt, mask, mask)
        albedo_pred = 2 * albedo_pred.permute(0, 1, 3, 4, 2) - 1
        preds["albedo"] = albedo_pred

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
