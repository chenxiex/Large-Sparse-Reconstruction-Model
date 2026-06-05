# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import datetime
import json
import os
import random
import time
from functools import partial

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.tensorboard import SummaryWriter

import misc.dist_helper as dist_helper
import misc.logging as logging
import misc.utils as utils
from data_loader.dtc_dataset import DtcDataset
from geometry.volsdf import VolSdf
from loss.perceptual_loss import PerceptualLoss
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


logger = logging.get_logger(__name__)


def _worker_init_fn(worker_id, seed, num_workers, rank):
    worker_seed = seed + worker_id + rank * num_workers
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**31))


def create_output_dir(args):
    args.user = os.environ["USER"] or "default"
    args.output_dir = os.path.join(args.exp_root, args.exp_name)
    args.checkpoint_dir = os.path.join(args.output_dir, "checkpoints")
    args.image_dir = os.path.join(args.output_dir, "images")
    args.logging_save_path = os.path.join(args.output_dir, "log_verbose_rank_0.txt")

    mkdirs(args.output_dir)
    mkdirs(args.checkpoint_dir)
    mkdirs(args.image_dir)

    # Setup logging format.
    logging.setup_logging(
        args.logging_save_path,
        mode="a",
        buffering=args.buffering,
    )
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
        "--output_image_res", default=128, type=int, help="output image resolution"
    )
    parser.add_argument(
        "--output_res_range",
        default=[256, 256],
        nargs=2,
        type=int,
        help="output image resize range",
    )

    parser.add_argument(
        "--bbox_radius",
        default=0.5,
        type=float,
        help="the size of the bounding box; 1.05 for sdf",
    )

    parser.add_argument(
        "--dtc_white_envmap",
        action="store_true",
        help="whether to load data with white environment map",
    )
    parser.add_argument(
        "--use_adobe_view_selection",
        action="store_true",
        help="whether to use adobe view selection for training",
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

    # Training/Optimization parameters
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.05,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=1.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=8,
        type=int,
        help="Per-GPU batch-size: number of distinct 3D models loaded on one GPU.",
    )
    parser.add_argument(
        "--image_num_per_batch_range",
        default=[4, 8],
        nargs=2,
        type=int,
        help="Range [min, max] of input image count per 3D model, randomly sampled each iteration.",
    )
    parser.add_argument(
        "--output_image_num",
        default=4,
        type=int,
        help="novel views used for supervision",
    )
    parser.add_argument(
        "--epochs", default=10000, type=int, help="Number of epochs of training."
    )
    parser.add_argument(
        "--lr",
        default=4e-4,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_iters",
        default=3000,
        type=int,
        help="Number of iterations for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--start_sdf_inv_std",
        type=float,
        default=10,
        help="the sdf std when the training starts",
    )
    parser.add_argument(
        "--warmup_sdf_inv_std",
        type=float,
        default=20,
        help="the sdf std after the warmup steps.",
    )
    parser.add_argument(
        "--end_sdf_inv_std",
        type=float,
        default=200,
        help="the sdf std after training ends",
    )
    parser.add_argument(
        "--gradient_checkpointing_freq",
        default=1,
        type=int,
        help="Frequency of block for gradient checkpointing",
    )
    parser.add_argument(
        "--restart_from_checkpoint",
        action="store_true",
        help="restart from checkpoints",
    )
    parser.add_argument(
        "--checkpoint", default=None, type=str, help="path to the checkpoint"
    )
    parser.add_argument(
        "--weights_dir",
        default="config",
        help="the folder that contains yaml files for weights",
    )
    parser.add_argument(
        "--ocgrid_acc",
        action="store_true",
        help="whether to use occupancy grid to accelerate rendering",
    )
    parser.add_argument(
        "--filter_normal",
        action="store_true",
        help="whether to filter normal when computing normal loss",
    )
    parser.add_argument(
        "--mesh_resolution",
        type=int,
        default=256,
        help="the resolution when output mesh",
    )

    # Misc
    parser.add_argument(
        "--data_path",
        type=str,
        help="Please specify path to the training data.",
    )
    parser.add_argument(
        "--sep_gt_dir",
        default=None,
        type=str,
        help="Please specify path to the separate ground truth data directory.",
    )
    parser.add_argument(
        "--saveckp_epoch_freq",
        default=5,
        type=int,
        help="Save checkpoint every x epochs.",
    )
    parser.add_argument(
        "--backup_ckp_epoch_freq",
        default=-1,
        type=int,
        help="backup checkpoint every x epochs.",
    )
    parser.add_argument(
        "--saveckp_iter_freq",
        default=500,
        type=int,
        help="Save checkpoint every x iters.",
    )
    parser.add_argument(
        "--saveimg_iter_freq",
        default=1000,
        type=int,
        help="Save images every x iterations.",
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers",
        default=10,
        type=int,
        help="Number of data loading workers per GPU.",
    )
    parser.add_argument(
        "--load_weights_only",
        action="store_true",
        help="only load weights",
    )
    parser.add_argument(
        "--buffering",
        default=1024,
        type=int,
        help="logging buffer size",
    )
    parser.add_argument(
        "--centralized_cropping",
        action="store_true",
        help="crop the center of the image.",
    )
    parser.add_argument(
        "--perturb_color",
        action="store_true",
        help="whether to perturb the image color.",
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
    return parser


def train_lrm(local_rank, ngpus_per_node, args):
    init_distributed_mode(local_rank, ngpus_per_node, args)
    args = create_output_dir(args)
    fix_random_seeds(args.seed)

    logger.info("*" * 80)
    logger.info(
        f"GPU: {args.gpu}, Local rank: {args.global_rank}/{args.world_size} for training"
    )
    logger.info(args)
    logger.info("*" * 80)

    tb_writer = (
        SummaryWriter(log_dir=args.output_dir) if utils.is_main_process() else None
    )

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
        cp_freq=args.gradient_checkpointing_freq,
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
        occgrid_res=(256 if args.ocgrid_acc else 1),
        auto_cast_dtype=torch.bfloat16,
    )

    loss = VolLoss(
        h_chunk=args.output_image_res // 128,
        filter_normal=args.filter_normal,
        filter_normal_threshold=args.bbox_radius / 20.0,
    )

    mse_loss_func = F.mse_loss
    perceptual_loss_func = PerceptualLoss(loss_type="zhang")
    for param in perceptual_loss_func.parameters():
        param.requires_grad = False
    loss_func_dict = {}
    loss_func_dict["mse"] = mse_loss_func
    loss_func_dict["perceptual"] = perceptual_loss_func

    if args.use_dino != "no":
        dinoencoder = dinoencoder.cuda()
        dinoencoder = dist_helper.get_fsdp_model_frozen(dinoencoder, args.gpu)
    mvencoder = mvencoder.cuda()
    voldecoder = voldecoder.cuda()
    volsdf = volsdf.cuda()
    perceptual_loss_func = perceptual_loss_func.cuda()
    renderer = renderer.cuda()
    pdecoder = pdecoder.cuda()

    dataset = DtcDataset(
        root_dir=args.data_path,
        sep_gt_dir=args.sep_gt_dir,
        input_image_num=args.image_num_per_batch_range,
        input_image_res=args.input_image_res,
        output_image_num=args.output_image_num,
        output_image_res=args.output_image_res,
        output_res_range=args.output_res_range,
        radius=args.bbox_radius,
        load_normal=(max(normal_weight_max, numerical_normal_weight_max) > 0),
        load_depth=True,
        load_brdf=(args.prediction_type == "brdf" or args.prediction_type == "both"),
        load_bg=(args.encode_background != "none"),
        relative_cam_pose=args.relative_cam_pose,
        centralized_cropping=args.centralized_cropping,
        use_adobe_view_selection=args.use_adobe_view_selection,
        perturb_color=args.perturb_color,
        white_env=args.dtc_white_envmap,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True, seed=args.seed)
    data_loader = utils.RepeatedDataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        worker_init_fn=partial(
            _worker_init_fn,
            seed=args.seed,
            num_workers=args.num_workers,
            rank=dist_helper.get_rank(),
        ),
    )
    logger.info(f"Data loaded: there are {len(dataset)} 3D models.")

    mvencoder = dist_helper.get_parallel_model(mvencoder, args.gpu)
    voldecoder = dist_helper.get_fsdp_model(voldecoder, args.gpu)
    volsdf = dist_helper.get_parallel_model(volsdf, args.gpu)
    pdecoder = dist_helper.get_parallel_model(pdecoder, args.gpu)

    # Separate parameter groups for FSDP models (voldecoder) and DDP models (volsdf + pdecoder)
    fsdp_params_groups = utils.get_params_groups(
        args,
        voldecoder=voldecoder,
    )
    ddp_params_groups = utils.get_params_groups(
        args,
        mvencoder=mvencoder,
        volsdf=volsdf,
        pdecoder=pdecoder,
    )
    optimizer_fsdp = torch.optim.AdamW(
        fsdp_params_groups, betas=(0.9, 0.95), foreach=True
    )
    optimizer_ddp = torch.optim.AdamW(
        ddp_params_groups, betas=(0.9, 0.95), foreach=True
    )

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_iters=args.warmup_iters,
    )
    std_scheduler = utils.linear_scheduler(
        args.warmup_sdf_inv_std,
        args.end_sdf_inv_std,
        args.epochs,
        len(data_loader),
        warmup_iters=args.warmup_iters,
        start_warmup_value=args.start_sdf_inv_std,
    )

    logger.info("Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, "start_it": -1}
    last_ckpt = os.path.join(args.checkpoint_dir, "last.pth")
    if pathmgr.isfile(last_ckpt):
        args.checkpoint = last_ckpt
        args.load_weights_only = False
        utils.restart_from_checkpoint(
            args.checkpoint,
            run_variables=to_restore,
            mvencoder=mvencoder,
            voldecoder=voldecoder,
            volsdf=volsdf,
            pdecoder=pdecoder,
            optimizer_fsdp=optimizer_fsdp,
            optimizer_ddp=optimizer_ddp,
            load_weights_only=args.load_weights_only,
        )
    elif args.restart_from_checkpoint and args.checkpoint is not None:
        utils.restart_from_checkpoint(
            args.checkpoint,
            run_variables=to_restore,
            mvencoder=mvencoder,
            voldecoder=voldecoder,
            volsdf=volsdf,
            pdecoder=pdecoder,
            optimizer_ddp=optimizer_ddp,
            optimizer_fsdp=optimizer_fsdp,
            load_weights_only=args.load_weights_only,
        )
    start_epoch = to_restore["epoch"]
    start_it = to_restore["start_it"]

    start_time = time.time()
    logger.info("Starting LRM training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            dinoencoder,
            mvencoder,
            voldecoder,
            volsdf,
            pdecoder,
            renderer,
            loss,
            loss_func_dict,
            loss_weights_dict,
            data_loader,
            optimizer_fsdp,
            optimizer_ddp,
            lr_schedule,
            std_scheduler,
            epoch,
            args,
            start_iter=start_it,
            tb_writer=tb_writer,
        )
        start_it = -1

        # ============ writing logs ... ============
        # Use FSDP-specific state dict functions for FSDP models
        save_dict = {
            "mvencoder": mvencoder.state_dict(),
            "voldecoder": utils.get_fsdp_full_state_dict(voldecoder),
            "volsdf": volsdf.state_dict(),
            "pdecoder": pdecoder.state_dict(),
            "optimizer_fsdp": utils.get_fsdp_full_optim_state_dict(
                voldecoder, optimizer_fsdp
            ),
            "optimizer_ddp": optimizer_ddp.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }

        if (epoch == args.epochs - 1) or (epoch + 1) % args.saveckp_epoch_freq == 0:
            backup_ckp_epoch = (
                epoch
                if args.backup_ckp_epoch_freq > 0
                and (epoch + 1) % args.backup_ckp_epoch_freq == 0
                else -1
            )
            utils.save_on_master(
                save_dict,
                os.path.join(args.checkpoint_dir, "last.pth"),
                backup_ckp_epoch=backup_ckp_epoch,
            )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            "epoch": epoch,
        }
        if utils.is_main_process():
            with pathmgr.open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info("Training time {}".format(total_time_str))

    return


def train_one_epoch(
    dinoencoder,
    mvencoder,
    voldecoder,
    volsdf,
    pdecoder,
    renderer,
    loss,
    loss_func_dict,
    loss_weights_dict,
    data_loader,
    optimizer_fsdp,
    optimizer_ddp,
    lr_schedule,
    std_scheduler,
    epoch,
    args,
    start_iter=None,
    tb_writer=None,
):
    metric_logger = utils.MetricLogger(
        delimiter="  ",
        logger=logger,
        tb_writer=tb_writer,
        epoch=epoch,
    )
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 5, header)):
        # update weight decay and learning rate according to their schedule
        abs_it = it
        it = len(data_loader) * epoch + it  # global training iteration
        for _, param_group in enumerate(optimizer_fsdp.param_groups):
            param_group["lr"] = lr_schedule[it]
        for _, param_group in enumerate(optimizer_ddp.param_groups):
            param_group["lr"] = lr_schedule[it]
        inv_std = std_scheduler[it]

        auto_cast_dtype = torch.bfloat16
        preds, gts, volume, losses = train_one_iteration(
            batch,
            dinoencoder,
            mvencoder,
            voldecoder,
            volsdf,
            inv_std,
            pdecoder,
            renderer,
            loss,
            loss_func_dict,
            loss_weights_dict,
            args,
            auto_cast_dtype=auto_cast_dtype,
        )

        if args.clip_grad:
            utils.clip_gradients(mvencoder, args.clip_grad)
            utils.clip_gradients(voldecoder, args.clip_grad)
            utils.clip_gradients(volsdf, args.clip_grad)
            utils.clip_gradients(pdecoder, args.clip_grad)

        optimizer_fsdp.step()
        optimizer_ddp.step()

        optimizer_fsdp.zero_grad(set_to_none=True)
        optimizer_ddp.zero_grad(set_to_none=True)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(**losses)
        metric_logger.update(lr=lr_schedule[it])
        metric_logger.update(inv_std=inv_std)

        if it % args.saveimg_iter_freq == 0 or it == 0 or abs_it == 0:
            if utils.is_main_process():
                input_image_list = os.path.join(args.image_dir, f"{it:06d}_inputs.txt")
                utils.save_image_list(batch["rgb_names_input"], input_image_list)

                input_image_out = os.path.join(args.image_dir, f"{it:06d}_inputs.png")
                inputs = batch["rgb_input"].detach().cpu()
                utils.save_image(inputs, input_image_out)
                if args.encode_background != "none":
                    bgs_image_out = os.path.join(args.image_dir, f"{it:06d}_bgs.png")
                    bgs = batch["bgs_input"].detach().cpu()
                    utils.save_image(bgs, bgs_image_out)

                if "depth" in gts:
                    depth_mask = batch["depth_masks_output"].detach().cpu()
                    depth_gt = batch["depth_output"].detach().cpu()
                    depth_gt = depth_gt.reshape(-1)[depth_mask.reshape(-1) > 0]
                    if depth_gt.numel() != 0:
                        depth_min = depth_gt.min().item()
                        depth_max = depth_gt.max().item()
                    else:
                        depth_min = 0
                        depth_max = 0
                else:
                    depth_mask = None
                    depth_min = None
                    depth_max = None

                if "depth_input" in batch:
                    utils.save_depth(
                        batch["depth_input"].detach().cpu(),
                        batch["depth_masks_input"].detach().cpu(),
                        os.path.join(args.image_dir, f"{it:06d}_depth_inputs.png"),
                        depth_min,
                        depth_max,
                    )

                for key in gts.keys():
                    gt = gts[key].detach().cpu()
                    gts_image_out = os.path.join(
                        args.image_dir, f"{it:06d}_gts_{key}.png"
                    )
                    if key == "depth":
                        utils.save_depth(
                            gt,
                            depth_mask,
                            gts_image_out,
                            depth_min,
                            depth_max,
                        )
                    else:
                        utils.save_image(gt, gts_image_out, is_gamma=False)

                for key in preds.keys():
                    if (
                        key == "gradient"
                        or key == "points"
                        or key == "surface_points"
                        or "camera" in key
                    ):
                        continue  # to be finished.
                    pred = preds[key]
                    if len(pred.shape) == 5:
                        pred = pred.permute(0, 1, 4, 2, 3).detach().cpu()
                    preds_image_out = os.path.join(
                        args.image_dir, f"{it:06d}_preds_{key}.png"
                    )
                    if key == "depth" and depth_mask is not None:
                        utils.save_depth(
                            pred,
                            depth_mask,
                            preds_image_out,
                            depth_min,
                            depth_max,
                        )
                    elif key == "surface_depth":
                        utils.save_depth(
                            pred,
                            batch["depth_masks_input"].detach().cpu(),
                            preds_image_out,
                            depth_min,
                            depth_max,
                        )
                    else:
                        utils.save_image(pred, preds_image_out, is_gamma=False)

                cams = batch["cameras_output"].detach().cpu().numpy()
                with pathmgr.open(
                    os.path.join(args.image_dir, f"{it:06d}_cam.npy"), "wb"
                ) as fp:
                    np.save(fp, cams)

        if it % args.saveckp_iter_freq == 0 and it != 0 and abs_it != 0:
            save_dict = {
                "mvencoder": mvencoder.state_dict(),
                "voldecoder": utils.get_fsdp_full_state_dict(voldecoder),
                "volsdf": volsdf.state_dict(),
                "pdecoder": pdecoder.state_dict(),
                "optimizer_fsdp": utils.get_fsdp_full_optim_state_dict(
                    voldecoder, optimizer_fsdp
                ),
                "optimizer_ddp": optimizer_ddp.state_dict(),
                "epoch": epoch,
                "start_it": abs_it + 1,
                "args": args,
            }
            utils.save_on_master(
                save_dict,
                os.path.join(args.checkpoint_dir, "last.pth"),
                backup_ckp_epoch=-1,
            )

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    logger.info(f"Averaged stats: {metric_logger}")
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def train_one_iteration(
    batch,
    dinoencoder,
    mvencoder,
    voldecoder,
    volsdf,
    inv_std,
    pdecoder,
    renderer,
    loss,
    loss_func_dict,
    loss_weights_dict,
    args,
    auto_cast_dtype=torch.bfloat16,
):
    if args.image_num_per_batch_range[0] < args.image_num_per_batch_range[1] + 1:
        image_num = np.random.randint(
            args.image_num_per_batch_range[0],
            args.image_num_per_batch_range[1] + 1,
        )
        batch["rgb_input"] = batch["rgb_input"][:, :image_num, :]
        batch["mask_input"] = batch["mask_input"][:, :image_num, :]
        batch["rays_o_input"] = batch["rays_o_input"][:, :image_num, :]
        batch["rays_d_input"] = batch["rays_d_input"][:, :image_num, :]
        batch["rays_d_un_input"] = batch["rays_d_un_input"][:, :image_num, :]
        batch["uv_input"] = batch["uv_input"][:, :image_num, :]
        batch["K_input"] = batch["K_input"][:, :image_num, :, :]
        batch["cameras_input"] = batch["cameras_input"][:, :image_num, :]
        if "crop_input" in batch:
            batch["crop_input"] = batch["crop_input"][:, :image_num, :]
        if "depth_input" in batch:
            batch["depth_input"] = batch["depth_input"][:, :image_num, :, :, :]
        if "normal_input" in batch:
            batch["normal_input"] = batch["normal_input"][:, :image_num, :, :, :]
        if "depth_masks_input" in batch:
            batch["depth_masks_input"] = batch["depth_masks_input"][
                :, :image_num, :, :, :
            ]
        if "bgs_input" in batch:
            batch["bgs_input"] = batch["bgs_input"][:, :image_num, :, :, :]

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

    gts = {}
    if args.prediction_type == "rgb":
        keys = ["rgb", "normal", "numerical_normal", "mask", "depth"]
    elif args.prediction_type == "brdf":
        keys = [
            "albedo",
            "roughness",
            "metallic",
            "normal",
            "numerical_normal",
            "mask",
            "depth",
        ]
    elif args.prediction_type == "both":
        keys = [
            "rgb",
            "albedo",
            "roughness",
            "metallic",
            "normal",
            "numerical_normal",
            "mask",
            "depth",
        ]

    for _, key in enumerate(keys):
        if key == "numerical_normal":
            if "normal_output" not in batch:
                continue
            gts[key] = batch["normal_output"].to(device=args.gpu)
        else:
            if f"{key}_output" not in batch:
                continue
            gts[key] = batch[f"{key}_output"].to(device=args.gpu)

    if "depth_masks_output" in batch:
        depth_mask = batch["depth_masks_output"].to(device=args.gpu)
    else:
        depth_mask = None

    if "normal_masks_output" in batch:
        normal_mask = batch["normal_masks_output"].to(device=args.gpu)
    else:
        normal_mask = None

    normal_weight_max = max(
        loss_weights_dict["mse"]["normal"], loss_weights_dict["perceptual"]["normal"]
    )
    numerical_normal_weight_max = max(
        loss_weights_dict["mse"]["numerical_normal"],
        loss_weights_dict["perceptual"]["numerical_normal"],
    )

    # Compute point loss
    aux_loss = 0
    depth_mask_input = batch["depth_masks_input"].to(
        device=args.gpu, dtype=torch.float32
    )
    depth_mask_input = depth_mask_input.reshape(batch_size, image_num, height, width, 1)
    points_gt = batch["surface_points_input"][:, :image_num, :].to(
        device=args.gpu, dtype=torch.float32
    )
    point_loss = (
        F.mse_loss(points * depth_mask_input, points_gt * depth_mask_input)
        * loss_weights_dict["mse"]["surface_point"]
    )
    cams_input = (
        batch["cameras_input"][:, :image_num, :16]
        .to(device=args.gpu, dtype=torch.float32)
        .reshape(batch_size, image_num, 4, 4)
    )
    z_axis = cams_input[:, :, :3, 2].reshape(batch_size, image_num, 1, 1, 3)
    origin_posts = cams_input[:, :, :3, 3].reshape(batch_size, image_num, 1, 1, 3)
    surface_depth = torch.sum(-z_axis * (points - origin_posts), dim=-1, keepdim=True)
    aux_loss += torch.nan_to_num(point_loss, nan=0.0, posinf=0.0, neginf=0.0)

    preds, losses = loss.forward_and_backward(
        volsdf,
        renderer,
        volume,
        inv_std,
        rays_o,
        rays_d,
        cams_output,
        gts,
        depth_mask,
        normal_mask,
        loss_weights_dict,
        loss_func_dict,
        aux_loss=aux_loss,
        use_occ_grid=args.ocgrid_acc,
        compute_normal=(normal_weight_max > 0),
        compute_numerical_normal=(numerical_normal_weight_max > 0),
    )
    losses["point_mse"] = point_loss
    preds["surface_depth"] = surface_depth
    preds["surface_points"] = points
    volume = {
        "volume": volume,
        "inv_std": inv_std,
    }
    return preds, gts, volume, losses


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()

    if "LOCAL_RANK" not in os.environ.keys():
        local_rank = 0
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
    ngpus_per_node = torch.cuda.device_count()

    train_lrm(local_rank, ngpus_per_node, args)
