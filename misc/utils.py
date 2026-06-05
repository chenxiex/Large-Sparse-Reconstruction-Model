# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import builtins
import copy
import datetime
import os

import pytorch3d

# pyre-fixme[21]: Could not find module `pytorch3d.renderer`.
import pytorch3d.renderer

# pyre-fixme[21]: Could not find module `pytorch3d.structures`.
import pytorch3d.structures

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict, deque, OrderedDict

import cv2
import ffmpeg
import matplotlib.cm as cm
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.utils.data
import torchvision
import trimesh
from plyfile import PlyData, PlyElement

from skimage import measure
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from data_loader.utils import compute_rays
from misc.dist_helper import get_rank, get_world_size, is_dist_avail_and_initialized
from misc.io_helper import pathmgr

print_debug_info = False


class _RepeatSampler(object):
    """Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


class RepeatedDataLoader(torch.utils.data.dataloader.DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self.iterator)


def linear_to_srgb(l):
    # s = np.zeros_like(l)
    s = torch.zeros_like(l)
    m = l <= 0.00313066844250063
    s[m] = l[m] * 12.92
    s[~m] = 1.055 * (l[~m] ** (1.0 / 2.4)) - 0.055
    return s


def _suppress_print(gpu=None):
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    if (gpu is not None and gpu != 0) or (gpu is None and not is_main_process()):
        builtins.print = print_pass


def clip_gradients(model, clip, check_nan_inf=True, file_name=None):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if check_nan_inf:
        for _, p in model.named_parameters():
            if p.grad is not None:
                p.grad.data = torch.nan_to_num(
                    p.grad.data, nan=0.0, posinf=0.0, neginf=0.0
                )

    if isinstance(model, FSDP):
        total_norm = model.clip_grad_norm_(clip)
        return [total_norm.item() if hasattr(total_norm, "item") else total_norm]
    else:
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        return [total_norm.item() if hasattr(total_norm, "item") else total_norm]


def _scale_pos_embed_3d(value, target_value, name):
    num_token, dim = value.shape[1:3]
    volume_res = int(np.cbrt(num_token))
    target_num_token = target_value.shape[1]
    target_volume_res = int(np.cbrt(target_num_token))
    value = value.permute(0, 2, 1)
    value = value.reshape(1, dim, volume_res, volume_res, volume_res)
    value = nn.functional.interpolate(value, target_volume_res, mode="trilinear")
    value = value.reshape(1, dim, target_num_token)
    value = value.permute(0, 2, 1)
    print(f"=> Scaled {name} from {num_token} to {target_num_token} tokens (3D)")
    return value


def _scale_pos_embed_1d(value, target_value, name):
    num_token, dim = value.shape[1:3]
    target_num_token = target_value.shape[1]
    value = value.permute(0, 2, 1)
    value = nn.functional.interpolate(value, target_num_token, mode="linear")
    value = value.permute(0, 2, 1)
    print(f"=> Scaled {name} from {num_token} to {target_num_token} tokens (1D)")
    return value


def _is_pos_embed_3d(name):
    base_name = name.replace("module.", "")
    return base_name == "pos_embed"


def _is_pos_embed_1d(name):
    base_name = name.replace("module.", "")
    return base_name in ("pos_embed_x", "pos_embed_y", "pos_embed_z")


def filter_weights_with_wrong_size(
    model, weights, scale_pos_embed=False, state_dict=None
):
    new_weights = OrderedDict()
    missing_keys = []

    if state_dict is None:
        state_dict = model.state_dict()

    for name, value in weights.items():
        if name in state_dict:
            target_value = state_dict[name]
            if value.size() != target_value.size():
                if _is_pos_embed_3d(name) and scale_pos_embed:
                    new_weights[name] = _scale_pos_embed_3d(value, target_value, name)
                elif _is_pos_embed_1d(name) and scale_pos_embed:
                    new_weights[name] = _scale_pos_embed_1d(value, target_value, name)
                else:
                    missing_keys.append(name)
            else:
                new_weights[name] = value
        elif "module." + name in state_dict:
            target_value = state_dict["module." + name]
            if value.size() != target_value.size():
                if _is_pos_embed_3d(name) and scale_pos_embed:
                    new_weights["module." + name] = _scale_pos_embed_3d(
                        value, target_value, name
                    )
                elif _is_pos_embed_1d(name) and scale_pos_embed:
                    new_weights["module." + name] = _scale_pos_embed_1d(
                        value, target_value, name
                    )
                else:
                    missing_keys.append(name)
            else:
                new_weights["module." + name] = value
        else:
            new_weights[name] = value

    return new_weights, missing_keys


def _init_brdf_mlps_from_rgb(weights):
    """
    Initialize mlp_basecolor and mlp_specular weights from mlp_rgb if they are missing.
    This is used when loading volsdf/volsdf_sparse checkpoints that only have mlp_rgb.
    Handles both prefixed (module.mlp_rgb.) and non-prefixed (mlp_rgb.) keys.
    """
    # Check for mlp_rgb with or without module. prefix
    has_mlp_rgb = any("mlp_rgb." in k for k in weights.keys())
    has_mlp_basecolor = any("mlp_basecolor." in k for k in weights.keys())
    has_mlp_specular = any("mlp_specular." in k for k in weights.keys())

    if has_mlp_rgb and not has_mlp_basecolor:
        print("=> Initializing mlp_basecolor from mlp_rgb weights")
        for k, v in list(weights.items()):
            if "mlp_rgb." in k:
                new_key = k.replace("mlp_rgb.", "mlp_basecolor.")
                weights[new_key] = v.clone()

    if has_mlp_rgb and not has_mlp_specular:
        print("=> Initializing mlp_specular from mlp_rgb weights")
        for k, v in list(weights.items()):
            if "mlp_rgb." in k:
                new_key = k.replace("mlp_rgb.", "mlp_specular.")
                # mlp_specular has 2 output channels vs mlp_rgb's 3
                # For the final layer, we need to slice the weights
                if "weight" in k and v.shape[0] == 3:
                    weights[new_key] = v[:2].clone()
                elif "bias" in k and v.shape[0] == 3:
                    weights[new_key] = v[:2].clone()
                else:
                    weights[new_key] = v.clone()

    return weights


def load_ddp_state_dict(model, weights, key=None, filter_mismatch=True):
    weights = copy.deepcopy(weights)

    # Initialize BRDF MLPs from RGB for volsdf/volsdf_sparse if needed
    if key in ("volsdf", "volsdf_sparse"):
        weights = _init_brdf_mlps_from_rgb(weights)

    if isinstance(model, nn.parallel.DistributedDataParallel):
        if filter_mismatch:
            weights, missing_keys = filter_weights_with_wrong_size(
                model, weights, scale_pos_embed=(key == "voldecoder")
            )
            if len(missing_keys) > 0:
                print(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch.",
                )
        msg = model.load_state_dict(weights, strict=False)
    elif isinstance(model, torch.optim.Optimizer):
        # Preserve current param_groups hyperparameters (like betas, lr, weight_decay)
        # since load_state_dict replaces param_groups entirely
        saved_param_groups = [
            {k: v for k, v in pg.items() if k != "params"} for pg in model.param_groups
        ]
        msg = model.load_state_dict(weights)
        # Restore hyperparameters that may be missing from the checkpoint
        for pg, saved_pg in zip(model.param_groups, saved_param_groups):
            for key, value in saved_pg.items():
                if key not in pg:
                    pg[key] = value
    else:
        new_weights = OrderedDict()
        for k, v in weights.items():
            if k[:7] == "module.":
                name = k[7:]  # remove 'module.' of DataParallel/DistributedDataParallel
            else:
                name = k
            new_weights[name] = v
        if filter_mismatch:
            new_weights, missing_keys = filter_weights_with_wrong_size(
                model, new_weights, scale_pos_embed=(key == "voldecoder")
            )
            if len(missing_keys) > 0:
                print(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch.",
                )
        msg = model.load_state_dict(new_weights, strict=False)
    return msg


def load_fsdp_state_dict(
    model,
    weights,
    optimizer=None,
    optimizer_weights=None,
    key=None,
    filter_mismatch=True,
):
    from collections import namedtuple

    LoadResult = namedtuple("LoadResult", ["missing_keys", "unexpected_keys"])

    weights = copy.deepcopy(weights)

    if isinstance(model, FSDP):
        # For FSDP models, we need to handle the state dict properly
        # The saved state dict should be a full state dict (gathered from all ranks)
        new_weights = OrderedDict()
        for k, v in weights.items():
            # Remove any prefix like 'module.' or '_fsdp_wrapped_module.'
            if k.startswith("_fsdp_wrapped_module."):
                name = k[len("_fsdp_wrapped_module.") :]
            elif k.startswith("module."):
                name = k[7:]
            else:
                name = k
            # Ensure tensors are contiguous - required by set_model_state_dict
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                v = v.contiguous()
            new_weights[name] = v

        # Use the new get_state_dict/set_state_dict APIs
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            broadcast_from_rank0=True,  # Required for loading full state dict
        )

        if filter_mismatch:
            # Get model's current state dict using the new API
            model_state_dict = get_model_state_dict(model, options=options)

            # Use the shared filtering function that supports pos_embed scaling
            scale_pos_embed = key == "voldecoder"
            new_weights, missing_keys = filter_weights_with_wrong_size(
                model=None,
                weights=new_weights,
                scale_pos_embed=scale_pos_embed,
                state_dict=model_state_dict,
            )

            if len(missing_keys) > 0:
                print(
                    "Keys ",
                    missing_keys,
                    " are filtered out due to parameter size mismatch.",
                )

        for k, v in new_weights.items():
            if isinstance(v, torch.Tensor) and not v.is_contiguous():
                v = v.contiguous()
            new_weights[k] = v

        # Load model state dict using the new API
        set_model_state_dict(
            model,
            model_state_dict=new_weights,
            options=options,
        )

        # Compute missing and unexpected keys for return message
        model_keys = set(get_model_state_dict(model, options=options).keys())
        loaded_keys = set(new_weights.keys())
        msg = LoadResult(
            missing_keys=list(model_keys - loaded_keys),
            unexpected_keys=list(loaded_keys - model_keys),
        )
        # Load optimizer state if provided
        if optimizer is not None and optimizer_weights is not None:
            # Check if the optimizer state dict is in FSDP format (FQN keys) or regular format (int keys)
            state_keys = list(optimizer_weights.get("state", {}).keys())
            is_fsdp_format = len(state_keys) > 0 and isinstance(state_keys[0], str)

            if is_fsdp_format:
                # Preserve current param_groups hyperparameters (like betas, lr, weight_decay)
                # since set_optimizer_state_dict may replace param_groups entirely
                saved_param_groups = [
                    {k: v for k, v in pg.items() if k != "params"}
                    for pg in optimizer.param_groups
                ]
                # Use the new set_optimizer_state_dict API
                set_optimizer_state_dict(
                    model,
                    optimizer,
                    optim_state_dict=optimizer_weights,
                    options=options,
                )
                # Restore hyperparameters that may be missing from the checkpoint
                for pg, saved_pg in zip(optimizer.param_groups, saved_param_groups):
                    for key, value in saved_pg.items():
                        if key not in pg:
                            pg[key] = value
            else:
                # Regular format (int keys): load directly
                # This handles checkpoints saved before FSDP optimizer state dict was used
                print(
                    "=> Loading optimizer with regular format (int keys), skipping FSDP conversion"
                )
                try:
                    # Preserve current param_groups hyperparameters (like betas, lr, weight_decay)
                    # since load_state_dict replaces param_groups entirely
                    saved_param_groups = [
                        {k: v for k, v in pg.items() if k != "params"}
                        for pg in optimizer.param_groups
                    ]
                    optimizer.load_state_dict(optimizer_weights)
                    # Restore hyperparameters that may be missing from the checkpoint
                    for pg, saved_pg in zip(optimizer.param_groups, saved_param_groups):
                        for key, value in saved_pg.items():
                            if key not in pg:
                                pg[key] = value
                except Exception as e:
                    print(f"=> Warning: Could not load optimizer state directly: {e}")
                    print("=> Optimizer will start fresh")
        return msg
    else:
        # Fall back to regular loading for non-FSDP models
        return load_ddp_state_dict(
            model, weights, key=key, filter_mismatch=filter_mismatch
        )


def get_fsdp_full_state_dict(model):
    if isinstance(model, FSDP):
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        return get_model_state_dict(model, options=options)
    else:
        return model.state_dict()


def get_fsdp_full_optim_state_dict(model, optimizer):
    if isinstance(model, FSDP):
        options = StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
        )
        return get_optimizer_state_dict(model, optimizer, options=options)
    else:
        return optimizer.state_dict()


def remap_spatial_sparse_attention_weights(state_dict):
    """
    Remap weights from the old SpatialSparseAttention module to the new split modules:
    SpatialSparseAttentionIn, SpatialSparseAttentionMid, SpatialSparseAttentionOut.

    Old structure (SpatialSparseAttention):
        - compression_key.*
        - compression_value.*
        - gate.*
        (Note: norm_layer has no parameters since elementwise_affine=False)

    New structure:
        - SpatialSparseAttentionIn:
            - compression_key.*
            - compression_value.*
        - SpatialSparseAttentionMid:
            (no learnable parameters - norm_layer has elementwise_affine=False)
        - SpatialSparseAttentionOut:
            - gate.*

    Also handles typo: y_2_x_atten -> y_2_x_attn (maps to y_2_x_attn_in and y_2_x_attn_out)
    """
    import re

    new_state_dict = OrderedDict()

    # Pattern to match old SpatialSparseAttention module paths
    # Examples:
    #   crs_attn.x_2_x_attn.compression_key.* -> crs_attn.x_2_x_attn_in.compression_key.*
    #   crs_attn.x_2_x_attn.gate.* -> crs_attn.x_2_x_attn_out.gate.*

    # Pattern to detect old-style SpatialSparseAttention keys
    # Match patterns like: prefix.x_2_x_attn.compression_key.suffix or prefix.y_2_y_attn.gate.suffix
    # Note: norm_layer is excluded since it has no learnable parameters (elementwise_affine=False)
    attn_pattern = re.compile(
        r"^(.*\.)([xy]_2_[xy]_attn)\.(compression_key|compression_value|gate)\.(.*)$"
    )

    # Pattern to match typo: y_2_x_atten (should be y_2_x_attn)
    atten_typo_pattern = re.compile(
        r"^(.*\.)(y_2_x_atten)\.(compression_key|compression_value|gate)\.(.*)$"
    )

    remapped_count = 0
    for key, value in state_dict.items():
        match = attn_pattern.match(key)
        typo_match = atten_typo_pattern.match(key)

        if match or typo_match:
            m = match if match else typo_match
            prefix = m.group(1)  # e.g., "crs_attn."
            attn_name = m.group(2)  # e.g., "x_2_x_attn" or "y_2_x_atten" (typo)
            component = m.group(3)  # e.g., "compression_key"
            suffix = m.group(4)  # e.g., "layer1.weight"

            # Fix typo: y_2_x_atten -> y_2_x_attn
            if attn_name == "y_2_x_atten":
                attn_name = "y_2_x_attn"

            # Determine which new module this belongs to
            if component in ["compression_key", "compression_value"]:
                new_module_suffix = "_in"
            elif component == "gate":
                new_module_suffix = "_out"
            else:
                # Unknown component, keep as is
                new_state_dict[key] = value
                continue

            # Create new key
            new_key = f"{prefix}{attn_name}{new_module_suffix}.{component}.{suffix}"
            new_state_dict[new_key] = value
            remapped_count += 1
        else:
            # Not a SpatialSparseAttention key, keep as is
            new_state_dict[key] = value

    if remapped_count > 0:
        print(
            f"=> Remapped {remapped_count} SpatialSparseAttention weights to new In/Mid/Out structure"
        )

    return new_state_dict


def restart_from_checkpoint(
    ckp_path, run_variables=None, load_weights_only=False, **kwargs
):
    """
    Re-start from checkpoint
    """
    if not pathmgr.isfile(ckp_path):
        raise FileNotFoundError(ckp_path)
    print("Found checkpoint at {}".format(ckp_path))

    # open checkpoint file
    if get_world_size() == 1:
        ckp_path = pathmgr.get_local_path(ckp_path, force=True)
    with pathmgr.open(ckp_path, "rb") as fb:
        checkpoint = torch.load(fb, map_location="cpu", weights_only=False)

    # Remap SpatialSparseAttention weights to new In/Mid/Out structure
    # This allows loading old checkpoints with new model architecture
    for ckpt_key in ["voldecoder", "mvencoder"]:
        if ckpt_key in checkpoint and isinstance(checkpoint[ckpt_key], dict):
            checkpoint[ckpt_key] = remap_spatial_sparse_attention_weights(
                checkpoint[ckpt_key]
            )

    # Separate FSDP models and their optimizers from DDP models
    fsdp_models = {}
    fsdp_optimizers = {}
    ddp_models_and_optimizers = {}

    for key, value in kwargs.items():
        if value is None:
            continue
        if isinstance(value, FSDP):
            fsdp_models[key] = value
        elif key.endswith("_fsdp") and isinstance(value, torch.optim.Optimizer):
            # This is an FSDP optimizer (e.g., optimizer_fsdp)
            fsdp_optimizers[key] = value
        else:
            ddp_models_and_optimizers[key] = value

    # Load FSDP models
    for key, model in fsdp_models.items():
        if key in checkpoint:
            try:
                # Get corresponding optimizer if exists
                optimizer_key = "optimizer_fsdp"
                optimizer = fsdp_optimizers.get(optimizer_key)
                optimizer_weights = (
                    checkpoint.get(optimizer_key) if not load_weights_only else None
                )

                msg = load_fsdp_state_dict(
                    model,
                    checkpoint[key],
                    optimizer=optimizer,
                    optimizer_weights=optimizer_weights,
                    key=key,
                )
                print(
                    "=> loaded FSDP model '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except Exception as e:
                print(
                    "=> failed to load FSDP model '{}' from checkpoint: '{}', error: {}".format(
                        key, ckp_path, e
                    )
                )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # Load FSDP optimizer if not already loaded with model
    # Note: FSDP optimizer loading is handled in load_fsdp_state_dict above
    # This is for cases where we want to load optimizer separately
    for key, _ in fsdp_optimizers.items():
        if key in checkpoint and not load_weights_only:
            # Check if we already loaded this optimizer with the model
            # We need to find the corresponding FSDP model
            model_key = None
            for mk in fsdp_models.keys():
                model_key = mk
                break  # Use the first FSDP model for optimizer loading

            if model_key is not None and fsdp_models.get(model_key) is not None:
                # Already loaded with the model, skip
                print("=> FSDP optimizer '{}' already loaded with model".format(key))
            else:
                print(
                    "=> Warning: FSDP optimizer '{}' found but no matching model to load with".format(
                        key
                    )
                )

    # Load DDP models and regular optimizers
    for key, value in ddp_models_and_optimizers.items():
        # Skip loading optimizers when load_weights_only is True
        if load_weights_only and isinstance(value, torch.optim.Optimizer):
            print("=> skipping optimizer '{}' (load_weights_only=True)".format(key))
            continue
        if key in checkpoint and value is not None:
            try:
                msg = load_ddp_state_dict(value, checkpoint[key], key=key)
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                print(
                    "=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path)
                )
        elif key == "volupsampler":
            try:
                msg = load_ddp_state_dict(
                    value, checkpoint["voldecoder"], key="volupsampler"
                )
                print(
                    "=> loaded '{}' from checkpoint '{}' with msg {}".format(
                        key, ckp_path, msg
                    )
                )
            except TypeError:
                print(
                    "=> failed to load '{}' from checkpoint: '{}'".format(key, ckp_path)
                )
        else:
            print("=> key '{}' not found in checkpoint: '{}'".format(key, ckp_path))

    # re load variable important for the run
    if not load_weights_only and run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]


def cosine_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
):
    print(
        f"cosine scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    warmup_iters = min(warmup_iters, epochs * niter_per_ep)
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            raise RuntimeError(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep, (
        f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
    )
    return schedule


def linear_scheduler(
    base_value,
    final_value,
    epochs,
    niter_per_ep,
    warmup_iters,
    start_warmup_value=1e-10,
):
    print(
        f"linear scheduler - lr: {base_value}, min_lr: {final_value}, epochs: {epochs}, it_per_epoch: {niter_per_ep}, warmup_iters: {warmup_iters}, startup_warmup: {start_warmup_value}"
    )
    warmup_schedule = np.array([])
    if warmup_iters > 0:
        if warmup_iters > epochs * niter_per_ep:
            raise RuntimeError(
                f"warm iterations number is exceeding the total number iterations. Epoch: {epochs}: Iteration/Epoch: {niter_per_ep}"
            )

        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = np.linspace(base_value, final_value, len(iters))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep, (
        f"Schedule length {len(schedule)} needs to match epoch {epochs} x iteration per epoch {niter_per_ep}"
    )
    return schedule


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=5000, fmt=None):
        if fmt is None:
            fmt = "{median:.8f} ({global_avg:.8f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", logger=None, tb_writer=None, epoch=0):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.logger = logger
        self.tb_writer = tb_writer
        self.epoch = epoch
        assert self.logger is not None

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(
            "'{}' object has no attribute '{}'".format(type(self).__name__, attr)
        )

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.6f}")
        data_time = SmoothedValue(fmt="{avg:.6f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if self.tb_writer:
                    for k in self.meters:
                        global_step = self.epoch * len(iterable) + i
                        self.tb_writer.add_scalar(
                            f"train/{k}",
                            self.meters[k].global_avg,
                            global_step=global_step,
                            new_style=True,
                        )
                if torch.cuda.is_available():
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB,
                        )
                    )
                else:
                    self.logger.info(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
                sys.stdout.flush()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(
            "{} Total time: {} ({:.6f} s / it)".format(
                header, total_time_str, total_time / len(iterable)
            ),
        )


def is_main_process():
    return get_rank() == 0


def save_on_master(ckpt, model_path, backup_ckp_epoch=-1, topk=-1):
    if not is_main_process():
        return

    basedir = os.path.dirname(model_path)
    if not pathmgr.isdir(basedir):
        pathmgr.mkdirs(basedir)
    with pathmgr.open(model_path, "wb") as fp:
        torch.save(ckpt, fp)

    if backup_ckp_epoch >= 0:
        target_path = os.path.join(basedir, f"ckpt_{backup_ckp_epoch}.pth")
        pathmgr.copy(model_path, target_path, overwrite=True)

    return


def save_image(image, name, is_gamma=False):
    if len(image.shape) == 5:
        batch_size, im_num, _, h, w = image.shape
        image = image.reshape((batch_size * im_num, -1, h, w))
        nrow = im_num
    else:
        batch_size = image.shape[0]
        nrow = batch_size

    if "mask" not in name.split("/")[-1] and "env" not in name.split("/")[-1]:
        image = 0.5 * (image + 1)

    if is_gamma:
        image = linear_to_srgb(image)

    with pathmgr.open(name, "wb") as fp:
        torchvision.utils.save_image(image, fp, nrow=nrow)


def save_image_list(image_list, name):
    im_num = len(image_list)
    batch_size = len(image_list[0])
    with pathmgr.open(name, "w") as fp:
        for n in range(0, batch_size):
            for m in range(0, im_num):
                fp.write(image_list[m][n] + "\n")


def save_single_png(images, name, image_ids=None, is_gamma=False):
    if "mask" not in name:
        images = 0.5 * (images + 1)
    if is_gamma:
        images = linear_to_srgb(images)
    images = images.detach().cpu().numpy()
    images = (255 * np.clip(images, 0, 1)).astype(np.uint8)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        if im.shape[-1] == 3:
            im = np.ascontiguousarray(im[:, :, ::-1])
        else:
            im = np.concatenate([im, im, im], axis=-1)

        buffer = cv2.imencode(".png", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with pathmgr.open(im_name, "wb") as fp:
            fp.write(buffer)


def save_single_exr(images, name, image_ids=None, is_gamma=False):
    images = images.detach().cpu().float().numpy()
    images = images.astype(np.float32)

    batch_size = images.shape[0]
    for n in range(0, batch_size):
        im = images[n].transpose(1, 2, 0)
        if im.shape[-1] == 3:
            im = np.ascontiguousarray(im[:, :, ::-1])
        else:
            im = np.concatenate([im, im, im], axis=-1)
        buffer = cv2.imencode(".exr", im)[1]
        buffer = np.array(buffer).tobytes()
        if image_ids is None:
            im_name = name % n
        else:
            im_name = name % image_ids[n]
        with pathmgr.open(im_name, "wb") as fp:
            fp.write(buffer)


def save_depth(depth, depth_mask, name, depth_min=1.5, depth_max=2.5):
    batch_size, im_num, _, h, w = depth.shape
    depth = depth * depth_mask
    depth = np.clip(depth, depth_min, depth_max)

    cmap = cm.get_cmap("jet")
    depth = (depth.reshape(-1) - depth_min) / (depth_max - depth_min)
    depth = depth.detach().cpu().numpy()
    colors = cmap(depth.flatten())[:, :3]
    colors = colors.reshape(batch_size * im_num, h, w, 3)
    colors = colors.transpose(0, 3, 1, 2)
    colors = torch.from_numpy(colors)
    with pathmgr.open(name, "wb") as fp:
        torchvision.utils.save_image(colors, fp, nrow=im_num)


def save_ply(path, xyz, rgb=None, opacity=None, scale=None, rotation=None):
    def construct_list_of_attributes():
        l = ["x", "y", "z"]
        # All channels except the 3 DC
        if rgb is not None:
            l = l + ["r", "g", "b"]
        if opacity is not None:
            l.append("opacity")
        if scale is not None:
            for i in range(3):
                l.append("scale_{}".format(i))
        if rotation is not None:
            for i in range(4):
                l.append("rot_{}".format(i))
        return l

    data = []
    xyz = xyz.to(dtype=torch.float32)
    xyz = xyz.detach().cpu().numpy().astype(np.float32)
    data.append(xyz)

    if rgb is not None:
        rgb = rgb.to(dtype=torch.float32)
        rgb = rgb.detach().cpu().numpy().astype(np.float32)
        data.append(rgb)

    if opacity is not None:
        opacity = opacity.to(dtype=torch.float32)
        opacity = opacity.detach().cpu().numpy().astype(np.float32)
        data.append(opacity)

    if scale is not None:
        scale = scale.to(dtype=torch.float32)
        scale = scale.detach().cpu().numpy().astype(np.float32)
        data.append(scale)

    if rotation is not None:
        rotation = rotation.to(dtype=torch.float32)
        rotation = rotation.detach().cpu().numpy().astype(np.float32)
        data.append(rotation)

    dtype_full = [(attribute, "f4") for attribute in construct_list_of_attributes()]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate(data, axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, "vertex")
    with tempfile.TemporaryDirectory() as temp_dir:
        local_point_path = os.path.join(temp_dir, path.split("/")[-1])
        PlyData([el]).write(local_point_path)
        pathmgr.copy_from_local(local_point_path, path, overwrite=True)


def flood_fill_exterior(values, threshold=0.0):
    """
    Use flood fill to identify exterior voxels and remove floating artifacts.

    Starting from the 8 corner voxels of the cube, identify all connected
    components of positive voxels that are reachable from those corners.
    Components that are not connected to any corner are considered floating
    artifacts inside the object and are set to negative.

    Uses scipy.ndimage for fast C-optimized connected component labeling.

    Args:
        values: 3D numpy array of SDF values (N, N, N)
        threshold: SDF threshold for surface (default 0.0)

    Returns:
        Modified values array with floating artifacts removed
    """
    from scipy import ndimage

    N = values.shape[0]

    # Create binary mask of positive (exterior) voxels
    positive_mask = values > threshold

    # Label connected components using 6-connectivity (face-adjacent)
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity
    labeled_array, num_features = ndimage.label(positive_mask, structure=structure)

    if num_features == 0:
        return values

    # Define the 8 corner voxels of the cube
    corners = [
        (0, 0, 0),
        (0, 0, N - 1),
        (0, N - 1, 0),
        (0, N - 1, N - 1),
        (N - 1, 0, 0),
        (N - 1, 0, N - 1),
        (N - 1, N - 1, 0),
        (N - 1, N - 1, N - 1),
    ]

    # Find which labels are connected to the 8 corners
    exterior_labels = set()
    for corner in corners:
        label = labeled_array[corner]
        if label > 0:  # label 0 is background (negative voxels)
            exterior_labels.add(label)

    # Find floating components (positive but not connected to any corner)
    all_labels = set(range(1, num_features + 1))
    floating_labels = all_labels - exterior_labels

    if len(floating_labels) > 0:
        # Create mask of floating voxels
        floating_mask = np.isin(labeled_array, list(floating_labels))
        num_floating = np.sum(floating_mask)
        print(
            f"Flood fill: removed {num_floating} floating voxels from {len(floating_labels)} components"
        )
        # Set floating voxels to negative (inside the object)
        values[floating_mask] = -np.abs(values[floating_mask])

    return values


def save_mesh(
    path,
    values,
    N,
    threshold,
    radius,
    init_cam_rot=None,
    save_volume=False,
    use_flood_fill=True,
):
    values = values.detach().cpu().numpy()
    values = values.reshape(N, N, N).astype(np.float32)

    # Apply flood fill to remove floating artifacts inside the object
    if use_flood_fill:
        values = flood_fill_exterior(values, threshold)

    if save_volume:
        with pathmgr.open(path + ".npy", "wb") as fp:
            np.save(fp, values)

    try:
        vertices, triangles, normals, _ = measure.marching_cubes(values, threshold)
        print(
            "vertices num %d triangles num %d threshold %.3f"
            % (vertices.shape[0], triangles.shape[0], threshold)
        )

        vertices = vertices / (N - 1.0) * 2 * radius - radius
        if init_cam_rot is not None:
            init_cam_rot = init_cam_rot.detach().cpu().numpy().squeeze()
            vertices = np.matmul(init_cam_rot, vertices.transpose(1, 0)).transpose(1, 0)

        mesh = trimesh.Trimesh(
            vertices=vertices, faces=triangles, vertex_normals=normals
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            local_mesh_path = os.path.join(temp_dir, os.path.basename(path))
            mesh.export(local_mesh_path)
            pathmgr.copy_from_local(local_mesh_path, path, overwrite=True)
    except:
        print("Failed to extract mesh.")


def load_ply(path):
    plydata = PlyData.read(path)

    xyz = np.stack(
        (
            np.asarray(plydata.elements[0]["x"]),
            np.asarray(plydata.elements[0]["y"]),
            np.asarray(plydata.elements[0]["z"]),
        ),
        axis=1,
    )

    rgb = np.stack(
        (
            np.asarray(plydata.elements[0]["r"]),
            np.asarray(plydata.elements[0]["g"]),
            np.asarray(plydata.elements[0]["b"]),
        ),
        axis=1,
    )

    opacity = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    scale_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")
    ]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scale = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scale[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [
        p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
    ]
    rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
    rotation = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rotation[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.from_numpy(xyz.astype(np.float32))
    rgb = torch.from_numpy(rgb.astype(np.float32))
    opacity = torch.from_numpy(opacity.astype(np.float32))
    scale = torch.from_numpy(scale.astype(np.float32))
    rotation = torch.from_numpy(rotation.astype(np.float32))

    return {
        "xyz": xyz,
        "rgb": rgb,
        "opacity": opacity,
        "scale": scale,
        "rotation": rotation,
    }


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_params_group_single_model(args, model, freeze_transformer=False):
    upsampler = []
    regularized = []
    not_regularized = []
    for name, param in model.named_parameters():
        if param is None:
            continue
        if not param.requires_grad:
            continue
        # we do not regularize biases nor Norm parameters
        if "upsampler" in name:
            upsampler.append(param)
        else:
            if name.endswith(".bias") or len(param.shape) == 1:
                not_regularized.append(param)
            elif len(name.split(".")) > 3 and "norm" in name.split(".")[2]:
                not_regularized.append(param)
            else:
                regularized.append(param)

    if freeze_transformer:
        lr_t = 0
        lr_c = args.lr
    else:
        lr_t = args.lr
        lr_c = args.lr

    return [
        {"params": regularized, "weight_decay": args.weight_decay, "lr": lr_t},
        {"params": not_regularized, "weight_decay": 0.0, "lr": lr_t},
        {"params": upsampler, "weight_decay": args.weight_decay, "lr": lr_c},
    ]


def get_params_groups(args, **kwargs):
    params_groups = []
    for key, value in kwargs.items():
        if "mvencoder" in key or "voldecoder" in key:
            params_groups += get_params_group_single_model(
                args,
                value,
            )
        else:
            params_groups += get_params_group_single_model(args, value)
    return params_groups


def create_video_cameras(radius, frame_num, res, elevation=20, fov=60, init_cam=None):
    fov = fov / 180.0 * np.pi
    dist = radius / np.sin(fov / 2.0)
    theta = elevation / 180.0 * np.pi
    x_axis = np.array([1.0, 0, 0], dtype=np.float32)
    y_axis = np.array([0, 1.0, 0], dtype=np.float32)
    z_axis = np.array([0, 0, 1.0], dtype=np.float32)

    if init_cam is not None:
        init_cam = init_cam.detach().cpu().numpy()
        init_cam = init_cam.transpose(1, 0)
        inv = np.eye(4, dtype=np.float32)
        inv[0:3, 0:3] = init_cam

    camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr = [], [], [], []
    for n in range(0, frame_num):
        phi = float(n) / frame_num * np.pi * 2
        origin = (
            np.cos(theta) * np.cos(phi) * x_axis
            + np.cos(theta) * np.sin(phi) * y_axis
            + np.sin(theta) * z_axis
        )
        origin = origin * dist

        target = np.array([0, 0, 0], dtype=np.float32)
        up = np.array([0, 0, 1], dtype=np.float32)
        cam_z_axis = (origin - target) / np.linalg.norm(origin - target)
        cam_y_axis = up - np.sum(cam_z_axis * up) * cam_z_axis
        cam_y_axis = cam_y_axis / np.linalg.norm(cam_y_axis)
        cam_x_axis = np.cross(cam_y_axis, cam_z_axis)

        extrinsic = np.eye(4, dtype=np.float32)
        extrinsic[:3, 0] = cam_x_axis
        extrinsic[:3, 1] = cam_y_axis
        extrinsic[:3, 2] = cam_z_axis
        extrinsic[:3, 3] = origin
        if init_cam is not None:
            extrinsic = np.matmul(inv, extrinsic)

        camera = np.zeros(20, dtype=np.float32)
        camera[0:16] = extrinsic.reshape(-1)
        camera[16:20] = np.array([fov, fov, 0.5, 0.5])

        rays_o, rays_d, rays_d_un, _, _ = compute_rays(fov, extrinsic, res)

        camera_arr.append(camera)
        rays_o_arr.append(rays_o)
        rays_d_arr.append(rays_d)
        rays_d_un_arr.append(rays_d_un)

    camera_arr = np.stack(camera_arr, axis=0)[None, :, :].astype(np.float32)
    rays_o_arr = np.stack(rays_o_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_arr = np.stack(rays_d_arr, axis=0)[None, :, :, :].astype(np.float32)
    rays_d_un_arr = np.stack(rays_d_un_arr, axis=0)[None, :, :, :].astype(np.float32)

    return camera_arr, rays_o_arr, rays_d_arr, rays_d_un_arr


def convert_exr_to_png(exr_path, png_path):
    """Convert a 32-bit linear EXR image to a 16-bit sRGB PNG image."""
    img = cv2.imread(exr_path, cv2.IMREAD_UNCHANGED)
    img = np.clip(img, 0.0, 1.0)
    linear = img[:, :, :3]
    srgb = np.where(
        linear <= 0.00313066844250063,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )
    if img.shape[2] == 4:
        srgb = np.concatenate([srgb, img[:, :, 3:4]], axis=2)
    srgb = (srgb * 65535).astype(np.uint16)
    cv2.imwrite(png_path, srgb)


def render_images_from_mesh(
    path,
    prediction_type,
    blender_bin,
    camera_arr,
    eva_output_views,
    output_dir,
    env_path=None,
    env_mean=None,
    mode="texture",
    show_bg=False,
    save_video=False,
    image_resolution=512,
    srgb_specularity=False,
    rotate_y_to_z=False,
    white_bg=False,
):
    mesh_path = pathmgr.get_local_path(os.path.join(path, "mesh_uv.obj"), force=True)
    if prediction_type == "rgb" or prediction_type == "both":
        rgb_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_rgb.png"), force=True
        )
    if prediction_type == "brdf" or prediction_type == "both":
        albedo_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_albedo.png"),
            force=True,
        )
        roughness_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_roughness.png"),
            force=True,
        )
        metallic_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_metallic.png"),
            force=True,
        )

    if env_path is not None:
        env_file = pathmgr.get_local_path(env_path, force=True)
        if env_mean is not None:
            env = cv2.imread(env_file, -1)
            scale = env_mean / np.mean(env)
            env = env * scale
            cv2.imwrite(env_file, env)

    with tempfile.TemporaryDirectory() as temp_dir:

        def write_images_to_video(name, im_num):
            local_video_path = os.path.join(temp_dir, "video_%s.mp4" % name)
            video_path = os.path.join(output_dir, "video_%s.mp4" % name)
            ffmpeg.input(
                os.path.join(temp_dir, "%03d_" + "%s.png" % name), framerate=24
            ).output(
                local_video_path,
                pix_fmt="yuv420p",
                vcodec="libx264",
                crf=18,
                preset="slow",
            ).run()
            pathmgr.copy_from_local(local_video_path, video_path, overwrite=True)

        def copy_images_to_output(name, im_num, is_hdr=False):
            for n in range(0, im_num):
                if is_hdr:
                    exr_path = os.path.join(temp_dir, "%03d_%s.exr" % (n, name))
                    exr_output_path = os.path.join(
                        output_dir, "%03d_%s.exr" % (eva_output_views[n], name)
                    )
                    pathmgr.copy_from_local(exr_path, exr_output_path, overwrite=True)
                    png_path = os.path.join(temp_dir, "%03d_%s.png" % (n, name))
                    convert_exr_to_png(exr_path, png_path)
                    png_output_path = os.path.join(
                        output_dir, "%03d_%s.png" % (eva_output_views[n], name)
                    )
                    pathmgr.copy_from_local(png_path, png_output_path, overwrite=True)
                else:
                    im_path = os.path.join(temp_dir, "%03d_%s.png" % (n, name))
                    im_output_path = os.path.join(
                        output_dir, "%03d_%s.png" % (eva_output_views[n], name)
                    )
                    pathmgr.copy_from_local(im_path, im_output_path, overwrite=True)

        camera_file = os.path.join(temp_dir, "camera_info.npy")
        fov = camera_arr[0, 16]
        camera_arr = camera_arr[:, :16].reshape(-1, 4, 4)
        camera_arr = camera_arr[:, :3, :]
        np.save(camera_file, camera_arr)
        fov_file = os.path.join(temp_dir, "fov.txt")
        with open(fov_file, "w") as fOut:
            fOut.write("%.7f" % fov)

        if mode == "texture" or mode == "both":
            blender_args = [
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                os.path.join(
                    os.path.dirname(__file__),
                    "render_pass_blender.py",
                ),
                "--",
                "--obj_path",
                mesh_path,
                "--output_dir",
                temp_dir,
                "--cam_pos_path",
                camera_file,
                "--cam_fov_path",
                fov_file,
                "--image_resolution",
                str(image_resolution),
            ]

            if prediction_type == "rgb" or prediction_type == "both":
                blender_args += ["--rgb_image_path", rgb_tex_path]
            if prediction_type == "brdf" or prediction_type == "both":
                blender_args += ["--albedo_image_path", albedo_tex_path]
                blender_args += ["--roughness_image_path", roughness_tex_path]
                blender_args += ["--metallic_image_path", metallic_tex_path]

            if show_bg and env_path is not None:
                blender_args += ["--environment_map_path", env_file]
                blender_args += ["--environment_show_image"]

            if rotate_y_to_z:
                blender_args += ["--rotate_y_to_z"]

            cmd = [blender_bin] + blender_args
            cmd_str = " ".join(cmd)
            print(cmd_str)
            subprocess.check_call(cmd)

            im_num = len(eva_output_views)
            if prediction_type == "rgb" or prediction_type == "both":
                copy_images_to_output("rgb", im_num)

            if prediction_type == "brdf" or prediction_type == "both":
                copy_images_to_output("albedo", im_num)
                copy_images_to_output("roughness", im_num)
                copy_images_to_output("metallic", im_num)

            if save_video:
                if prediction_type == "rgb" or prediction_type == "both":
                    write_images_to_video("rgb", im_num)

                if prediction_type == "brdf" or prediction_type == "both":
                    write_images_to_video("albedo", im_num)
                    write_images_to_video("roughness", im_num)
                    write_images_to_video("metallic", im_num)

        if mode == "relighting" or mode == "both":
            if prediction_type != "brdf" and prediction_type != "both":
                raise ValueError("relighting mode requires brdf prediction")

            if env_path is None:
                raise ValueError("relighting mode requires environment map")

            blender_args = [
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                os.path.join(
                    os.path.dirname(__file__),
                    "render_pbr_blender.py",
                ),
                "--",
                "--obj_path",
                mesh_path,
                "--output_dir",
                temp_dir,
                "--cam_pos_path",
                camera_file,
                "--cam_fov_path",
                fov_file,
                "--image_resolution",
                str(image_resolution),
            ]

            blender_args += ["--albedo_image_path", albedo_tex_path]
            blender_args += ["--roughness_image_path", roughness_tex_path]
            blender_args += ["--metallic_image_path", metallic_tex_path]
            blender_args += ["--environment_map_path", env_file]
            if show_bg:
                blender_args += ["--environment_show_image"]
            if srgb_specularity:
                blender_args += ["--srgb_specularity"]
            if rotate_y_to_z:
                blender_args += ["--rotate_y_to_z"]

            cmd = [blender_bin] + blender_args
            cmd_str = " ".join(cmd)
            print(cmd_str)
            subprocess.check_call(cmd)

            im_num = len(eva_output_views)
            copy_images_to_output("relight", im_num, is_hdr=True)

            if save_video:
                write_images_to_video("relight", im_num)

        if mode == "geometry":
            blender_args = [
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                os.path.join(
                    os.path.dirname(__file__),
                    "render_geometry_blender.py",
                ),
                "--",
                "--obj_path",
                mesh_path,
                "--output_dir",
                temp_dir,
                "--cam_pos_path",
                camera_file,
                "--cam_fov_path",
                fov_file,
                "--image_resolution",
                str(image_resolution),
            ]

            if rotate_y_to_z:
                blender_args += ["--rotate_y_to_z"]

            if white_bg:
                blender_args += ["--white_bg"]

            cmd = [blender_bin] + blender_args
            cmd_str = " ".join(cmd)
            print(cmd_str)
            subprocess.check_call(cmd)

            im_num = len(eva_output_views)
            for pass_name in ["normal", "depth"]:
                for n in range(0, im_num):
                    exr_path = os.path.join(temp_dir, "%03d_%s.exr" % (n, pass_name))
                    exr_output_path = os.path.join(
                        output_dir,
                        "%03d_%s.exr" % (eva_output_views[n], pass_name),
                    )
                    pathmgr.copy_from_local(exr_path, exr_output_path, overwrite=True)


def render_images_from_mesh_multienv(
    path,
    prediction_type,
    blender_bin,
    camera_arr,
    eva_output_views,
    output_dir,
    env_paths=None,
    env_mean=None,
    mode="texture",
    show_bg=False,
    camera_center_coord=False,
    image_resolution=512,
    srgb_specularity=False,
    rotate_y_to_z=False,
    white_bg=False,
):
    mesh_path = pathmgr.get_local_path(os.path.join(path, "mesh_uv.obj"), force=True)
    if prediction_type == "rgb" or prediction_type == "both":
        rgb_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_rgb.png"), force=True
        )
    if prediction_type == "brdf" or prediction_type == "both":
        albedo_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_albedo.png"),
            force=True,
        )
        roughness_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_roughness.png"),
            force=True,
        )
        metallic_tex_path = pathmgr.get_local_path(
            os.path.join(path, "mesh_uv_metallic.png"),
            force=True,
        )

    if env_paths is not None:
        env_files = []
        for env_path in env_paths:
            env_file = pathmgr.get_local_path(env_path, force=True)
            if env_mean is not None:
                env = cv2.imread(env_file, -1)
                scale = env_mean / np.mean(env)
                env = env * scale
                cv2.imwrite(env_file, env)
            env_files.append(env_file)
    else:
        raise ValueError("Multi-environment maps required")

    with tempfile.TemporaryDirectory() as temp_dir:

        def copy_images_to_output(name, n, is_hdr=False):
            if is_hdr:
                exr_path = os.path.join(temp_dir, "000_%s.exr" % name)
                exr_output_path = os.path.join(
                    output_dir, "%03d_%s.exr" % (eva_output_views[n], name)
                )
                pathmgr.copy_from_local(exr_path, exr_output_path, overwrite=True)
                png_path = os.path.join(temp_dir, "000_%s.png" % name)
                convert_exr_to_png(exr_path, png_path)
                png_output_path = os.path.join(
                    output_dir, "%03d_%s.png" % (eva_output_views[n], name)
                )
                pathmgr.copy_from_local(png_path, png_output_path, overwrite=True)
            else:
                im_path = os.path.join(temp_dir, "000_%s.png" % name)
                im_output_path = os.path.join(
                    output_dir, "%03d_%s.png" % (eva_output_views[n], name)
                )
                pathmgr.copy_from_local(im_path, im_output_path, overwrite=True)

        camera_file = os.path.join(temp_dir, "camera_info.npy")
        fov = camera_arr[0, 16]
        camera_arr = camera_arr[:, :16].reshape(-1, 4, 4)
        camera_arr = camera_arr[:, :3, :]
        if camera_arr.shape[0] != len(env_files):
            raise ValueError("Camera number and environment map numbers mismatch")

        fov_file = os.path.join(temp_dir, "fov.txt")
        with open(fov_file, "w") as fOut:
            fOut.write("%.7f" % fov)

        if mode == "texture" or mode == "both":
            for n in range(0, len(env_files)):
                if camera_center_coord:
                    cam = camera_arr[n]
                    cam = np.concatenate(
                        [cam, np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 4)],
                        axis=0,
                    )
                    inv_cam = np.linalg.inv(cam)
                    inv_rot = inv_cam[:3, :3]
                    inv_trans = inv_cam[:3, 3:4]
                    new_cam = np.concatenate(
                        [
                            np.array(
                                [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
                                dtype=np.float32,
                            ),
                            np.zeros((3, 1), dtype=np.float32),
                        ],
                        axis=1,
                    )
                    np.save(camera_file, new_cam)
                    mesh = trimesh.load(mesh_path)
                    vertices = mesh.vertices
                    vertices = np.matmul(inv_rot, vertices.transpose(1, 0)) + inv_trans
                    mesh.vertices = vertices.transpose(1, 0)
                    mesh_new_path = os.path.join(temp_dir, "mesh_uv_new.obj")
                    mesh.export(mesh_new_path)
                else:
                    np.save(camera_file, camera_arr[n])

                blender_args = [
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    os.path.join(
                        os.path.dirname(__file__),
                        "render_pass_blender.py",
                    ),
                    "--",
                    "--obj_path",
                    mesh_path if not camera_center_coord else mesh_new_path,
                    "--output_dir",
                    temp_dir,
                    "--cam_pos_path",
                    camera_file,
                    "--cam_fov_path",
                    fov_file,
                    "--image_resolution",
                    str(image_resolution),
                ]

                if prediction_type == "rgb" or prediction_type == "both":
                    blender_args += ["--rgb_image_path", rgb_tex_path]
                if prediction_type == "brdf" or prediction_type == "both":
                    blender_args += ["--albedo_image_path", albedo_tex_path]
                    blender_args += ["--roughness_image_path", roughness_tex_path]
                    blender_args += ["--metallic_image_path", metallic_tex_path]

                if show_bg and env_path is not None:
                    blender_args += ["--environment_map_path", env_files[n]]
                    blender_args += ["--environment_show_image"]

                if rotate_y_to_z:
                    blender_args += ["--rotate_y_to_z"]

                cmd = [blender_bin] + blender_args
                cmd_str = " ".join(cmd)
                print(cmd_str)
                subprocess.check_call(cmd)

                if prediction_type == "rgb" or prediction_type == "both":
                    copy_images_to_output("rgb", n)

                if prediction_type == "brdf" or prediction_type == "both":
                    copy_images_to_output("albedo", n)
                    copy_images_to_output("roughness", n)
                    copy_images_to_output("metallic", n)

        if mode == "relighting" or mode == "both":
            if prediction_type != "brdf" and prediction_type != "both":
                raise ValueError("relighting mode requires brdf prediction")

            if env_path is None:
                raise ValueError("relighting mode requires environment map")

            for n in range(0, len(env_files)):
                if camera_center_coord:
                    cam = camera_arr[n]
                    cam = np.concatenate(
                        [cam, np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 4)],
                        axis=0,
                    )
                    inv_cam = np.linalg.inv(cam)
                    new_cam = np.concatenate(
                        [
                            np.array(
                                [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
                                dtype=np.float32,
                            ),
                            np.zeros((3, 1), dtype=np.float32),
                        ],
                        axis=1,
                    )
                    np.save(camera_file, new_cam)

                    transform = np.matmul(
                        np.concatenate(
                            [
                                new_cam,
                                np.array([0, 0, 0, 1], dtype=np.float32).reshape(1, 4),
                            ],
                            axis=0,
                        ),
                        inv_cam,
                    )
                    inv_rot = transform[:3, :3]
                    inv_trans = transform[:3, 3:4]
                    mesh = trimesh.load(mesh_path)
                    vertices = mesh.vertices
                    vertices = np.matmul(inv_rot, vertices.transpose(1, 0)) + inv_trans
                    mesh.vertices = vertices.transpose(1, 0)
                    mesh_new_path = os.path.join(temp_dir, "mesh_uv_new.obj")
                    mesh.export(mesh_new_path)
                else:
                    np.save(camera_file, camera_arr[n])

                blender_args = [
                    "--background",
                    "--python-exit-code",
                    "1",
                    "--python",
                    os.path.join(
                        os.path.dirname(__file__),
                        "render_pbr_blender.py",
                    ),
                    "--",
                    "--obj_path",
                    mesh_path if not camera_center_coord else mesh_new_path,
                    "--output_dir",
                    temp_dir,
                    "--cam_pos_path",
                    camera_file,
                    "--cam_fov_path",
                    fov_file,
                    "--image_resolution",
                    str(image_resolution),
                ]

                blender_args += ["--albedo_image_path", albedo_tex_path]
                blender_args += ["--roughness_image_path", roughness_tex_path]
                blender_args += ["--metallic_image_path", metallic_tex_path]
                blender_args += ["--environment_map_path", env_files[n]]
                if show_bg:
                    blender_args += ["--environment_show_image"]
                if srgb_specularity:
                    blender_args += ["--srgb_specularity"]
                if rotate_y_to_z:
                    blender_args += ["--rotate_y_to_z"]

                cmd = [blender_bin] + blender_args
                cmd_str = " ".join(cmd)
                print(cmd_str)
                subprocess.check_call(cmd)

                copy_images_to_output("relight", n, is_hdr=True)

        if mode == "geometry":
            np.save(camera_file, camera_arr)

            blender_args = [
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                os.path.join(
                    os.path.dirname(__file__),
                    "render_geometry_blender.py",
                ),
                "--",
                "--obj_path",
                mesh_path if not camera_center_coord else mesh_new_path,
                "--output_dir",
                temp_dir,
                "--cam_pos_path",
                camera_file,
                "--cam_fov_path",
                fov_file,
                "--image_resolution",
                str(image_resolution),
            ]

            if rotate_y_to_z:
                blender_args += ["--rotate_y_to_z"]

            if white_bg:
                blender_args += ["--white_bg"]

            cmd = [blender_bin] + blender_args
            cmd_str = " ".join(cmd)
            print(cmd_str)
            subprocess.check_call(cmd)

            im_num = len(eva_output_views)
            for pass_name in ["normal", "depth"]:
                for n in range(0, im_num):
                    exr_path = os.path.join(temp_dir, "%03d_%s.exr" % (n, pass_name))
                    exr_output_path = os.path.join(
                        output_dir,
                        "%03d_%s.exr" % (eva_output_views[n], pass_name),
                    )
                    pathmgr.copy_from_local(exr_path, exr_output_path, overwrite=True)


def compute_mesh_textures(
    path,
    blender_bin,
    volume,
    volsdf,
    output_path,
    output_texture_res,
    prediction_type,
    save_texture=False,
    overwrite_mesh=True,
    auto_cast_dtype=torch.bfloat16,
    init_cam_rot=None,
):
    local_mesh_path = pathmgr.get_local_path(path, force=True)
    output_dir = os.path.dirname(output_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        if overwrite_mesh or not pathmgr.isfile(output_path):
            local_mesh_output_path = os.path.join(temp_dir, "mesh_uv.obj")
            blender_args = [
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                os.path.join(
                    os.path.dirname(__file__),
                    "uv_blender.py",
                ),
                "--",
                local_mesh_path,
                local_mesh_output_path,
            ]

            # command for process
            cmd = [blender_bin] + blender_args
            cmd_str = " ".join(cmd)
            print(cmd_str)
            subprocess.check_call(cmd)
            pathmgr.copy_from_local(local_mesh_output_path, output_path, overwrite=True)
        else:
            local_mesh_output_path = pathmgr.get_local_path(output_path, force=True)

        """ load unwrapped mesh """
        if save_texture:
            mesh = trimesh.load(local_mesh_output_path)
            # Use float32 for pytorch3d operations (doesn't support bfloat16)
            vert = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
            face = torch.LongTensor(mesh.faces).cuda()

            uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device="cuda")
            # compute xyz map
            vert_uv = torch.cat([uv * 2 - 1, torch.ones_like(uv[:, [0]])], -1)
            vert_uv[..., 0] *= -1
            mesh_torch = pytorch3d.structures.Meshes(
                verts=vert_uv[None], faces=face[None]
            )
            pix_to_face, zbuf, bary_coords, dists = (
                pytorch3d.renderer.mesh.rasterize_meshes(
                    mesh_torch,
                    image_size=(output_texture_res, output_texture_res),
                    faces_per_pixel=1,
                    perspective_correct=True,
                )
            )
            frag = pytorch3d.renderer.mesh.rasterizer.Fragments(
                pix_to_face, zbuf, bary_coords, dists
            )
            mesh_torch.textures = pytorch3d.renderer.TexturesVertex(
                verts_features=vert[None]
            )
            tex_xyz = mesh_torch.sample_textures(frag).squeeze()
            tex_mask = (pix_to_face != -1).squeeze()
            tex_xyz[~tex_mask] = tex_xyz[tex_mask].amin(0) - 1
            local_texture_mask_path = os.path.join(temp_dir, "mesh_uv_mask.png")
            texture_mask_output_path = os.path.join(output_dir, "mesh_uv_mask.png")
            tex_mask_output = (
                tex_mask.cpu()
                .detach()
                .numpy()
                .reshape(output_texture_res, output_texture_res)
            )
            tex_mask_output = (tex_mask_output * 255).astype(np.uint8)
            cv2.imwrite(local_texture_mask_path, tex_mask_output)
            pathmgr.copy_from_local(
                local_texture_mask_path, texture_mask_output_path, overwrite=True
            )

            valid_xyz = tex_xyz[tex_mask].reshape(-1, 3)
            # Cast to volume dtype for model inference
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                    valid_xyz = valid_xyz.to(auto_cast_dtype)
                    # Apply inverse rotation to transform from canonical space back to volume space
                    if init_cam_rot is not None:
                        valid_xyz = torch.matmul(
                            valid_xyz,
                            init_cam_rot.to(valid_xyz.dtype).to(valid_xyz.device),
                        )
                    pred = volsdf(
                        valid_xyz,
                        volume,
                        mode="image",
                    )
            # Convert prediction to float32 for texture output processing
            pred = pred.float()

            if prediction_type == "rgb" or prediction_type == "both":
                local_texture_path = os.path.join(temp_dir, "mesh_uv_rgb.png")
                texture_output_path = os.path.join(output_dir, "mesh_uv_rgb.png")
                rgb = (
                    torch.zeros(output_texture_res, output_texture_res, 3).cuda() + 0.5
                )

                rgb = rgb.reshape(-1, 3)
                rgb[tex_mask.reshape(-1), :] = pred[:, :3]
                rgb = rgb.reshape(output_texture_res, output_texture_res, 3)
                rgb = rgb.detach().cpu().numpy()
                rgb = (255 * rgb).astype(np.uint8)
                rgb_inpainted = cv2.inpaint(
                    rgb, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_texture_path, rgb_inpainted[:, :, ::-1])
                pathmgr.copy_from_local(
                    local_texture_path, texture_output_path, overwrite=True
                )
                with pathmgr.open(os.path.join(output_dir, "mesh_uv.mtl"), "w") as fOut:
                    fOut.write("newmtl Default_OBJ\n")
                    fOut.write("Kd 1.000 1.000 1.000\n")
                    fOut.write("Tr 0.000\n")
                    fOut.write("map_Kd mesh_uv_rgb.png")

            if prediction_type == "brdf" or prediction_type == "both":
                local_albedo_path = os.path.join(temp_dir, "mesh_uv_albedo.png")
                albedo_output_path = os.path.join(output_dir, "mesh_uv_albedo.png")
                local_roughness_path = os.path.join(temp_dir, "mesh_uv_roughness.png")
                roughness_output_path = os.path.join(
                    output_dir, "mesh_uv_roughness.png"
                )
                local_metallic_path = os.path.join(temp_dir, "mesh_uv_metallic.png")
                metallic_output_path = os.path.join(output_dir, "mesh_uv_metallic.png")

                albedo = (
                    torch.zeros(output_texture_res, output_texture_res, 3).cuda() + 0.5
                )
                roughness = (
                    torch.zeros(output_texture_res, output_texture_res, 1).cuda() + 0.5
                )
                metallic = (
                    torch.zeros(output_texture_res, output_texture_res, 1).cuda() + 0.5
                )

                albedo = albedo.reshape(-1, 3)
                roughness = roughness.reshape(-1, 1)
                metallic = metallic.reshape(-1, 1)

                if prediction_type == "both":
                    albedo[tex_mask.reshape(-1), :] = pred[:, 3:6]
                    roughness[tex_mask.reshape(-1), :] = pred[:, 6:7]
                    metallic[tex_mask.reshape(-1), :] = pred[:, 7:8]
                elif prediction_type == "brdf":
                    albedo[tex_mask.reshape(-1), :] = pred[:, 0:3]
                    roughness[tex_mask.reshape(-1), :] = pred[:, 3:4]
                    metallic[tex_mask.reshape(-1), :] = pred[:, 4:5]

                albedo = albedo.reshape(output_texture_res, output_texture_res, 3)
                albedo = albedo.detach().cpu().numpy()
                albedo = (255 * albedo).astype(np.uint8)
                albedo_inpainted = cv2.inpaint(
                    albedo, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_albedo_path, albedo_inpainted[:, :, ::-1])
                pathmgr.copy_from_local(
                    local_albedo_path, albedo_output_path, overwrite=True
                )

                roughness = roughness.reshape(output_texture_res, output_texture_res)
                roughness = roughness.detach().cpu().numpy()
                roughness = (255 * roughness).astype(np.uint8)
                roughness_inpainted = cv2.inpaint(
                    roughness, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_roughness_path, roughness_inpainted)
                pathmgr.copy_from_local(
                    local_roughness_path, roughness_output_path, overwrite=True
                )

                metallic = metallic.reshape(output_texture_res, output_texture_res)
                metallic = metallic.detach().cpu().numpy()
                metallic = (255 * metallic).astype(np.uint8)
                metallic_inpainted = cv2.inpaint(
                    metallic, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_metallic_path, metallic_inpainted)
                pathmgr.copy_from_local(
                    local_metallic_path, metallic_output_path, overwrite=True
                )


def compute_mesh_textures_sparse(
    path,
    blender_bin,
    volume,
    volsdf,
    volsdf_dense,
    output_path,
    output_texture_res,
    prediction_type,
    save_texture=False,
    overwrite_mesh=True,
    auto_cast_dtype=torch.bfloat16,
    init_cam_rot=None,
):
    local_mesh_path = pathmgr.get_local_path(path, force=True)
    output_dir = os.path.dirname(output_path)
    with tempfile.TemporaryDirectory() as temp_dir:
        if overwrite_mesh or not pathmgr.isfile(output_path):
            local_mesh_output_path = os.path.join(temp_dir, "mesh_uv.obj")
            blender_args = [
                "--background",
                "--python-exit-code",
                "1",
                "--python",
                os.path.join(
                    os.path.dirname(__file__),
                    "uv_blender.py",
                ),
                "--",
                local_mesh_path,
                local_mesh_output_path,
            ]

            # command for process
            cmd = [blender_bin] + blender_args
            cmd_str = " ".join(cmd)
            print(cmd_str)
            subprocess.check_call(cmd)
            pathmgr.copy_from_local(local_mesh_output_path, output_path, overwrite=True)
        else:
            local_mesh_output_path = pathmgr.get_local_path(output_path, force=True)

        """ load unwrapped mesh """
        if save_texture:
            mesh = trimesh.load(local_mesh_output_path)
            # Use float32 for pytorch3d operations (doesn't support bfloat16)
            vert = torch.tensor(mesh.vertices, dtype=torch.float32, device="cuda")
            face = torch.LongTensor(mesh.faces).cuda()

            uv = torch.tensor(mesh.visual.uv, dtype=torch.float32, device="cuda")
            # compute xyz map
            vert_uv = torch.cat([uv * 2 - 1, torch.ones_like(uv[:, [0]])], -1)
            vert_uv[..., 0] *= -1
            mesh_torch = pytorch3d.structures.Meshes(
                verts=vert_uv[None], faces=face[None]
            )
            pix_to_face, zbuf, bary_coords, dists = (
                pytorch3d.renderer.mesh.rasterize_meshes(
                    mesh_torch,
                    image_size=(output_texture_res, output_texture_res),
                    faces_per_pixel=1,
                    perspective_correct=True,
                )
            )
            frag = pytorch3d.renderer.mesh.rasterizer.Fragments(
                pix_to_face, zbuf, bary_coords, dists
            )
            mesh_torch.textures = pytorch3d.renderer.TexturesVertex(
                verts_features=vert[None]
            )
            tex_xyz = mesh_torch.sample_textures(frag).squeeze()
            tex_mask = (pix_to_face != -1).squeeze()
            tex_xyz[~tex_mask] = tex_xyz[tex_mask].amin(0) - 1
            local_texture_mask_path = os.path.join(temp_dir, "mesh_uv_mask.png")
            texture_mask_output_path = os.path.join(output_dir, "mesh_uv_mask.png")
            tex_mask_output = (
                tex_mask.cpu()
                .detach()
                .numpy()
                .reshape(output_texture_res, output_texture_res)
            )
            tex_mask_output = (tex_mask_output * 255).astype(np.uint8)
            cv2.imwrite(local_texture_mask_path, tex_mask_output)
            pathmgr.copy_from_local(
                local_texture_mask_path, texture_mask_output_path, overwrite=True
            )

            valid_xyz = tex_xyz[tex_mask].reshape(-1, 3)

            # Apply inverse rotation to transform from canonical space back to volume space
            if init_cam_rot is not None:
                valid_xyz = torch.matmul(
                    valid_xyz, init_cam_rot.to(valid_xyz.dtype).to(valid_xyz.device)
                )

            # Separate positions into sparse and dense regions
            batch_id = volume.get("batch_id", 0)
            # Ensure batch_id is set in volume dict for grid_sample_sparse
            volume["batch_id"] = batch_id
            index_volume = volume["index_volume"][batch_id, :]
            sparse_res = index_volume.shape[0]

            pos_int = torch.clamp(valid_xyz + 0.5, 0, 1) * (sparse_res - 1)
            pos_int = torch.round(pos_int).int()
            index = index_volume[pos_int[:, 2], pos_int[:, 1], pos_int[:, 0]]
            mask_dense = index < 0
            mask_sparse = index >= 0

            positions_dense = valid_xyz[mask_dense, :]
            positions_sparse = valid_xyz[mask_sparse, :]

            with torch.no_grad():
                # Query sparse volsdf for sparse positions
                if positions_sparse.shape[0] > 0:
                    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                        pred_sparse = volsdf(
                            positions_sparse,
                            volume,
                            mode="image",
                        )
                        pred_sparse = pred_sparse.to(torch.float32)

                # Query dense volsdf for dense positions
                if positions_dense.shape[0] > 0:
                    with torch.amp.autocast("cuda", dtype=auto_cast_dtype):
                        pred_dense = volsdf_dense(
                            positions_dense,
                            volume["dense_feat"][batch_id : batch_id + 1, :],
                            mode="image",
                        )
                        pred_dense = pred_dense.to(torch.float32)

                # Merge predictions
                points_num = valid_xyz.shape[0]
                if positions_sparse.shape[0] > 0 and positions_dense.shape[0] > 0:
                    feature_dim = pred_sparse.shape[1]
                    pred = torch.zeros(
                        (points_num, feature_dim),
                        dtype=torch.float32,
                        device=valid_xyz.device,
                    )
                    pred[mask_dense, :] = pred_dense
                    pred[mask_sparse, :] = pred_sparse
                elif positions_dense.shape[0] == 0 and positions_sparse.shape[0] > 0:
                    pred = pred_sparse
                else:
                    pred = pred_dense

            if prediction_type == "rgb" or prediction_type == "both":
                local_texture_path = os.path.join(temp_dir, "mesh_uv_rgb.png")
                texture_output_path = os.path.join(output_dir, "mesh_uv_rgb.png")
                rgb = (
                    torch.zeros(output_texture_res, output_texture_res, 3).cuda() + 0.5
                )

                rgb = rgb.reshape(-1, 3)
                rgb[tex_mask.reshape(-1), :] = pred[:, :3]
                rgb = rgb.reshape(output_texture_res, output_texture_res, 3)
                rgb = rgb.detach().cpu().numpy()
                rgb = (255 * rgb).astype(np.uint8)
                rgb_inpainted = cv2.inpaint(
                    rgb, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_texture_path, rgb_inpainted[:, :, ::-1])
                pathmgr.copy_from_local(
                    local_texture_path, texture_output_path, overwrite=True
                )
                with pathmgr.open(os.path.join(output_dir, "mesh_uv.mtl"), "w") as fOut:
                    fOut.write("newmtl Default_OBJ\n")
                    fOut.write("Kd 1.000 1.000 1.000\n")
                    fOut.write("Tr 0.000\n")
                    fOut.write("map_Kd mesh_uv_rgb.png")

            if prediction_type == "brdf" or prediction_type == "both":
                local_albedo_path = os.path.join(temp_dir, "mesh_uv_albedo.png")
                albedo_output_path = os.path.join(output_dir, "mesh_uv_albedo.png")
                local_roughness_path = os.path.join(temp_dir, "mesh_uv_roughness.png")
                roughness_output_path = os.path.join(
                    output_dir, "mesh_uv_roughness.png"
                )
                local_metallic_path = os.path.join(temp_dir, "mesh_uv_metallic.png")
                metallic_output_path = os.path.join(output_dir, "mesh_uv_metallic.png")

                albedo = (
                    torch.zeros(output_texture_res, output_texture_res, 3).cuda() + 0.5
                )
                roughness = (
                    torch.zeros(output_texture_res, output_texture_res, 1).cuda() + 0.5
                )
                metallic = (
                    torch.zeros(output_texture_res, output_texture_res, 1).cuda() + 0.5
                )

                albedo = albedo.reshape(-1, 3)
                roughness = roughness.reshape(-1, 1)
                metallic = metallic.reshape(-1, 1)

                if prediction_type == "both":
                    albedo[tex_mask.reshape(-1), :] = pred[:, 3:6]
                    roughness[tex_mask.reshape(-1), :] = pred[:, 6:7]
                    metallic[tex_mask.reshape(-1), :] = pred[:, 7:8]
                elif prediction_type == "brdf":
                    albedo[tex_mask.reshape(-1), :] = pred[:, 0:3]
                    roughness[tex_mask.reshape(-1), :] = pred[:, 3:4]
                    metallic[tex_mask.reshape(-1), :] = pred[:, 4:5]

                albedo = albedo.reshape(output_texture_res, output_texture_res, 3)
                albedo = albedo.detach().cpu().numpy()
                albedo = (255 * albedo).astype(np.uint8)
                albedo_inpainted = cv2.inpaint(
                    albedo, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_albedo_path, albedo_inpainted[:, :, ::-1])
                pathmgr.copy_from_local(
                    local_albedo_path, albedo_output_path, overwrite=True
                )

                roughness = roughness.reshape(output_texture_res, output_texture_res)
                roughness = roughness.detach().cpu().numpy()
                roughness = (255 * roughness).astype(np.uint8)
                roughness_inpainted = cv2.inpaint(
                    roughness, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_roughness_path, roughness_inpainted)
                pathmgr.copy_from_local(
                    local_roughness_path, roughness_output_path, overwrite=True
                )

                metallic = metallic.reshape(output_texture_res, output_texture_res)
                metallic = metallic.detach().cpu().numpy()
                metallic = (255 * metallic).astype(np.uint8)
                metallic_inpainted = cv2.inpaint(
                    metallic, 255 - tex_mask_output, 3, cv2.INPAINT_TELEA
                )
                cv2.imwrite(local_metallic_path, metallic_inpainted)
                pathmgr.copy_from_local(
                    local_metallic_path, metallic_output_path, overwrite=True
                )
