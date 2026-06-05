# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import functools
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def init_process_group(**kwargs):
    if "backend" not in kwargs:
        kwargs["backend"] = "nccl"
    if "init_method" not in kwargs:
        kwargs["init_method"] = "env://"
    dist.init_process_group(**kwargs)


def get_parallel_model(model, device):
    if get_world_size() >= 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device], find_unused_parameters=False
        )
    else:
        raise NotImplementedError
    return model


def synchronize() -> None:
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def get_fsdp_model(
    model,
    device,
    transformer_layer_cls=None,
    mixed_precision_dtype=torch.bfloat16,
    cpu_offload_optimizer: bool = False,
):
    """
    Wrap model with FSDP (Fully Sharded Data Parallel) using FULL_SHARD.

    Shards parameters, gradients, and optimizer states across all GPUs for
    maximum memory savings.

    Args:
        model: The model to wrap with FSDP.
        device: The device ID to use.
        transformer_layer_cls: Optional list of transformer layer classes to wrap.
                               If provided, uses transformer_auto_wrap_policy.
        mixed_precision_dtype: The dtype to use for mixed precision training.
                               Defaults to torch.bfloat16. Set to None to disable mixed precision.
        cpu_offload_optimizer: If True, offloads optimizer states and gradients to CPU.
                               This significantly reduces GPU memory usage but may slow down
                               training due to CPU-GPU data transfers. The parameters remain
                               on GPU for computation. Defaults to False.

    Returns:
        FSDP wrapped model.
    """
    if get_world_size() >= 1:
        if transformer_layer_cls is not None:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=set(transformer_layer_cls),
            )
        else:
            auto_wrap_policy = None

        # Configure mixed precision for FSDP
        # - param_dtype: Parameters are kept in this dtype (bfloat16 for memory efficiency)
        # - reduce_dtype: Gradient reduction uses this dtype (bfloat16 for communication efficiency)
        # - buffer_dtype: Buffers are kept in this dtype (bfloat16 for consistency)
        if mixed_precision_dtype is not None:
            mp_policy = MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=mixed_precision_dtype,
                buffer_dtype=mixed_precision_dtype,
            )
        else:
            mp_policy = None

        # Configure CPU offload for optimizer states
        # When enabled, optimizer states (momentum, variance for Adam) and gradients
        # are kept on CPU. Parameters remain on GPU for computation.
        # This can reduce GPU memory by ~2x the model size (for Adam optimizer states)
        if cpu_offload_optimizer:
            cpu_offload_policy = CPUOffload(offload_params=False)
        else:
            cpu_offload_policy = None

        # FULL_SHARD: Shards parameters, gradients, and optimizer states
        # across all GPUs for maximum memory savings
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload_policy,
            device_id=device,
            use_orig_params=True,
        )
    else:
        raise NotImplementedError
    return model


def get_fsdp_model_frozen(
    model,
    device,
    mixed_precision_dtype=torch.bfloat16,
    cpu_offload=False,
):
    """
    Wrap a frozen (no gradient) model with FSDP for memory-efficient inference.

    This is useful for large pretrained models like DinoV3Encoder that are used
    in eval mode without gradient computation. FSDP shards the model parameters
    across GPUs to reduce per-GPU memory usage.

    Args:
        model: The frozen model to wrap with FSDP. All parameters should have
               requires_grad=False.
        device: The device ID to use.
        mixed_precision_dtype: The dtype to use for mixed precision.
                               Defaults to torch.bfloat16. Set to None to disable.
        cpu_offload: Whether to offload parameters to CPU when not in use.
                     Can further reduce GPU memory but may slow down inference.

    Returns:
        FSDP wrapped frozen model.
    """
    if get_world_size() >= 1:
        # Configure mixed precision for FSDP
        if mixed_precision_dtype is not None:
            mp_policy = MixedPrecision(
                param_dtype=mixed_precision_dtype,
                reduce_dtype=mixed_precision_dtype,
                buffer_dtype=mixed_precision_dtype,
            )
        else:
            mp_policy = None

        # Configure CPU offload if requested
        if cpu_offload:
            cpu_offload_policy = CPUOffload(offload_params=True)
        else:
            cpu_offload_policy = None

        # Use FULL_SHARD for frozen models to maximize memory savings
        # Since there are no gradients, we don't need hybrid sharding
        model = FSDP(
            model,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            cpu_offload=cpu_offload_policy,
            device_id=device,
            use_orig_params=True,
        )
    else:
        raise NotImplementedError
    return model
