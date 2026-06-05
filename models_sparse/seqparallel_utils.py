# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os

import torch
import torch.distributed as dist
from torch.autograd import Function

# Cache for the intra-node process group (created once and reused)
_LOCAL_PROCESS_GROUP = None


class AllToAllSingle(Function):
    """
    Custom autograd function for all-to-all communication that supports gradient descent.
    Uses torch.distributed.all_to_all_single for the underlying communication.

    IMPORTANT: This function handles zero-size splits safely by padding tensors to ensure
    all ranks have non-zero data to exchange. This prevents NCCL hangs that can occur
    when some ranks have zero tokens to send/receive.
    """

    @staticmethod
    def forward(ctx, input_tensor, output_split_sizes, input_split_sizes, group):
        # Save sizes and group for backward pass
        ctx.output_split_sizes = output_split_sizes
        ctx.input_split_sizes = input_split_sizes
        ctx.group = group

        hidden_dim = input_tensor.shape[1]

        # CRITICAL FIX: Handle zero-size splits by padding with 1 token per rank
        # This ensures all ranks have non-zero data to exchange, preventing NCCL hangs.
        # We'll strip the padding after the all_to_all operation.
        has_zero_split = any(s == 0 for s in input_split_sizes) or any(
            s == 0 for s in output_split_sizes
        )

        if has_zero_split:
            # Pad input_split_sizes: add 1 to each zero entry
            padded_input_split_sizes = [max(s, 1) for s in input_split_sizes]
            padded_output_split_sizes = [max(s, 1) for s in output_split_sizes]

            # Calculate how much padding we need to add to input tensor
            input_padding_needed = sum(padded_input_split_sizes) - sum(
                input_split_sizes
            )

            if input_padding_needed > 0:
                # Insert padding at the positions where original split size was 0
                padded_input_parts = []
                offset = 0
                for orig_size, padded_size in zip(
                    input_split_sizes, padded_input_split_sizes
                ):
                    if orig_size > 0:
                        padded_input_parts.append(
                            input_tensor[offset : offset + orig_size]
                        )
                        offset += orig_size
                    if padded_size > orig_size:
                        # Add padding for this rank
                        padded_input_parts.append(
                            torch.zeros(
                                [padded_size - orig_size, hidden_dim],
                                dtype=input_tensor.dtype,
                                device=input_tensor.device,
                            )
                        )
                input_tensor_padded = torch.cat(padded_input_parts, dim=0)
            else:
                input_tensor_padded = input_tensor

            # Create padded output tensor
            padded_output_token_num = sum(padded_output_split_sizes)
            output_tensor_padded = torch.empty(
                [padded_output_token_num, hidden_dim],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

            # Perform all-to-all with padded sizes
            dist.all_to_all_single(
                output_tensor_padded,
                input_tensor_padded.contiguous(),
                output_split_sizes=padded_output_split_sizes,
                input_split_sizes=padded_input_split_sizes,
                group=group,
            )

            # Strip padding from output
            output_parts = []
            offset = 0
            for orig_size, padded_size in zip(
                output_split_sizes, padded_output_split_sizes
            ):
                if orig_size > 0:
                    output_parts.append(
                        output_tensor_padded[offset : offset + orig_size]
                    )
                offset += padded_size

            if output_parts:
                output_tensor = torch.cat(output_parts, dim=0)
            else:
                # All output sizes are zero
                output_tensor = torch.empty(
                    [0, hidden_dim],
                    dtype=input_tensor.dtype,
                    device=input_tensor.device,
                )

            ctx.has_zero_split = True
        else:
            # No zero splits - use original logic
            ctx.has_zero_split = False
            output_token_num = sum(output_split_sizes)
            output_tensor = torch.empty(
                [output_token_num, hidden_dim],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )

            # Perform all-to-all communication
            dist.all_to_all_single(
                output_tensor,
                input_tensor.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )

        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # In backward pass, reverse the all-to-all operation:
        # - output_split_sizes becomes input_split_sizes
        # - input_split_sizes becomes output_split_sizes
        output_split_sizes = ctx.input_split_sizes
        input_split_sizes = ctx.output_split_sizes
        group = ctx.group

        hidden_dim = grad_output.shape[1]

        # Use same zero-split handling as forward
        has_zero_split = any(s == 0 for s in input_split_sizes) or any(
            s == 0 for s in output_split_sizes
        )

        if has_zero_split:
            # Pad split sizes
            padded_input_split_sizes = [max(s, 1) for s in input_split_sizes]
            padded_output_split_sizes = [max(s, 1) for s in output_split_sizes]

            # Calculate padding needed for grad_output (which is the "input" for backward)
            input_padding_needed = sum(padded_input_split_sizes) - sum(
                input_split_sizes
            )

            if input_padding_needed > 0:
                # Create padded grad_output tensor
                padded_input_parts = []
                offset = 0
                for orig_size, padded_size in zip(
                    input_split_sizes, padded_input_split_sizes
                ):
                    if orig_size > 0:
                        padded_input_parts.append(
                            grad_output[offset : offset + orig_size]
                        )
                        offset += orig_size
                    if padded_size > orig_size:
                        padded_input_parts.append(
                            torch.zeros(
                                [padded_size - orig_size, hidden_dim],
                                dtype=grad_output.dtype,
                                device=grad_output.device,
                            )
                        )
                grad_output_padded = torch.cat(padded_input_parts, dim=0)
            else:
                grad_output_padded = grad_output

            # Create padded output tensor for gradients
            padded_output_token_num = sum(padded_output_split_sizes)
            grad_input_padded = torch.empty(
                [padded_output_token_num, hidden_dim],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            # Perform reverse all-to-all with padded sizes
            dist.all_to_all_single(
                grad_input_padded,
                grad_output_padded.contiguous(),
                output_split_sizes=padded_output_split_sizes,
                input_split_sizes=padded_input_split_sizes,
                group=group,
            )

            # Strip padding from grad_input
            grad_parts = []
            offset = 0
            for orig_size, padded_size in zip(
                output_split_sizes, padded_output_split_sizes
            ):
                if orig_size > 0:
                    grad_parts.append(grad_input_padded[offset : offset + orig_size])
                offset += padded_size

            if grad_parts:
                grad_input = torch.cat(grad_parts, dim=0)
            else:
                grad_input = torch.empty(
                    [0, hidden_dim],
                    dtype=grad_output.dtype,
                    device=grad_output.device,
                )
        else:
            # No zero splits - use original logic
            output_token_num = sum(output_split_sizes)
            grad_input = torch.empty(
                [output_token_num, hidden_dim],
                dtype=grad_output.dtype,
                device=grad_output.device,
            )

            # Perform reverse all-to-all communication for gradients
            dist.all_to_all_single(
                grad_input,
                grad_output.contiguous(),
                output_split_sizes=output_split_sizes,
                input_split_sizes=input_split_sizes,
                group=group,
            )

        # Return gradients for each input (None for non-tensor inputs)
        return grad_input, None, None, None


class AllGatherSingle(Function):
    """
    Custom autograd function for all-gather communication that supports gradient descent.
    Forward: all-gather (each rank gets the full tensor from all ranks)
    Backward: reduce-scatter (gradients are summed and scattered back to original ranks)
    """

    @staticmethod
    def forward(ctx, input_tensor, gather_sizes, group):
        """
        Forward pass: all-gather tensors from all ranks.

        Args:
            input_tensor: Local tensor to gather [local_tokens, hidden_dim]
            gather_sizes: List of sizes from each rank (how many tokens each rank contributes)
            group: Process group for communication

        Returns:
            Gathered tensor [total_tokens, hidden_dim] containing data from all ranks
        """
        ctx.gather_sizes = gather_sizes
        ctx.group = group

        world_size = dist.get_world_size(group)

        # Save local size for backward
        ctx.local_size = input_tensor.shape[0]

        # Pad tensors to max size for all_gather (since all_gather requires same size)
        max_size = max(gather_sizes)
        hidden_dim = input_tensor.shape[1]

        if input_tensor.shape[0] < max_size:
            pad_size = max_size - input_tensor.shape[0]
            padding = torch.zeros(
                [pad_size, hidden_dim],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            input_padded = torch.cat([input_tensor.contiguous(), padding], dim=0)
        else:
            input_padded = input_tensor.contiguous()

        # All-gather the padded tensors
        gathered_list = [
            torch.empty(
                [max_size, hidden_dim],
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            for _ in range(world_size)
        ]
        dist.all_gather(gathered_list, input_padded, group=group)

        # Slice each gathered tensor to its original size and concatenate
        result_list = []
        for i, gathered in enumerate(gathered_list):
            result_list.append(gathered[: gather_sizes[i]])

        output_tensor = torch.cat(result_list, dim=0)
        return output_tensor

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: reduce-scatter the gradients.

        The gradient of all-gather is reduce-scatter:
        - Each rank receives the sum of gradients corresponding to its original tokens
        """
        gather_sizes = ctx.gather_sizes
        group = ctx.group

        local_rank = dist.get_rank(group)
        hidden_dim = grad_output.shape[1]

        # Split grad_output according to gather_sizes
        grad_splits = torch.split(grad_output, gather_sizes, dim=0)

        # Pad to max size for reduce_scatter
        max_size = max(gather_sizes)
        grad_padded_list = []
        for _, grad_chunk in enumerate(grad_splits):
            if grad_chunk.shape[0] < max_size:
                pad_size = max_size - grad_chunk.shape[0]
                padding = torch.zeros(
                    [pad_size, hidden_dim],
                    dtype=grad_output.dtype,
                    device=grad_output.device,
                )
                grad_padded = torch.cat([grad_chunk.contiguous(), padding], dim=0)
            else:
                grad_padded = grad_chunk.contiguous()
            grad_padded_list.append(grad_padded)

        # Output tensor for reduce_scatter
        grad_input_padded = torch.empty(
            [max_size, hidden_dim],
            dtype=grad_output.dtype,
            device=grad_output.device,
        )

        # Reduce-scatter: sum gradients and scatter to each rank
        dist.reduce_scatter(
            grad_input_padded, grad_padded_list, op=dist.ReduceOp.SUM, group=group
        )

        # Slice to original size
        grad_input = grad_input_padded[: gather_sizes[local_rank]]

        return grad_input, None, None


def get_local_process_group(num_gpus_per_node=None):
    global _LOCAL_PROCESS_GROUP

    if _LOCAL_PROCESS_GROUP is not None:
        return _LOCAL_PROCESS_GROUP

    if not dist.is_initialized():
        return None

    # Determine the number of GPUs per node
    if num_gpus_per_node is None:
        num_gpus_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", 8))

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    # If only one node or world_size <= num_gpus_per_node, no need for sub-groups
    if world_size <= num_gpus_per_node:
        _LOCAL_PROCESS_GROUP = dist.group.WORLD
        return _LOCAL_PROCESS_GROUP

    # Calculate number of nodes
    num_nodes = world_size // num_gpus_per_node

    # Create process groups for each node
    # All ranks must participate in new_group() calls, even if they're not in the group
    for node_idx in range(num_nodes):
        start_rank = node_idx * num_gpus_per_node
        end_rank = start_rank + num_gpus_per_node
        ranks_on_node = list(range(start_rank, end_rank))

        # Create the group (all ranks must call this collectively)
        group = dist.new_group(ranks=ranks_on_node)

        # Store the group if this rank belongs to it
        if rank in ranks_on_node:
            _LOCAL_PROCESS_GROUP = group

    return _LOCAL_PROCESS_GROUP


def all_gather_fixed_size(
    data: torch.Tensor, local_only=True, num_gpus_per_node=None
) -> list:
    if not dist.is_initialized():
        return [data]

    # Get the appropriate process group
    if local_only:
        group = get_local_process_group(num_gpus_per_node)
    else:
        group = dist.group.WORLD  # Use default world group
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    data = data.contiguous()

    # Create placeholder tensors for gathering
    gathered_list = [torch.zeros_like(data) for _ in range(world_size)]

    # Perform all_gather within the specified group
    dist.all_gather(gathered_list, data, group=group)

    return gathered_list


def all_gather_various_size(
    data: torch.Tensor,
    dim: int = 0,
    local_only: bool = True,
    num_gpus_per_node=None,
) -> list:
    if not dist.is_initialized():
        return [data]

    # Get the appropriate process group
    if local_only:
        group = get_local_process_group(num_gpus_per_node)
    else:
        group = dist.group.WORLD  # Use default world group

    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    data = data.contiguous()

    # Step 1: Gather sizes from all ranks
    local_size = torch.tensor([data.shape[dim]], dtype=torch.long, device=data.device)
    size_list = [
        torch.zeros(1, dtype=torch.long, device=data.device) for _ in range(world_size)
    ]
    dist.all_gather(size_list, local_size, group=group)
    sizes = [int(s.item()) for s in size_list]
    max_size = max(sizes)

    # Step 2: Pad local tensor to max_size along the specified dimension
    if data.shape[dim] < max_size:
        pad_size = max_size - data.shape[dim]
        pad_shape = list(data.shape)
        pad_shape[dim] = pad_size
        padding = torch.zeros(pad_shape, dtype=data.dtype, device=data.device)
        data_padded = torch.cat([data, padding], dim=dim)
    else:
        data_padded = data

    # Step 3: All-gather the padded tensors
    gathered_list = [torch.zeros_like(data_padded) for _ in range(world_size)]
    dist.all_gather(gathered_list, data_padded, group=group)

    # Step 4: Slice each gathered tensor to its original size
    result_list = []
    for i, gathered in enumerate(gathered_list):
        slices = [slice(None)] * gathered.ndim
        slices[dim] = slice(0, sizes[i])
        result_list.append(gathered[tuple(slices)])

    return result_list


def get_local_rank(local_only=True, num_gpus_per_node=None):
    # Get the appropriate process group
    if local_only:
        group = get_local_process_group(num_gpus_per_node)
    else:
        group = dist.group.WORLD  # Use default world group

    local_rank = dist.get_rank(group)
    return local_rank


def compute_tokens_mapping(
    cu_seqblocks, cu_block_include_tokens, local_only=True, num_gpus_per_node=None
):
    import torch.distributed as dist

    cu_seqblocks_list = all_gather_fixed_size(
        cu_seqblocks, local_only=local_only, num_gpus_per_node=num_gpus_per_node
    )
    cu_block_include_tokens_list = all_gather_various_size(
        cu_block_include_tokens,
        local_only=local_only,
        num_gpus_per_node=num_gpus_per_node,
    )
    local_world_size = len(cu_seqblocks_list)

    with torch.no_grad():
        # Build global_cu_seqblocks: cumulative block counts per batch across all GPUs
        # cu_seqblocks_list[n] has format [0, blocks_in_batch_0, blocks_in_batch_0+blocks_in_batch_1, ...]
        # We skip [1:] to avoid duplicate entries (the 0 from each rank would duplicate the last cumsum)
        global_cu_seqblocks = cu_seqblocks_list[0]
        for n in range(1, len(cu_seqblocks_list)):
            global_cu_seqblocks = torch.cat(
                [
                    global_cu_seqblocks,
                    cu_seqblocks_list[n][1:] + global_cu_seqblocks[-1],
                ],
                dim=0,
            )

        # Build global_cu_block_include_tokens: cumulative token counts per block across all GPUs
        # cu_block_include_tokens_list[n] has format [0, tokens_in_block_0, tokens_in_block_0+tokens_in_block_1, ...]
        # We must skip [1:] to avoid duplicate entries (the 0 from each rank would duplicate the last cumsum)
        global_cu_block_include_tokens = cu_block_include_tokens_list[0]
        for n in range(1, len(cu_block_include_tokens_list)):
            global_cu_block_include_tokens = torch.cat(
                [
                    global_cu_block_include_tokens,
                    cu_block_include_tokens_list[n][1:]
                    + global_cu_block_include_tokens[-1],
                ],
                dim=0,
            )

    # Get the appropriate process group
    if local_only:
        group = get_local_process_group(num_gpus_per_node)
    else:
        group = dist.group.WORLD  # Use default world group

    local_rank = dist.get_rank(group)
    group_size = dist.get_world_size(group)
    assert group_size == local_world_size

    total_block_num = global_cu_seqblocks[-1].item()
    total_token_num = global_cu_block_include_tokens[-1].item()

    # Distribute blocks to balance token counts across ranks
    # We use a greedy approach: assign blocks sequentially, moving to the next rank
    # when the current rank's token count exceeds or reaches the target
    block_num_for_each_rank = []
    current_block_idx = 0

    for rank_idx in range(group_size):
        if rank_idx == group_size - 1:
            # Last rank gets all remaining blocks
            remaining_blocks = total_block_num - current_block_idx
            block_num_for_each_rank.append(remaining_blocks)
        else:
            # Calculate how many tokens we should target for this rank
            remaining_ranks = group_size - rank_idx
            remaining_tokens = (
                total_token_num
                - global_cu_block_include_tokens[current_block_idx].item()
            )
            target_for_this_rank = remaining_tokens / remaining_ranks

            rank_token_count = 0
            rank_block_count = 0

            while current_block_idx < total_block_num:
                # Get token count for the next block
                next_block_tokens = (
                    global_cu_block_include_tokens[current_block_idx + 1].item()
                    - global_cu_block_include_tokens[current_block_idx].item()
                )

                # Check if adding this block would exceed the target
                # We add the block if:
                # 1. We haven't added any blocks yet (ensure at least one block per rank)
                # 2. Adding this block gets us closer to the target than not adding it
                if rank_block_count == 0:
                    # Must have at least one block
                    rank_token_count += next_block_tokens
                    rank_block_count += 1
                    current_block_idx += 1
                else:
                    # Check if adding this block gets us closer to target
                    distance_without = abs(rank_token_count - target_for_this_rank)
                    distance_with = abs(
                        rank_token_count + next_block_tokens - target_for_this_rank
                    )

                    if distance_with <= distance_without:
                        rank_token_count += next_block_tokens
                        rank_block_count += 1
                        current_block_idx += 1
                    else:
                        # Stop adding blocks to this rank
                        break

            block_num_for_each_rank.append(rank_block_count)

    # For debugging purposes
    token_num_for_each_rank = []
    start_block_id = 0
    for n in range(group_size):
        end_block_id = start_block_id + block_num_for_each_rank[n]
        token_num_for_each_rank.append(
            global_cu_block_include_tokens[end_block_id].item()
            - global_cu_block_include_tokens[start_block_id].item()
        )
        start_block_id = end_block_id
    print(f"block_num_for_each_rank: {block_num_for_each_rank}")
    print(f"token_num_for_each_rank: {token_num_for_each_rank}")

    cu_seqblocks_ranks_original = [0]
    for n in range(0, group_size):
        cu_seqblocks_ranks_original.append(
            cu_seqblocks_ranks_original[-1] + cu_seqblocks_list[n][-1].item()
        )
    cu_seqblocks_ranks_distributed = [0]
    for n in range(0, group_size):
        cu_seqblocks_ranks_distributed.append(
            cu_seqblocks_ranks_distributed[-1] + block_num_for_each_rank[n]
        )

    # Compute output_list: how many tokens the current rank will RECEIVE from each rank
    # after redistribution (i.e., output_split_sizes for all_to_all_single)
    # This computes the intersection between blocks originally on rank n and blocks
    # that should be on local_rank after even distribution
    output_block_list, output_list = [], []
    for n in range(group_size):
        # Blocks originally on rank n: [cu_seqblocks_ranks_original[n], cu_seqblocks_ranks_original[n+1])
        # Blocks that local_rank should have after distribution:
        # [cu_seqblocks_ranks_distributed[local_rank], cu_seqblocks_ranks_distributed[local_rank+1])
        original_start = cu_seqblocks_ranks_original[n]
        original_end = cu_seqblocks_ranks_original[n + 1]
        distributed_start = cu_seqblocks_ranks_distributed[local_rank]
        distributed_end = cu_seqblocks_ranks_distributed[local_rank + 1]

        # Find intersection of block ranges
        intersect_start = max(original_start, distributed_start)
        intersect_end = min(original_end, distributed_end)
        num_blocks_from_n = max(0, intersect_end - intersect_start)
        output_block_list.append(num_blocks_from_n)

        # Convert blocks to tokens using cu_block_include_tokens
        if num_blocks_from_n > 0:
            # Get the token range for these blocks from rank n's cu_block_include_tokens
            # Local block indices within rank n
            local_block_start = intersect_start - original_start
            local_block_end = intersect_end - original_start
            cu_tokens_n = cu_block_include_tokens_list[n]
            token_start = cu_tokens_n[local_block_start].item()
            token_end = cu_tokens_n[local_block_end].item()
            num_tokens = token_end - token_start
        else:
            num_tokens = 0
        output_list.append(num_tokens)

    # Compute input_list: how many tokens the current rank will SEND to each rank
    # after redistribution (i.e., input_split_sizes for all_to_all_single)
    # This computes the intersection between blocks originally on local_rank and blocks
    # that should be on rank n after even distribution
    input_block_list, input_list = [], []
    for n in range(group_size):
        # Blocks originally on local_rank: [cu_seqblocks_ranks_original[local_rank], cu_seqblocks_ranks_original[local_rank+1])
        # Blocks that rank n should have after distribution:
        # [cu_seqblocks_ranks_distributed[n], cu_seqblocks_ranks_distributed[n+1])
        original_start = cu_seqblocks_ranks_original[local_rank]
        original_end = cu_seqblocks_ranks_original[local_rank + 1]
        distributed_start = cu_seqblocks_ranks_distributed[n]
        distributed_end = cu_seqblocks_ranks_distributed[n + 1]

        # Find intersection of block ranges
        intersect_start = max(original_start, distributed_start)
        intersect_end = min(original_end, distributed_end)
        num_blocks_to_n = max(0, intersect_end - intersect_start)
        input_block_list.append(num_blocks_to_n)

        # Convert blocks to tokens using cu_block_include_tokens
        if num_blocks_to_n > 0:
            # Get the token range for these blocks from local_rank's cu_block_include_tokens
            # Local block indices within local_rank
            local_block_start = intersect_start - original_start
            local_block_end = intersect_end - original_start
            cu_tokens_local = cu_block_include_tokens_list[local_rank]
            token_start = cu_tokens_local[local_block_start].item()
            token_end = cu_tokens_local[local_block_end].item()
            num_tokens = token_end - token_start
        else:
            num_tokens = 0
        input_list.append(num_tokens)

    # Build local_cu_block_include_tokens: cumulative token counts for blocks assigned to local_rank
    # We collect token counts for each block in the distributed range from global_cu_block_include_tokens
    local_block_token_counts = []

    for n in range(group_size):
        if output_block_list[n] == 0:
            continue

        # Get the block range from rank n that local_rank will receive
        original_start = cu_seqblocks_ranks_original[n]
        original_end = cu_seqblocks_ranks_original[n + 1]
        distributed_start = cu_seqblocks_ranks_distributed[local_rank]
        distributed_end = cu_seqblocks_ranks_distributed[local_rank + 1]

        intersect_start = max(original_start, distributed_start)
        intersect_end = min(original_end, distributed_end)

        # Local block indices within rank n
        local_block_start_in_n = intersect_start - original_start
        local_block_end_in_n = intersect_end - original_start

        cu_tokens_n = cu_block_include_tokens_list[n]

        # Extract token counts for each block in this range
        for block_idx in range(local_block_start_in_n, local_block_end_in_n):
            token_count = (
                cu_tokens_n[block_idx + 1].item() - cu_tokens_n[block_idx].item()
            )
            local_block_token_counts.append(token_count)

    # Build cumulative token counts
    local_cu_block_include_tokens_list = [0]
    for token_count in local_block_token_counts:
        local_cu_block_include_tokens_list.append(
            local_cu_block_include_tokens_list[-1] + token_count
        )

    local_cu_block_include_tokens = torch.tensor(
        local_cu_block_include_tokens_list,
        dtype=global_cu_block_include_tokens.dtype,
        device=global_cu_block_include_tokens.device,
    )

    # Build local_cu_seqlens: cumulative token counts for each batch on local_rank
    # This is used for spatial_sparse_attention_mid after all_gather of kv.
    # Requirements:
    # 1. Length should equal total_batch_size across all GPUs + 1 (for all_gather scenario)
    # 2. Should only count q tokens distributed to the current GPU (local rank)
    #
    # After all_gather of kv, each GPU has ALL kv tokens, but only LOCAL q tokens.
    # So local_cu_seqlens tracks: for each global batch, how many q tokens are on this GPU.
    #
    # We iterate through all sequences from all ranks and compute local q token counts.

    # Block range distributed to local_rank after redistribution
    local_distributed_block_start = cu_seqblocks_ranks_distributed[local_rank]
    local_distributed_block_end = cu_seqblocks_ranks_distributed[local_rank + 1]

    local_cu_seqlens_list = [0]
    cumulative_local_tokens = 0

    # Iterate through each rank's sequences
    for rank_n in range(group_size):
        num_seqs_on_rank_n = len(cu_seqblocks_list[rank_n]) - 1
        cu_seqblocks_n = cu_seqblocks_list[rank_n]
        cu_tokens_n = cu_block_include_tokens_list[rank_n]

        # Original block offset for rank_n (global block index of first block on rank_n)
        original_block_offset = cu_seqblocks_ranks_original[rank_n]

        for seq_idx in range(num_seqs_on_rank_n):
            # Global block range for this sequence
            seq_block_start_local = cu_seqblocks_n[seq_idx].item()
            seq_block_end_local = cu_seqblocks_n[seq_idx + 1].item()
            seq_block_start_global = original_block_offset + seq_block_start_local
            seq_block_end_global = original_block_offset + seq_block_end_local

            # Intersection with blocks distributed to local_rank
            intersect_start = max(seq_block_start_global, local_distributed_block_start)
            intersect_end = min(seq_block_end_global, local_distributed_block_end)
            num_local_blocks = max(0, intersect_end - intersect_start)

            if num_local_blocks > 0:
                # Compute token counts using local indexing within rank_n's cu_block_include_tokens
                local_block_start_in_n = intersect_start - original_block_offset
                local_block_end_in_n = intersect_end - original_block_offset
                token_start = cu_tokens_n[local_block_start_in_n].item()
                token_end = cu_tokens_n[local_block_end_in_n].item()
                num_tokens = token_end - token_start
            else:
                num_tokens = 0

            cumulative_local_tokens += num_tokens
            local_cu_seqlens_list.append(cumulative_local_tokens)

    local_cu_seqlens = torch.tensor(
        local_cu_seqlens_list,
        dtype=torch.int32,
        device=global_cu_seqblocks.device,
    )

    # Compute gather sizes for all_gather operation
    # gather_sizes[n] = number of tokens on rank n after redistribution
    # compressed_gather_sizes[n] = number of blocks on rank n = number of compressed tokens
    gather_sizes = []
    for n in range(group_size):
        block_start = cu_seqblocks_ranks_distributed[n]
        block_end = cu_seqblocks_ranks_distributed[n + 1]
        token_start = global_cu_block_include_tokens[block_start].item()
        token_end = global_cu_block_include_tokens[block_end].item()
        gather_sizes.append(token_end - token_start)

    # compressed_gather_sizes is just the number of blocks per rank
    compressed_gather_sizes = block_num_for_each_rank

    return (
        output_list,
        input_list,
        output_block_list,
        input_block_list,
        global_cu_seqblocks,
        global_cu_block_include_tokens,
        local_cu_block_include_tokens,
        local_cu_seqlens,
        gather_sizes,
        compressed_gather_sizes,
    )


def all_to_all_tokens(
    tokens,
    output_split_sizes,
    input_split_sizes,
    local_only=True,
    num_gpus_per_node=None,
):
    # Get the appropriate process group
    if local_only:
        group = get_local_process_group(num_gpus_per_node)
    else:
        group = dist.group.WORLD  # Use default world group

    # Use custom autograd function that supports gradient descent
    return AllToAllSingle.apply(tokens, output_split_sizes, input_split_sizes, group)


def all_gather_tokens(
    tokens,
    gather_sizes=None,
    local_only=True,
    num_gpus_per_node=None,
):
    """
    All-gather tokens from all ranks with gradient support.

    Args:
        tokens: Local tensor to gather [local_tokens, hidden_dim]
        gather_sizes: Optional list of sizes from each rank. If None, will be computed
                      via all-gather of local sizes.
        local_only: If True, use intra-node process group
        num_gpus_per_node: Number of GPUs per node (for local process group)

    Returns:
        Gathered tensor [total_tokens, hidden_dim] containing data from all ranks
    """
    # Get the appropriate process group
    if local_only:
        group = get_local_process_group(num_gpus_per_node)
    else:
        group = dist.group.WORLD  # Use default world group

    world_size = dist.get_world_size(group)

    # If gather_sizes not provided, compute them via all-gather
    if gather_sizes is None:
        local_size = torch.tensor(
            [tokens.shape[0]], dtype=torch.long, device=tokens.device
        )
        size_list = [
            torch.zeros(1, dtype=torch.long, device=tokens.device)
            for _ in range(world_size)
        ]
        dist.all_gather(size_list, local_size, group=group)
        gather_sizes = [int(s.item()) for s in size_list]

    # Use custom autograd function that supports gradient descent
    return AllGatherSingle.apply(tokens, gather_sizes, group)


def _all_gather_inner(tokens, gather_sizes, group):
    """
    Inner function for all-gather that can be checkpointed.
    Separated to work with torch.utils.checkpoint.
    """
    return AllGatherSingle.apply(tokens, gather_sizes, group)


def all_gather_tokens_checkpointed(
    tokens,
    gather_sizes=None,
    local_only=True,
    num_gpus_per_node=None,
):
    """
    All-gather tokens with gradient checkpointing to reduce memory usage.

    Trades computation for memory: during backward pass, the all-gather operation
    is recomputed instead of storing the gathered tensor.

    Memory savings: ~N_total × hidden_dim × 2 bytes per call (bf16)
    Extra cost: One additional all-gather during backward pass

    Args:
        tokens: Local tensor to gather [local_tokens, hidden_dim]
        gather_sizes: Optional list of sizes from each rank. If None, will be computed
                      via all-gather of local sizes.
        local_only: If True, use intra-node process group
        num_gpus_per_node: Number of GPUs per node (for local process group)

    Returns:
        Gathered tensor [total_tokens, hidden_dim] containing data from all ranks
    """
    # Get the appropriate process group
    if local_only:
        group = get_local_process_group(num_gpus_per_node)
    else:
        group = dist.group.WORLD

    world_size = dist.get_world_size(group)

    # If gather_sizes not provided, compute them via all-gather
    if gather_sizes is None:
        local_size = torch.tensor(
            [tokens.shape[0]], dtype=torch.long, device=tokens.device
        )
        size_list = [
            torch.zeros(1, dtype=torch.long, device=tokens.device)
            for _ in range(world_size)
        ]
        dist.all_gather(size_list, local_size, group=group)
        gather_sizes = [int(s.item()) for s in size_list]

    # Use gradient checkpointing to save memory
    # During backward, all_gather will be recomputed instead of stored
    return torch.utils.checkpoint.checkpoint(
        _all_gather_inner,
        tokens,
        gather_sizes,
        group,
        use_reentrant=False,
    )


def seq_parallel_token_blocks(
    vol_token_arr, img_token_arr, attn_map, mode="distribute"
):
    assert mode == "distribute" or mode == "collect"

    # Initialize return values to None (will be set if tokens exist)
    local_vol_cu_seqlens = None
    local_img_cu_seqlens = None

    vol_token_arr_out = []
    if len(vol_token_arr) > 0:
        vol_token_slice_arr = []
        for data in vol_token_arr:
            vol_token_slice_arr.append(data.shape[1])

        vol_tokens = torch.cat(vol_token_arr, dim=1)
        if mode == "distribute":
            # Uniformly distribute vol token_blocks across GPUs on one node
            cu_seqblocks = attn_map["vol_to_vol_attn"]["cu_seqblocks"]
            cu_block_include_tokens = attn_map["vol_to_vol_attn"][
                "cu_block_include_tokens"
            ]
            (
                dist_map,
                collect_map,
                dist_blocks_map,
                collect_blocks_map,
                global_cu_seqblocks,
                global_cu_block_include_tokens,
                local_cu_block_include_tokens,
                local_vol_cu_seqlens,
                vol_gather_sizes,
                vol_compressed_gather_sizes,
            ) = compute_tokens_mapping(
                cu_seqblocks,
                cu_block_include_tokens,
            )
            attn_map["vol_dist_map"] = dist_map
            attn_map["vol_collect_map"] = collect_map
            attn_map["vol_dist_blocks_map"] = dist_blocks_map
            attn_map["vol_collect_blocks_map"] = collect_blocks_map

            # Store gather sizes for all_gather operations
            attn_map["vol_gather_sizes"] = vol_gather_sizes
            attn_map["vol_compressed_gather_sizes"] = vol_compressed_gather_sizes

            attn_map["vol_to_vol_attn"]["cu_seqblocks"] = global_cu_seqblocks
            attn_map["vol_to_vol_attn"]["cu_block_include_tokens"] = (
                global_cu_block_include_tokens
            )
            attn_map["vol_to_vol_attn"]["local_cu_block_include_tokens"] = (
                local_cu_block_include_tokens
            )

            attn_map["img_to_vol_attn"]["cu_seqblocks"] = global_cu_seqblocks
            attn_map["img_to_vol_attn"]["cu_block_include_tokens"] = (
                global_cu_block_include_tokens
            )
            attn_map["img_to_vol_attn"]["local_cu_block_include_tokens"] = (
                local_cu_block_include_tokens
            )

        else:
            dist_map = attn_map["vol_dist_map"]
            collect_map = attn_map["vol_collect_map"]

        if mode == "distribute":
            if attn_map["vol_to_vol_attn"]["block_topk"] is not None:
                assert attn_map["vol_to_img_attn"]["block_topk"] is not None
                vol_topk = torch.cat(
                    [
                        attn_map["vol_to_vol_attn"]["block_topk"][
                            0, :
                        ],  # Assume 2 heads has the same topk
                        attn_map["vol_to_img_attn"]["block_topk"][0, :],
                    ],
                    dim=1,
                )
                vol_to_vol_topknum = attn_map["vol_to_vol_attn"]["block_topk"].shape[2]
                vol_topk = all_to_all_tokens(vol_topk, dist_map, collect_map)
            vol_tokens = all_to_all_tokens(vol_tokens, dist_map, collect_map)
        elif mode == "collect":
            vol_tokens = all_to_all_tokens(vol_tokens, collect_map, dist_map)

        start_slice = 0
        for n in range(0, len(vol_token_slice_arr)):
            slice_size = vol_token_slice_arr[n]
            vol_token_arr_out.append(
                vol_tokens[:, start_slice : start_slice + slice_size]
            )
            start_slice += slice_size
        if (
            mode == "distribute"
            and attn_map["vol_to_vol_attn"]["block_topk"] is not None
        ):
            vol_to_vol_topk = vol_topk[:, :vol_to_vol_topknum]
            vol_to_img_topk = vol_topk[:, vol_to_vol_topknum:]
            attn_map["vol_to_vol_attn"]["block_topk"] = torch.stack(
                [
                    vol_to_vol_topk,
                    vol_to_vol_topk,
                ],
                dim=0,
            )
            attn_map["vol_to_img_attn"]["block_topk"] = torch.stack(
                [
                    vol_to_img_topk,
                    vol_to_img_topk,
                ],
                dim=0,
            )

    img_token_arr_out = []

    if len(img_token_arr) > 0:
        img_token_slice_arr = []
        for data in img_token_arr:
            img_token_slice_arr.append(data.shape[1])

        img_tokens = torch.cat(img_token_arr, dim=1)
        if mode == "distribute":
            # Uniformly distribute img token_blocks across GPUs on one node
            cu_seqblocks = attn_map["img_to_img_attn"]["cu_seqblocks"]
            cu_block_include_tokens = attn_map["img_to_img_attn"][
                "cu_block_include_tokens"
            ]
            (
                dist_map,
                collect_map,
                dist_blocks_map,
                collect_blocks_map,
                global_cu_seqblocks,
                global_cu_block_include_tokens,
                local_cu_block_include_tokens,
                local_img_cu_seqlens,
                img_gather_sizes,
                img_compressed_gather_sizes,
            ) = compute_tokens_mapping(
                cu_seqblocks,
                cu_block_include_tokens,
            )
            attn_map["img_dist_map"] = dist_map
            attn_map["img_collect_map"] = collect_map
            attn_map["img_dist_blocks_map"] = dist_blocks_map
            attn_map["img_collect_blocks_map"] = collect_blocks_map

            # Store gather sizes for all_gather operations
            attn_map["img_gather_sizes"] = img_gather_sizes
            attn_map["img_compressed_gather_sizes"] = img_compressed_gather_sizes

            attn_map["vol_to_img_attn"]["cu_seqblocks"] = global_cu_seqblocks
            attn_map["vol_to_img_attn"]["cu_block_include_tokens"] = (
                global_cu_block_include_tokens
            )
            attn_map["vol_to_img_attn"]["local_cu_block_include_tokens"] = (
                local_cu_block_include_tokens
            )

            attn_map["img_to_img_attn"]["cu_seqblocks"] = global_cu_seqblocks
            attn_map["img_to_img_attn"]["cu_block_include_tokens"] = (
                global_cu_block_include_tokens
            )
            attn_map["img_to_img_attn"]["local_cu_block_include_tokens"] = (
                local_cu_block_include_tokens
            )
        else:
            dist_map = attn_map["img_dist_map"]
            collect_map = attn_map["img_collect_map"]

        if mode == "distribute":
            if attn_map["img_to_img_attn"]["block_topk"] is not None:
                assert attn_map["img_to_vol_attn"]["block_topk"] is not None
                img_topk = torch.cat(
                    [
                        attn_map["img_to_img_attn"]["block_topk"][
                            0, :
                        ],  # Assume 2 heads has the same topk
                        attn_map["img_to_vol_attn"]["block_topk"][0, :],
                    ],
                    dim=1,
                )
                img_to_img_topknum = attn_map["img_to_img_attn"]["block_topk"].shape[2]
                img_topk = all_to_all_tokens(img_topk, dist_map, collect_map)
            img_tokens = all_to_all_tokens(img_tokens, dist_map, collect_map)
        elif mode == "collect":
            img_tokens = all_to_all_tokens(img_tokens, collect_map, dist_map)

        start_slice = 0
        for n in range(0, len(img_token_slice_arr)):
            slice_size = img_token_slice_arr[n]
            img_token_arr_out.append(
                img_tokens[:, start_slice : start_slice + slice_size]
            )
            start_slice += slice_size
        if (
            mode == "distribute"
            and attn_map["img_to_img_attn"]["block_topk"] is not None
        ):
            img_to_img_topk = img_topk[:, :img_to_img_topknum]
            img_to_vol_topk = img_topk[:, img_to_img_topknum:]
            attn_map["img_to_img_attn"]["block_topk"] = torch.stack(
                [
                    img_to_img_topk,
                    img_to_img_topk,
                ],
                dim=0,
            )
            attn_map["img_to_vol_attn"]["block_topk"] = torch.stack(
                [
                    img_to_vol_topk,
                    img_to_vol_topk,
                ],
                dim=0,
            )

    return (
        vol_token_arr_out,
        img_token_arr_out,
        local_vol_cu_seqlens,
        local_img_cu_seqlens,
        attn_map,
    )
