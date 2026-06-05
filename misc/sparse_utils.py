# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
import random

import torch
import torch.nn.functional as F

from models_sparse.seqparallel_utils import all_gather_fixed_size


def compute_block_topk_dist_chunked(
    points: torch.Tensor,
    block_topk_oneim: torch.Tensor,
    impoints: torch.Tensor,
    impoints_mask: torch.Tensor,
    chunk_size: int = 8194,
    max_dist: float = 1e10,
) -> torch.Tensor:
    points_num = points.shape[0]
    block_num = block_topk_oneim.shape[1]
    device = points.device
    dtype = points.dtype

    block_topk_dist = torch.empty(points_num, block_num, device=device, dtype=dtype)

    for start in range(0, points_num, chunk_size):
        end = min(start + chunk_size, points_num)
        chunk_len = end - start
        points_chunk = points[start:end]
        block_topk_oneim_chunk = block_topk_oneim[start:end]

        impoints_chunk = impoints[block_topk_oneim_chunk.reshape(-1)].reshape(
            chunk_len, block_num, -1, 3
        )
        impoints_mask_chunk = impoints_mask[block_topk_oneim_chunk.reshape(-1)].reshape(
            chunk_len, block_num, -1
        )

        chunk_dist = (
            torch.sum(
                (points_chunk.reshape(-1, 1, 1, 3) - impoints_chunk) ** 2,
                dim=-1,
            )
            ** 0.5
        )
        chunk_dist = (
            chunk_dist * impoints_mask_chunk + (1 - impoints_mask_chunk) * max_dist
        )
        block_topk_dist[start:end] = torch.min(chunk_dist, dim=2)[0]

    return block_topk_dist


def reservoir_sampling(weight, index, max_num):
    weight_nonz = weight[weight > 0]
    weight_index = torch.nonzero(weight).int()
    weight_index = weight_index.squeeze(1)

    r = torch.rand_like(weight_nonz)
    sampling_weight = -(r ** (1 / weight_nonz))

    # Always set the first voxel of each batch to be sampled
    min_weight = sampling_weight.min()
    start_index = torch.nonzero(index[1:, 0] - index[:-1, 0]) + 1
    sampling_weight[0] = min_weight - 1
    if start_index.numel() > 0:
        start_index = start_index.squeeze(1)  # Fix: squeeze to 1D for proper indexing
        sampling_weight[start_index] = min_weight - 1

    # Ensure max_num is at least batch_size to guarantee one token per batch
    batch_size = index[:, 0].max().item() + 1
    max_num = max(max_num, batch_size)

    selected_index = torch.argsort(sampling_weight)[:max_num]
    weight_index = weight_index[selected_index]
    weight = weight * 0
    weight[weight_index] = 1
    weight = weight.to(dtype=torch.bool)
    index = index[selected_index]

    # Verify mask and index stay in sync after sampling
    assert weight.sum() == index.shape[0], (
        f"Mismatch in reservoir_sampling output: "
        f"weight.sum()={weight.sum().item()}, index.shape[0]={index.shape[0]}"
    )

    return weight, index


def sort_image_blocks(feats, coords, args, coords_idx=None):
    if coords_idx is None:
        im_num = coords[:, 1].max().item() + 1

        coords_block = coords[:, 2:] // args.block_size
        coords_inblock = coords[:, 2:] % args.block_size

        patch_res = args.input_image_res[1] // args.patch_size
        offset1 = im_num * patch_res * patch_res
        offset2 = patch_res * patch_res
        offset3 = patch_res * args.block_size
        offset4 = args.block_size * args.block_size
        offset5 = args.block_size

        coords_idx = (
            offset1 * coords[:, 0]
            + offset2 * coords[:, 1]
            + offset3 * coords_block[:, 0]
            + offset4 * coords_block[:, 1]
            + offset5 * coords_inblock[:, 0]
            + coords_inblock[:, 1]
        )
        coords_idx = torch.argsort(coords_idx)
        coords = coords[coords_idx, :]
        feats = feats[coords_idx, :]
    else:
        coords = coords[coords_idx, :]
        feats = feats[coords_idx, :]
    return feats, coords, coords_idx


def sort_volume_blocks(feats, coords, args, coords_idx=None):
    if coords_idx is None:
        coords_block = coords[:, 1:] // args.block_size
        coords_inblock = coords[:, 1:] % args.block_size

        offset1 = args.volume_res * args.volume_res * args.volume_res
        offset2 = args.volume_res * args.volume_res * args.block_size
        offset3 = args.volume_res * args.block_size * args.block_size
        offset4 = args.block_size * args.block_size * args.block_size
        offset5 = args.block_size * args.block_size
        offset6 = args.block_size
        coords_idx = (
            offset1 * coords[:, 0]
            + offset2 * coords_block[:, 0]
            + offset3 * coords_block[:, 1]
            + offset4 * coords_block[:, 2]
            + offset5 * coords_inblock[:, 0]
            + offset6 * coords_inblock[:, 1]
            + coords_inblock[:, 2]
        )
        coords_idx = torch.argsort(coords_idx)
        coords = coords[coords_idx, :]
        feats = feats[coords_idx, :]
    else:
        feats = feats[coords_idx, :]
        coords = coords[coords_idx, :]
    return feats, coords, coords_idx


def compute_token_num_on_local_rank(
    token_num_list,
    token_num_local,
    max_token_num,
):
    """
    Compute the token limit for the local rank when total tokens exceed the budget.

    The goal is to distribute tokens as fairly as possible across ranks while respecting
    the max_token_num limit. Ranks that are under the limit keep their tokens, while
    ranks that exceed the limit share the remaining budget proportionally.

    Args:
        token_num_list: List of token counts from all ranks
        token_num_local: Token count on the local rank
        max_token_num: Maximum allowed tokens per rank

    Returns:
        The token limit for this rank
    """
    group_size = len(token_num_list)
    total_budget = max_token_num * group_size
    total_tokens = sum(token_num_list)

    # If total tokens are within budget, no reduction needed
    if total_tokens <= total_budget:
        return token_num_local

    # If local rank is already under the limit, no reduction needed for this rank
    if token_num_local <= max_token_num:
        return token_num_local

    # Calculate tokens from ranks that exceed the limit
    tokens_from_exceeding_ranks = 0
    tokens_from_under_limit_ranks = 0
    for token_num in token_num_list:
        if token_num <= max_token_num:
            tokens_from_under_limit_ranks += token_num
        else:
            tokens_from_exceeding_ranks += token_num

    # Budget available for exceeding ranks = total budget - tokens kept by under-limit ranks
    budget_for_exceeding_ranks = total_budget - tokens_from_under_limit_ranks

    # Distribute this budget proportionally among exceeding ranks
    # Each exceeding rank gets a share proportional to its token count
    if tokens_from_exceeding_ranks > 0:
        proportion = float(token_num_local) / float(tokens_from_exceeding_ranks)
        token_num_limit = max(int(proportion * budget_for_exceeding_ranks), 1)
    else:
        token_num_limit = token_num_local

    # Ensure we don't exceed the original token count
    return min(token_num_limit, token_num_local)


def reduce_images_to_fit_limit(mask, target_token_num):
    """
    Reduce the number of images (from the end) until token count is below target.
    Returns the reduced mask and the actual number of images kept.

    Args:
        mask: Tensor of shape [batch_size, im_num, ph, pw] with token counts per patch
        target_token_num: Maximum number of tokens allowed

    Returns:
        reduced_mask: Mask with only the first N images kept
        actual_im_num: Number of images kept
    """
    im_num = mask.shape[1]

    # Count tokens per image (summed across batch and spatial dims)
    tokens_per_image = (mask > 0).sum(dim=(0, 2, 3))  # [im_num]
    cumsum_tokens = torch.cumsum(tokens_per_image, dim=0)  # [im_num]

    # Find max images we can keep while staying under limit
    actual_im_num = im_num
    for num_images in range(im_num, 0, -1):
        if num_images == 1:
            actual_im_num = 1
            break
        tokens_with_n_images = cumsum_tokens[num_images - 1].item()
        if tokens_with_n_images <= target_token_num:
            actual_im_num = num_images
            break

    if actual_im_num < im_num:
        print(
            "Reduce the number of images from %d to %d to fit token limit."
            % (im_num, actual_im_num)
        )
        mask = mask[:, :actual_im_num, :, :]

    return mask, actual_im_num


def sparsify_imfeat(imfeat, mask, args):
    # Identify patches with content
    batch_size, im_num, _, height, width = mask.shape
    ph, pw = height // args.patch_size, width // args.patch_size
    mask = mask.reshape(batch_size, im_num, ph, args.patch_size, pw, args.patch_size)
    mask = mask.permute(0, 1, 2, 4, 3, 5)
    mask = mask.reshape(batch_size, im_num, ph, pw, args.patch_size * args.patch_size)
    mask = torch.sum(mask, dim=-1)

    image_index = torch.nonzero(mask).int()
    mask_flat = mask.reshape(-1)
    actual_im_num = im_num

    if args.max_image_token_num is not None:
        if args.use_sequence_parallel:
            image_token_num = torch.tensor(
                [image_index.shape[0]], dtype=torch.long, device=mask_flat.device
            )
            image_token_num_list = all_gather_fixed_size(image_token_num)
            image_token_num_list = [x.item() for x in image_token_num_list]
            image_token_num_local = image_token_num.item()
            image_token_num_sum = sum(image_token_num_list)
            group_size = len(image_token_num_list)
            max_image_token_num_sum = args.max_image_token_num * group_size
            if image_token_num_sum > max_image_token_num_sum:
                image_token_num_limit = compute_token_num_on_local_rank(
                    image_token_num_list,
                    image_token_num_local,
                    args.max_image_token_num,
                )
                if image_token_num_limit < image_index.shape[0]:
                    # Instead of reservoir sampling, reduce number of images
                    mask, actual_im_num = reduce_images_to_fit_limit(
                        mask, image_token_num_limit
                    )
                    image_index = torch.nonzero(mask).int()
                    mask_flat = mask.reshape(-1)
        else:
            if image_index.shape[0] > args.max_image_token_num:
                # Instead of reservoir sampling, reduce number of images
                mask, actual_im_num = reduce_images_to_fit_limit(
                    mask, args.max_image_token_num
                )
                image_index = torch.nonzero(mask).int()
                mask_flat = mask.reshape(-1)

    mask = mask_flat > 0

    old_size = int(math.sqrt(imfeat[0].shape[1] / im_num))
    imfeat_sparse = []
    dtype = imfeat[0].dtype
    image_coords_idx = None
    for n in range(0, len(imfeat)):
        feat = imfeat[n].reshape(
            batch_size * im_num, old_size, old_size, args.embed_dim
        )
        feat = F.interpolate(feat.permute(0, 3, 1, 2), (ph, pw))
        feat = feat.reshape(batch_size * im_num, args.embed_dim, ph * pw).permute(
            0, 2, 1
        )
        feat = feat.reshape(batch_size, im_num, ph * pw, args.embed_dim)
        # Only take the first actual_im_num images
        feat = feat[:, :actual_im_num, :, :]
        feat = feat.reshape(-1, args.embed_dim)
        feat = feat[mask, :]
        feat = F.normalize(feat, dim=1).to(dtype=dtype)
        feat, _, image_coords_idx = sort_image_blocks(
            feat, image_index, args, image_coords_idx
        )
        imfeat_sparse.append(feat)
    image_index = image_index[image_coords_idx, :]
    return imfeat_sparse, mask, image_index, image_coords_idx, actual_im_num


def filter_volume_from_depth(volume_res, batch, radius, device):
    batch_size = batch["surface_points_input"].shape[0]
    surface_points_input = batch["surface_points_input"].to(device=device)
    depth_masks_input = batch["depth_masks_input"].to(device=device)

    occupancy_arr = []
    for b in range(0, batch_size):
        mask = torch.zeros(
            volume_res, volume_res, volume_res, dtype=torch.float32, device=device
        )
        surface_points = surface_points_input[b, :].reshape(-1, 3)
        surface_points_mask = depth_masks_input[b, :].reshape(-1)
        surface_points = surface_points[surface_points_mask == 1, :]
        x, y, z = surface_points[:, 0], surface_points[:, 1], surface_points[:, 2]
        x_index = torch.clamp((x + radius) / 2 / radius, 0, 1) * (volume_res - 1)
        y_index = torch.clamp((y + radius) / 2 / radius, 0, 1) * (volume_res - 1)
        z_index = torch.clamp((z + radius) / 2 / radius, 0, 1) * (volume_res - 1)
        x_index = x_index.to(dtype=torch.int32)
        y_index = y_index.to(dtype=torch.int32)
        z_index = z_index.to(dtype=torch.int32)
        mask[z_index, y_index, x_index] = 1
        occupancy_arr.append(mask)

    occupancy_arr = torch.stack(occupancy_arr, dim=0)
    return occupancy_arr


def augment_mask_volume(
    mask_volume,
    num_regions=2,
    region_size=6,
    augment_prob=1.0,
):
    if num_regions <= 0:
        return mask_volume

    print(
        "Augmenting mask volume with %d regions of size %d."
        % (num_regions, region_size)
    )
    device = mask_volume.device
    batch_size = mask_volume.shape[0]
    vol_size = mask_volume.shape[1]

    # Apply augmentation per batch with given probability
    for b in range(batch_size):
        if random.random() > augment_prob:
            continue

        # Find False positions (inactive voxels) that could be turned on
        false_mask = ~mask_volume[b]
        false_indices = torch.nonzero(false_mask, as_tuple=False)

        if false_indices.shape[0] == 0:
            continue

        # Randomly select seed points from False positions
        num_seeds = min(num_regions, false_indices.shape[0])
        perm = torch.randperm(false_indices.shape[0], device=device)[:num_seeds]
        seed_positions = false_indices[perm]

        # For each seed, turn on a cubic region around it
        half_size = region_size // 2
        half_size_z = random.randint(1, half_size)
        half_size_y = random.randint(1, half_size)
        half_size_x = random.randint(1, half_size)
        for seed_idx in range(seed_positions.shape[0]):
            z, y, x = seed_positions[seed_idx]

            # Compute bounds for the cubic region
            z_min = max(0, z.item() - half_size_z)
            z_max = min(vol_size, z.item() + half_size_z + 1)
            y_min = max(0, y.item() - half_size_y)
            y_max = min(vol_size, y.item() + half_size_y + 1)
            x_min = max(0, x.item() - half_size_x)
            x_max = min(vol_size, x.item() + half_size_x + 1)

            # Turn on the region
            mask_volume[b, z_min:z_max, y_min:y_max, x_min:x_max] = True

    return mask_volume


def sparsify_volfeat(
    volfeat,
    sdf_volume,
    batch,
    dense_points,
    alpha_volume,
    args,
    alpha_threshold=2e-3,
    augment_volume=True,
):
    batch_size = sdf_volume.shape[0]
    sdf_volume_size = int(round(sdf_volume.shape[1] ** (1.0 / 3.0)))
    sdf_volume = sdf_volume.reshape(
        batch_size, sdf_volume_size, sdf_volume_size, sdf_volume_size
    )
    sdf_patch_size = sdf_volume_size // args.volume_res
    sdf_block_size = sdf_patch_size * sdf_patch_size * sdf_patch_size
    sdf_volume = sdf_volume.reshape(
        batch_size,
        args.volume_res,
        sdf_patch_size,
        args.volume_res,
        sdf_patch_size,
        args.volume_res,
        sdf_patch_size,
    )
    alpha_volume = alpha_volume.reshape(
        batch_size,
        args.volume_res,
        sdf_patch_size,
        args.volume_res,
        sdf_patch_size,
        args.volume_res,
        sdf_patch_size,
    )
    sdf_volume = sdf_volume.permute(0, 1, 3, 5, 2, 4, 6)
    alpha_volume = alpha_volume.permute(0, 1, 3, 5, 2, 4, 6)
    sdf_volume = sdf_volume.reshape(
        batch_size, args.volume_res, args.volume_res, args.volume_res, sdf_block_size
    )
    alpha_volume = alpha_volume.reshape(
        batch_size, args.volume_res, args.volume_res, args.volume_res, sdf_block_size
    )
    sdf_volume_max, _ = torch.max(sdf_volume, dim=4)
    sdf_volume_min, _ = torch.min(sdf_volume, dim=4)
    alpha_volume_max, _ = torch.max(alpha_volume, dim=4)
    alpha_volume_min, _ = torch.min(alpha_volume, dim=4)
    mask_volume = torch.logical_or(
        torch.logical_and(sdf_volume_max > 0, sdf_volume_min < 0),
        torch.logical_and(sdf_volume_min > 0, alpha_volume_max > alpha_threshold),
    )
    mask_volume = torch.logical_or(
        mask_volume,
        torch.logical_and(sdf_volume_max < 0, alpha_volume_min < 1 - alpha_threshold),
    )

    # Dilate mask_volume
    if args.volume_dilation_radius > 0:
        kernel_size = int(2 * args.volume_dilation_radius + 1)
        padding = args.volume_dilation_radius
        mask_volume = (
            torch.nn.functional.max_pool3d(
                mask_volume.float().unsqueeze(1),
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
            )
            .squeeze(1)
            .bool()
        )

    # Every object must have at least one valid voxel
    for n in range(0, batch_size):
        if torch.sum(mask_volume[n, :].to(dtype=torch.long)) == 0:
            _, max_coords = torch.max(alpha_volume_max[n, :].reshape(-1), dim=0)
            z = max_coords // (args.volume_res * args.volume_res)
            max_coords = max_coords - z * (args.volume_res * args.volume_res)
            y = max_coords // args.volume_res
            max_coords = max_coords - y * args.volume_res
            x = max_coords
            mask_volume[n, z, y, x] = True

    # Augment mask_volume before reshaping (keep 4D shape for proper region handling)
    if augment_volume:
        mask_volume = augment_mask_volume(
            mask_volume,
            num_regions=getattr(args, "augment_num_regions", 2),
            region_size=getattr(args, "augment_region_size", 6),
            augment_prob=getattr(args, "augment_prob", 1.0),
        )
    volume_index = torch.nonzero(mask_volume).int()
    mask_volume = mask_volume.reshape(-1)

    if args.max_volume_token_num is not None:
        if args.use_sequence_parallel:
            volume_token_num = torch.tensor(
                [volume_index.shape[0]], dtype=torch.long, device=mask_volume.device
            )
            volume_token_num_list = all_gather_fixed_size(volume_token_num)
            volume_token_num_list = [x.item() for x in volume_token_num_list]
            volume_token_num_local = volume_token_num.item()
            volume_token_num_sum = sum(volume_token_num_list)
            group_size = len(volume_token_num_list)
            max_volume_token_num_sum = args.max_volume_token_num * group_size
            if volume_token_num_sum > max_volume_token_num_sum:
                volume_token_num_limit = compute_token_num_on_local_rank(
                    volume_token_num_list,
                    volume_token_num_local,
                    args.max_volume_token_num,
                )
                if volume_token_num_limit < volume_index.shape[0]:
                    occupancy_volume = filter_volume_from_depth(
                        volume_res=args.volume_res,
                        batch=batch if dense_points is None else dense_points,
                        radius=1.05 * args.bbox_radius,
                        device=alpha_volume.device,
                    )

                    print(
                        "Reduce the number of volume tokens from %d to %d."
                        % (volume_index.shape[0], int(volume_token_num_limit))
                    )
                    alpha_volume_sampling = torch.sum(alpha_volume, dim=-1)
                    alpha_volume_sampling += occupancy_volume * (sdf_patch_size**3) * 5
                    alpha_volume_sampling = torch.clamp(alpha_volume_sampling, min=1e-2)
                    alpha_volume_sampling = (
                        alpha_volume_sampling.reshape(-1) * mask_volume
                    )
                    mask_volume, volume_index = reservoir_sampling(
                        alpha_volume_sampling, volume_index, int(volume_token_num_limit)
                    )

                    sort_volume_index = (
                        volume_index[:, 0] * (args.volume_res**3)
                        + volume_index[:, 1] * (args.volume_res**2)
                        + volume_index[:, 2] * args.volume_res
                        + volume_index[:, 3]
                    )
                    sort_volume_index = torch.argsort(sort_volume_index)
                    volume_index = volume_index[sort_volume_index, :]
        else:
            if volume_index.shape[0] > args.max_volume_token_num:
                occupancy_volume = filter_volume_from_depth(
                    volume_res=args.volume_res,
                    batch=batch if dense_points is None else dense_points,
                    radius=1.05 * args.bbox_radius,
                    device=alpha_volume.device,
                )

                print(
                    "Reduce the number of volume tokens from %d to %d."
                    % (volume_index.shape[0], args.max_volume_token_num)
                )
                alpha_volume_sampling = torch.sum(alpha_volume, dim=-1)
                alpha_volume_sampling += occupancy_volume * (sdf_patch_size**3) * 5
                alpha_volume_sampling = torch.clamp(alpha_volume_sampling, min=1e-2)
                alpha_volume_sampling = alpha_volume_sampling.reshape(-1) * mask_volume
                mask_volume, volume_index = reservoir_sampling(
                    alpha_volume_sampling, volume_index, args.max_volume_token_num
                )

                sort_volume_index = (
                    volume_index[:, 0] * (args.volume_res**3)
                    + volume_index[:, 1] * (args.volume_res**2)
                    + volume_index[:, 2] * args.volume_res
                    + volume_index[:, 3]
                )
                sort_volume_index = torch.argsort(sort_volume_index)
                volume_index = volume_index[sort_volume_index, :]

    old_size = int(round(volfeat[-1].shape[1] ** (1.0 / 3.0)))
    dtype = volfeat[-1].dtype
    feat = volfeat[-1].reshape(batch_size, old_size, old_size, old_size, args.embed_dim)
    feat = feat.permute(0, 4, 1, 2, 3)
    feat = F.interpolate(feat, (args.volume_res, args.volume_res, args.volume_res))
    feat = feat.permute(0, 2, 3, 4, 1).reshape(-1, args.embed_dim)
    feat = feat[mask_volume, :]
    feat = F.normalize(feat, dim=-1).to(dtype=dtype)
    volfeat_sparse, volume_index, volume_coords_idx = sort_volume_blocks(
        feat, volume_index, args
    )
    return volfeat_sparse, mask_volume, volume_index, volume_coords_idx


def compute_block_separation(k_coords, DIM, block_size, batch_size, resolution):
    seqk_id = k_coords[1:, 0] - k_coords[:-1, 0]
    seqk = torch.nonzero(seqk_id) + 1
    cu_seqlens_k = []
    cu_seqlens_k.append(0)
    for n in range(batch_size - 1):
        cu_seqlens_k.append(seqk[n].item())
    cu_seqlens_k.append(k_coords.shape[0])

    block_res = [resolution[n] // block_size[n] for n in range(DIM)]
    cu_block_res = [1]
    for n in range(DIM):
        cu_block_res.insert(0, cu_block_res[0] * block_res[DIM - 1 - n])

    seqblocks, block_include_tokens = [], []
    for b in range(batch_size):
        k_start, k_end = cu_seqlens_k[b], cu_seqlens_k[b + 1]

        block_coords_b = k_coords[k_start:k_end]
        for n in range(DIM):
            block_coords_b[:, n + 1] = block_coords_b[:, n + 1] // block_size[n]
        block_coords_flatten_b = 0
        for n in range(DIM):
            block_coords_flatten_b = (
                block_coords_flatten_b + block_coords_b[:, n + 1] * cu_block_res[n + 1]
            )
        block_bins_b = torch.histc(
            block_coords_flatten_b, bins=cu_block_res[0], min=0, max=cu_block_res[0] - 1
        )
        block_include_tokens.append(block_bins_b[block_bins_b > 0])
        seqblocks.append(len(block_include_tokens[-1]))

    seqblocks = torch.Tensor(seqblocks).to(k_coords.device)
    cu_seqblocks = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=k_coords.device),
            torch.cumsum(seqblocks, dim=0),
        ],
        dim=0,
    ).to(torch.int32)
    block_include_tokens = torch.cat(block_include_tokens)
    cu_block_include_tokens = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=k_coords.device),
            torch.cumsum(block_include_tokens, dim=0),
        ],
        dim=0,
    ).to(torch.int32)

    return cu_seqblocks, cu_block_include_tokens


def from_volcoords_to_points(volcoords, args):
    radius = args.bbox_radius * 1.05
    z = 2 * (volcoords[:, 1].float() / (args.volume_res - 1) - 0.5) * radius
    y = 2 * (volcoords[:, 2].float() / (args.volume_res - 1) - 0.5) * radius
    x = 2 * (volcoords[:, 3].float() / (args.volume_res - 1) - 0.5) * radius
    points = torch.stack([x, y, z], dim=-1)  # align
    return points


def from_imcoords_to_points(imcoords, dense_points, args):
    # Median filter for each patch
    patch_res = args.input_image_res[1] // args.patch_size
    patch_size = dense_points["surface_points_input"][0, 0].shape[0] // patch_res
    device = imcoords.device

    batch_size, im_num = dense_points["surface_points_input"].shape[0:2]
    impoints = dense_points["surface_points_input"]
    impoints_mask = dense_points["masks_input"]

    impoints = impoints.reshape(
        batch_size, im_num, patch_res, patch_size, patch_res, patch_size, 3
    )
    impoints_mask = impoints_mask.reshape(
        batch_size, im_num, patch_res, patch_size, patch_res, patch_size, 1
    )
    impoints = impoints.permute(0, 1, 2, 4, 3, 5, 6)
    impoints = impoints.reshape(-1, patch_size * patch_size, 3)
    impoints_mask = impoints_mask.permute(0, 1, 2, 4, 3, 5, 6)
    impoints_mask = impoints_mask.reshape(-1, patch_size * patch_size)

    im_ind = (
        imcoords[:, 0] * im_num * patch_res * patch_res
        + imcoords[:, 1] * patch_res * patch_res
        + imcoords[:, 2] * patch_res
        + imcoords[:, 3]
    )
    impoints = impoints[im_ind, :, :]
    impoints_mask = impoints_mask[im_ind, :]

    # Assign the max value to 1
    impoints_mask_maxindex = impoints_mask.max(dim=-1).indices
    impoints_mask_idx = torch.arange(0, impoints_mask.shape[0], device=device)
    points = impoints[impoints_mask_idx, impoints_mask_maxindex, :]
    return points


def build_image_to_image_attn(imfeat_coords, batch, dense_points, args):
    block_topk = None
    cu_seqblocks = None
    cu_block_include_tokens = None
    max_dist = args.bbox_radius * 2
    win_size = args.attn_winsize**2

    device = imfeat_coords.device

    # Compute cu_seqblocks and cu_block_include_tokens
    DIM = 3
    block_size = [1, args.block_size, args.block_size]
    batch_size = imfeat_coords[:, 0].max().item() + 1

    k_coords = imfeat_coords.clone().detach()
    resolution = [
        dense_points["surface_points_input"].shape[1],
        args.input_image_res[-1] // args.block_size,
        args.input_image_res[-1] // args.block_size,
    ]
    cu_seqblocks, cu_block_include_tokens = compute_block_separation(
        k_coords, DIM, block_size, batch_size, resolution
    )

    if args.use_3d_aware_attn:
        imcoords_gap = imfeat_coords[1:, 0] - imfeat_coords[:-1, 0]
        seq_id = torch.nonzero(imcoords_gap) + 1
        q_cu_seq = [0]
        for n in range(0, batch_size - 1):
            q_cu_seq.append(seq_id[n].item())
        q_cu_seq.append(imfeat_coords.shape[0])
        image_num = dense_points["surface_points_input"].shape[1]

        block_topk = []
        topk_num = args.topk_multiimage * args.image_num_per_batch
        for n in range(0, batch_size):
            block_offset = 0
            imcoords = imfeat_coords[q_cu_seq[n] : q_cu_seq[n + 1], :]
            points = from_imcoords_to_points(imcoords, dense_points, args)

            block_topk_oneinst = []
            block_topk_dist_oneinst = []
            # separate imcoords for each image
            for m in range(0, image_num):
                # Compute mask_id
                mask = batch["mask_input"][n, m, 0].to(device=device)
                block_pixel_size = args.patch_size * args.block_size
                height, width = mask.shape
                block_height, block_width = (
                    height // block_pixel_size,
                    width // block_pixel_size,
                )
                mask = mask.reshape(
                    block_height, block_pixel_size, block_width, block_pixel_size
                )
                mask = mask.permute(0, 2, 1, 3).reshape(block_height * block_width, -1)

                mask_sum = torch.sum(mask, dim=-1)
                mask_selected_id = torch.nonzero(mask_sum)[:, 0]
                if mask_selected_id.shape[0] == 0:
                    continue

                # Compute points and points_mask for each block
                impoints = dense_points["surface_points_input"][n, m, :]
                impoints_mask = (dense_points["masks_input"][n, m, :] > 0.2).float()
                impoints_size = impoints.shape[0]
                impoints_block_size = impoints_size // block_height  # Assume square

                impoints = impoints.reshape(
                    block_height,
                    impoints_block_size,
                    block_width,
                    impoints_block_size,
                    3,
                )
                impoints = impoints.permute(0, 2, 1, 3, 4).reshape(
                    block_height * block_width, -1, 3
                )
                impoints_mask = impoints_mask.reshape(
                    block_height,
                    impoints_block_size,
                    block_width,
                    impoints_block_size,
                    1,
                )
                impoints_mask = impoints_mask.permute(0, 2, 1, 3, 4).reshape(
                    block_height * block_width, -1, 1
                )
                impoints = impoints[mask_selected_id, :]
                impoints_mask = impoints_mask[mask_selected_id, :]

                block_center_x = (
                    torch.linspace(0, block_width - 1, block_width) + 0.5
                ).to(device=device)
                block_center_y = (
                    torch.linspace(0, block_height - 1, block_height) + 0.5
                ).to(device=device)
                block_center_y, block_center_x = torch.meshgrid(
                    block_center_y, block_center_x, indexing="ij"
                )
                block_center_x = block_center_x.reshape(-1)[mask_selected_id]
                block_center_y = block_center_y.reshape(-1)[mask_selected_id]

                # Project points to image and compute distance to block
                camera = batch["cameras_input"][n, m, :].to(device=device)
                extrinsic = camera[:16].reshape(4, 4)
                fov = camera[16]

                world_to_cam = torch.inverse(extrinsic)

                points_proj = torch.cat(
                    [
                        points,
                        torch.ones(points.shape[0], dtype=points.dtype, device=device)[
                            :, None
                        ],
                    ],
                    dim=1,
                )
                points_proj = torch.matmul(points_proj, world_to_cam.permute(1, 0))
                points_proj = points_proj[:, :3]
                points_proj_x = points_proj[:, 0] / torch.clamp(
                    -points_proj[:, 2], min=1e-6
                )
                points_proj_y = -points_proj[:, 1] / torch.clamp(
                    -points_proj[:, 2], min=1e-6
                )
                points_proj_x = 0.5 * (points_proj_x / torch.tan(fov / 2.0) + 1)
                points_proj_y = 0.5 * (points_proj_y / torch.tan(fov / 2.0) + 1)
                if "crop_input" in batch.keys():
                    crop_info = batch["crop_input"][n, m].to(device=device)
                    origin_size = batch["origin_size"][n, 0].to(device=device)
                    crop_size = crop_info[1] - crop_info[0]

                    pixel_x = (
                        (points_proj_x * (origin_size - 1) - crop_info[2])
                        / (crop_size - 1)
                        * (impoints_size - 1)
                    )
                    pixel_y = (
                        (points_proj_y * (origin_size - 1) - crop_info[0])
                        / (crop_size - 1)
                        * (impoints_size - 1)
                    )
                    block_x = pixel_x / impoints_block_size
                    block_y = pixel_y / impoints_block_size
                else:
                    pixel_x = points_proj_x * (impoints_size - 1)
                    pixel_y = points_proj_y * (impoints_size - 1)
                    block_x = pixel_x / impoints_block_size
                    block_y = pixel_y / impoints_block_size

                block_dist = (block_x[:, None] - block_center_x[None, :]) ** 2 + (
                    block_y[:, None] - block_center_y
                ) ** 2
                true_topk = min(win_size, block_dist.shape[1])

                block_topk_oneim = (-block_dist).topk(k=true_topk, dim=1).indices
                block_topk_dist = compute_block_topk_dist_chunked(
                    points,
                    block_topk_oneim,
                    impoints,
                    impoints_mask.squeeze(-1),
                    chunk_size=8194,
                    max_dist=max_dist,
                )
                block_topk_oneinst.append(block_topk_oneim + block_offset)
                block_topk_dist_oneinst.append(block_topk_dist)

                block_offset += mask_selected_id.shape[0]

            block_topk_oneinst = torch.cat(block_topk_oneinst, dim=-1)
            block_topk_dist_oneinst = torch.cat(block_topk_dist_oneinst, dim=-1)
            true_topk = min(topk_num, block_topk_dist_oneinst.shape[-1])
            block_topk_oneinst_index = (
                (-block_topk_dist_oneinst)
                .topk(k=true_topk, dim=-1)
                .indices.sort(-1)
                .values
            )
            block_topk_oneinst = torch.gather(
                block_topk_oneinst, dim=1, index=block_topk_oneinst_index.long()
            )
            if true_topk < topk_num:
                block_topk_oneinst = torch.cat(
                    [
                        block_topk_oneinst,
                        -torch.ones(
                            block_topk_oneinst.shape[0],
                            topk_num - true_topk,
                            device=device,
                            dtype=torch.int32,
                        ),
                    ],
                    dim=-1,
                )
            block_topk.append(block_topk_oneinst)

        block_topk = torch.cat(block_topk, dim=0)
        block_topk = torch.stack([block_topk, block_topk], dim=0).to(dtype=torch.int32)
    else:
        block_topk = None

    res = {}
    res["block_topk"] = block_topk
    res["cu_seqblocks"] = cu_seqblocks.to(dtype=torch.int32)
    res["cu_block_include_tokens"] = cu_block_include_tokens.to(dtype=torch.int32)

    return res


def build_image_to_volume_attn(imfeat_coords, volfeat_coords, dense_points, args):
    block_topk = None
    cu_seqblocks = None
    cu_block_include_tokens = None

    # Compute cu_seqblocks and cu_block_include_tokens
    DIM = 3
    block_size = [args.block_size, args.block_size, args.block_size]
    batch_size = volfeat_coords[:, 0].max().item() + 1
    k_coords = volfeat_coords.clone().detach()
    resolution = [
        args.volume_res,
        args.volume_res,
        args.volume_res,
    ]
    cu_seqblocks, cu_block_include_tokens = compute_block_separation(
        k_coords, DIM, block_size, batch_size, resolution
    )

    if args.use_3d_aware_attn:
        imcoords_gap = imfeat_coords[1:, 0] - imfeat_coords[:-1, 0]
        seq_id = torch.nonzero(imcoords_gap) + 1
        q_cu_seq = [0]
        for n in range(0, batch_size - 1):
            q_cu_seq.append(seq_id[n].item())
        q_cu_seq.append(imfeat_coords.shape[0])

        # Separate volfeat_coords
        volcoords_gap = volfeat_coords[1:, 0] - volfeat_coords[:-1, 0]
        seq_id = torch.nonzero(volcoords_gap) + 1
        k_cu_seq = [0]
        for n in range(0, batch_size - 1):
            k_cu_seq.append(seq_id[n].item())
        k_cu_seq.append(volfeat_coords.shape[0])

        block_topk = []
        topk_num = args.topk_volume
        block_include_tokens = (
            cu_block_include_tokens[1:] - cu_block_include_tokens[:-1]
        )
        for n in range(0, batch_size):
            imcoords = imfeat_coords[q_cu_seq[n] : q_cu_seq[n + 1], :]
            volcoords = volfeat_coords[k_cu_seq[n] : k_cu_seq[n + 1], :]

            impoints = from_imcoords_to_points(imcoords, dense_points, args)
            volpoints = from_volcoords_to_points(volcoords, args)
            volblock_points = torch.segment_reduce(
                data=volpoints,
                reduce="mean",
                lengths=block_include_tokens[cu_seqblocks[n] : cu_seqblocks[n + 1]],
                axis=0,
            )

            block_dist = (
                torch.sum(
                    (impoints[:, None, :] - volblock_points[None, :, :]) ** 2, dim=-1
                )
                ** 0.5
            )
            true_topk = min(topk_num, block_dist.shape[-1])
            block_topk_oneinst = (
                (-block_dist).topk(k=true_topk, dim=-1).indices.sort(-1).values
            )
            if true_topk < topk_num:
                block_topk_oneinst = torch.cat(
                    [
                        block_topk_oneinst,
                        -torch.ones(
                            block_topk_oneinst.shape[0],
                            topk_num - true_topk,
                            device=impoints.device,
                            dtype=torch.int32,
                        ),
                    ],
                    dim=-1,
                )
            block_topk.append(block_topk_oneinst)

        block_topk = torch.cat(block_topk, dim=0)
        block_topk = torch.stack([block_topk, block_topk], dim=0).to(dtype=torch.int32)
    else:
        block_topk = None

    res = {}
    res["block_topk"] = block_topk
    res["cu_seqblocks"] = cu_seqblocks.to(dtype=torch.int32)
    res["cu_block_include_tokens"] = cu_block_include_tokens.to(dtype=torch.int32)
    return res


def build_volume_to_image_attn(
    imfeat_coords,
    volfeat_coords,
    batch,
    dense_points,
    args,
):
    block_topk = None
    cu_seqblocks = None
    cu_block_include_tokens = None
    max_dist = args.bbox_radius * 2
    win_size = args.attn_winsize**2

    device = imfeat_coords.device
    # Compute cu_seqblocks and cu_block_include_tokens
    DIM = 3
    block_size = [1, args.block_size, args.block_size]
    batch_size = imfeat_coords[:, 0].max().item() + 1
    k_coords = imfeat_coords.clone().detach()
    resolution = [
        dense_points["surface_points_input"].shape[1],
        args.input_image_res[-1] // args.block_size,
        args.input_image_res[-1] // args.block_size,
    ]
    cu_seqblocks, cu_block_include_tokens = compute_block_separation(
        k_coords, DIM, block_size, batch_size, resolution
    )

    if args.use_3d_aware_attn:
        volcoords_gap = volfeat_coords[1:, 0] - volfeat_coords[:-1, 0]
        seq_id = torch.nonzero(volcoords_gap) + 1
        q_cu_seq = [0]
        for n in range(0, batch_size - 1):
            q_cu_seq.append(seq_id[n].item())
        q_cu_seq.append(volfeat_coords.shape[0])
        image_num = dense_points["surface_points_input"].shape[1]

        block_topk = []
        topk_num = args.topk_multiimage * args.image_num_per_batch
        for n in range(0, batch_size):
            block_offset = 0
            volcoords = volfeat_coords[q_cu_seq[n] : q_cu_seq[n + 1], :]
            points = from_volcoords_to_points(volcoords, args)

            block_topk_oneinst = []
            block_topk_dist_oneinst = []
            # separate imcoords for each image
            for m in range(0, image_num):
                # Compute mask_id
                mask = batch["mask_input"][n, m, 0].to(device=device)
                block_pixel_size = args.patch_size * args.block_size
                height, width = mask.shape
                block_height, block_width = (
                    height // block_pixel_size,
                    width // block_pixel_size,
                )
                mask = mask.reshape(
                    block_height, block_pixel_size, block_width, block_pixel_size
                )
                mask = mask.permute(0, 2, 1, 3).reshape(block_height * block_width, -1)

                mask_sum = torch.sum(mask, dim=-1)
                mask_selected_id = torch.nonzero(mask_sum)[:, 0]
                if mask_selected_id.shape[0] == 0:
                    continue

                # Compute points and points_mask for each block
                impoints = dense_points["surface_points_input"][n, m, :]
                impoints_mask = (dense_points["masks_input"][n, m, :] > 0.2).float()
                impoints_size = impoints.shape[0]
                impoints_block_size = impoints_size // block_height  # Assume square

                impoints = impoints.reshape(
                    block_height,
                    impoints_block_size,
                    block_width,
                    impoints_block_size,
                    3,
                )
                impoints = impoints.permute(0, 2, 1, 3, 4).reshape(
                    block_height * block_width, -1, 3
                )
                impoints_mask = impoints_mask.reshape(
                    block_height,
                    impoints_block_size,
                    block_width,
                    impoints_block_size,
                    1,
                )
                impoints_mask = impoints_mask.permute(0, 2, 1, 3, 4).reshape(
                    block_height * block_width, -1, 1
                )
                impoints = impoints[mask_selected_id, :]
                impoints_mask = impoints_mask[mask_selected_id, :]

                block_center_x = (
                    torch.linspace(0, block_width - 1, block_width) + 0.5
                ).to(device=device)
                block_center_y = (
                    torch.linspace(0, block_height - 1, block_height) + 0.5
                ).to(device=device)
                block_center_y, block_center_x = torch.meshgrid(
                    block_center_y, block_center_x, indexing="ij"
                )
                block_center_x = block_center_x.reshape(-1)[mask_selected_id]
                block_center_y = block_center_y.reshape(-1)[mask_selected_id]

                # Project points to image and compute distance to block
                camera = batch["cameras_input"][n, m, :].to(device=device)
                extrinsic = camera[:16].reshape(4, 4)
                fov = camera[16]

                world_to_cam = torch.inverse(extrinsic)
                points_proj = torch.cat(
                    [
                        points,
                        torch.ones(points.shape[0], dtype=points.dtype, device=device)[
                            :, None
                        ],
                    ],
                    dim=1,
                )
                points_proj = torch.matmul(points_proj, world_to_cam.permute(1, 0))
                points_proj = points_proj[:, :3]
                points_proj_x = points_proj[:, 0] / torch.clamp(
                    -points_proj[:, 2], min=1e-6
                )
                points_proj_y = -points_proj[:, 1] / torch.clamp(
                    -points_proj[:, 2], min=1e-6
                )
                points_proj_x = 0.5 * (points_proj_x / torch.tan(fov / 2.0) + 1)
                points_proj_y = 0.5 * (points_proj_y / torch.tan(fov / 2.0) + 1)
                if "crop_input" in batch.keys():
                    crop_info = batch["crop_input"][n, m].to(device=device)
                    origin_size = batch["origin_size"][n, 0].to(device=device)
                    crop_size = crop_info[1] - crop_info[0]

                    pixel_x = (
                        (points_proj_x * (origin_size - 1) - crop_info[2])
                        / (crop_size - 1)
                        * (impoints_size - 1)
                    )
                    pixel_y = (
                        (points_proj_y * (origin_size - 1) - crop_info[0])
                        / (crop_size - 1)
                        * (impoints_size - 1)
                    )
                    block_x = pixel_x / impoints_block_size
                    block_y = pixel_y / impoints_block_size
                else:
                    pixel_x = points_proj_x * (impoints_size - 1)
                    pixel_y = points_proj_y * (impoints_size - 1)
                    block_x = pixel_x / impoints_block_size
                    block_y = pixel_y / impoints_block_size

                block_dist = (block_x[:, None] - block_center_x[None, :]) ** 2 + (
                    block_y[:, None] - block_center_y
                ) ** 2
                true_topk = min(block_dist.shape[1], win_size)

                block_topk_oneim = (-block_dist).topk(k=true_topk, dim=1).indices
                block_topk_dist = compute_block_topk_dist_chunked(
                    points,
                    block_topk_oneim,
                    impoints,
                    impoints_mask.squeeze(-1),
                    chunk_size=8194,
                    max_dist=max_dist,
                )
                block_topk_oneinst.append(block_topk_oneim + block_offset)
                block_topk_dist_oneinst.append(block_topk_dist)

                block_offset += mask_selected_id.shape[0]

            block_topk_oneinst = torch.cat(block_topk_oneinst, dim=-1)
            block_topk_dist_oneinst = torch.cat(block_topk_dist_oneinst, dim=-1)
            true_topk = min(block_topk_dist_oneinst.shape[-1], topk_num)
            block_topk_oneinst_index = (
                (-block_topk_dist_oneinst)
                .topk(k=true_topk, dim=-1)
                .indices.sort(-1)
                .values
            )
            block_topk_oneinst = torch.gather(
                block_topk_oneinst, dim=1, index=block_topk_oneinst_index.long()
            )
            if true_topk < topk_num:
                block_topk_oneinst = torch.cat(
                    [
                        block_topk_oneinst,
                        -torch.ones(
                            (block_topk_oneinst.shape[0], topk_num - true_topk),
                            dtype=block_topk_oneinst.dtype,
                            device=device,
                        ),
                    ],
                    dim=1,
                )
            block_topk.append(block_topk_oneinst)

        block_topk = torch.cat(block_topk, dim=0)
        block_topk = torch.stack([block_topk, block_topk], dim=0).to(dtype=torch.int32)
    else:
        block_topk = None

    res = {}
    res["block_topk"] = block_topk
    res["cu_seqblocks"] = cu_seqblocks.to(dtype=torch.int32)
    res["cu_block_include_tokens"] = cu_block_include_tokens.to(dtype=torch.int32)
    return res


def build_volume_to_volume_attn(volfeat_coords, args):
    block_topk = None
    cu_seqblocks = None
    cu_block_include_tokens = None

    # Compute cu_seqblocks and cu_block_include_tokens
    DIM = 3
    block_size = [args.block_size, args.block_size, args.block_size]
    batch_size = volfeat_coords[:, 0].max().item() + 1
    k_coords = volfeat_coords.clone().detach()
    resolution = [
        args.volume_res,
        args.volume_res,
        args.volume_res,
    ]
    cu_seqblocks, cu_block_include_tokens = compute_block_separation(
        k_coords, DIM, block_size, batch_size, resolution
    )

    if args.use_3d_aware_attn:
        volcoords_gap = volfeat_coords[1:, 0] - volfeat_coords[:-1, 0]
        seq_id = torch.nonzero(volcoords_gap) + 1
        k_cu_seq = [0]
        for n in range(0, batch_size - 1):
            k_cu_seq.append(seq_id[n].item())
        k_cu_seq.append(volfeat_coords.shape[0])

        block_topk = []
        topk_num = args.topk_volume
        block_include_tokens = (
            cu_block_include_tokens[1:] - cu_block_include_tokens[:-1]
        )
        for n in range(0, batch_size):
            volcoords = volfeat_coords[k_cu_seq[n] : k_cu_seq[n + 1], :]

            volpoints = from_volcoords_to_points(volcoords, args)
            volblock_points = torch.segment_reduce(
                data=volpoints,
                reduce="mean",
                lengths=block_include_tokens[cu_seqblocks[n] : cu_seqblocks[n + 1]],
                axis=0,
            )

            block_dist = (
                torch.sum(
                    (volpoints[:, None, :] - volblock_points[None, :, :]) ** 2, dim=-1
                )
                ** 0.5
            )
            true_topk = min(block_dist.shape[-1], topk_num)
            block_topk_oneinst = (
                (-block_dist).topk(k=true_topk, dim=-1).indices.sort(-1).values
            )
            if true_topk < topk_num:
                block_topk_oneinst = torch.cat(
                    [
                        block_topk_oneinst,
                        -torch.ones(
                            (block_topk_oneinst.shape[0], topk_num - true_topk),
                            dtype=block_topk_oneinst.dtype,
                            device=volpoints.device,
                        ),
                    ],
                    dim=1,
                )

            block_topk.append(block_topk_oneinst)

        block_topk = torch.cat(block_topk, dim=0)
        block_topk = torch.stack([block_topk, block_topk], dim=0).to(dtype=torch.int32)
    else:
        block_topk = None

    res = {}
    res["block_topk"] = block_topk
    res["cu_seqblocks"] = cu_seqblocks.to(dtype=torch.int32)
    res["cu_block_include_tokens"] = cu_block_include_tokens.to(dtype=torch.int32)
    return res


def build_3D_aware_attn(imfeat_coords, volfeat_coords, batch, dense_points, args):
    i_2_i = build_image_to_image_attn(imfeat_coords, batch, dense_points, args)
    i_2_v = build_image_to_volume_attn(
        imfeat_coords, volfeat_coords, dense_points, args
    )
    v_2_i = build_volume_to_image_attn(
        imfeat_coords, volfeat_coords, batch, dense_points, args
    )
    v_2_v = build_volume_to_volume_attn(volfeat_coords, args)
    res = {}
    res["img_to_img_attn"] = i_2_i
    res["img_to_vol_attn"] = i_2_v
    res["vol_to_img_attn"] = v_2_i
    res["vol_to_vol_attn"] = v_2_v

    return res
