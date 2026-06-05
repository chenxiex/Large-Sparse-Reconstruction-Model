# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#! /bin/sh
# Pull in env vars (NVCC_PREPEND_FLAGS, CPATH, TORCH_CUDA_ARCH_LIST,
# TORCH_EXTENSIONS_DIR) from install.sh's activate hook in the current shell.
# Conda only fires activate.d/*.sh on `conda activate`, so a shell that was
# already active when install.sh ran won't see those vars otherwise. Sourcing
# the hook here lets users run `bash install.sh && bash test_brdf.sh` without
# a `conda deactivate && conda activate` in between.
if [ -n "${CONDA_PREFIX:-}" ] && [ -f "$CONDA_PREFIX/etc/conda/activate.d/zz_lsrm_env.sh" ]; then
    . "$CONDA_PREFIX/etc/conda/activate.d/zz_lsrm_env.sh"
fi
# Make conda env's lib/ available to subprocesses (e.g. blender) that need
# libimf.so / libiomp5.so / etc. shipped by intel-cmplr-lib-rt.
# Blender's own lib/ goes FIRST so its bundled libs (libsycl, libur_loader, ...)
# take precedence over any newer versions in the conda env.
blender_bin_for_path="../blender/blender-4.5.3-linux-x64/blender"
blender_lib_dir="$(dirname "$blender_bin_for_path")/lib"
if [ -d "$blender_lib_dir" ]; then
    export LD_LIBRARY_PATH="$blender_lib_dir:${CONDA_PREFIX:+$CONDA_PREFIX/lib:}${LD_LIBRARY_PATH:-}"
elif [ -n "${CONDA_PREFIX:-}" ]; then
    export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
fi

batch_size_per_gpu=1
num_workers=2
bbox_radius=0.5
output_image_res=1024
image_num_per_batch=18
volume_res=96
volume_dim=32
volume_upsample_scale=4
transformer_depth=24
prediction_type="brdf"
patch_size=8
embed_dim=1024
mlp_brdf_depth=3
mlp_geo_depth=2
sdf_inv_std=200
num_samples_per_ray=1024
input_image_res="256 768"
mvencoder_type="plucker"
blender_bin="../blender/blender-4.5.3-linux-x64/blender"
exp_root="experiments/"

# To switch dataset, set DATASET=dtc (or DATASET=orb) before invoking this
# script, e.g. `DATASET=dtc bash test_brdf.sh`. Default is orb.
DATASET="${DATASET:-orb}"
data_path="datasets/brdf/${DATASET}_example/"
exp_name="sparsebrdf_im768_vol384_3dattn_fuseall_${DATASET}"

torchrun --nproc_per_node=1 --master_port=29501 test_lrm_vol_sparse.py \
    --exp_root ${exp_root} \
    --exp_name ${exp_name} \
    --mvencoder_type ${mvencoder_type} \
    --batch_size_per_gpu ${batch_size_per_gpu} \
    --num_workers ${num_workers} \
    --data_path ${data_path} \
    --bbox_radius ${bbox_radius} \
    --volume_res ${volume_res} \
    --volume_upsample_scale ${volume_upsample_scale} \
    --output_image_res ${output_image_res} \
    --prediction_type ${prediction_type} \
    --volume_dim ${volume_dim} \
    --transformer_depth ${transformer_depth} \
    --image_num_per_batch ${image_num_per_batch} \
    --dataset_type "real_dataset" \
    --seed 9 \
    --patch_size ${patch_size} \
    --num_samples_per_ray ${num_samples_per_ray} \
    --embed_dim ${embed_dim} \
    --input_image_res ${input_image_res} \
    --use_weight_norm \
    --mlp_brdf_depth ${mlp_brdf_depth} \
    --mlp_geo_depth ${mlp_geo_depth} \
    --sdf_inv_std ${sdf_inv_std} \
    --loss_weights_file "sdf_init" \
    --use_dino "v3" \
    --dense_pretrained_folder "checkpoints/brdf/dense" \
    --centralized_cropping \
    --checkpoint "checkpoints/brdf/sparse.pth" \
    --use_decomposed_embed \
    --use_3d_aware_attn \
    --fuse_all_layers \
    --output_eval_json \
    --ocgrid_acc \
    --relative_cam_pose \
    --encode_background "augment" \
    --save_mesh \
    --save_texture \
    --render_mesh \
    --render_mesh_mode "relighting" \
    --camera_center_coord \
    --render_white_bg \
    --output_texture_res 2048 \
    --volume_dilation_radius 3 \
    --topk_multiimage 4 \
    --topk_volume 16 \
    --attn_winsize 3 \
    --ocgrid_dilation_radius 2 \
    --blender_bin ${blender_bin} \
