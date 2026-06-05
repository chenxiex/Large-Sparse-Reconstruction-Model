# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#! /bin/sh
# Script for rendering mesh and textures saved on disk.
# Uses the data loader to get camera poses and environment maps, but skips
# network inference (no nerfacc needed) and renders directly from saved mesh
# files produced by test_brdf.sh.

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
prediction_type="brdf"
input_image_res="256 768"
exp_root="experiments/"

# To switch dataset, set DATASET=dtc (or DATASET=orb) before invoking this
# script, e.g. `DATASET=dtc bash test_brdf_video.sh`. Default is orb.
DATASET="${DATASET:-orb}"
data_path="datasets/brdf/${DATASET}_example/"
exp_name="sparsebrdf_im768_vol384_3dattn_fuseall_${DATASET}"

# `--rotate_y_to_z` is only meaningful for the DTC dataset; orb meshes are
# already in the expected coordinate frame.
extra_args=""
if [ "$DATASET" = "dtc" ]; then
    extra_args="$extra_args --rotate_y_to_z"
fi

torchrun --nproc_per_node=1 --master_port=29501 test_lrm_mesh_rendering.py \
    --exp_root ${exp_root} \
    --exp_name ${exp_name} \
    --batch_size_per_gpu ${batch_size_per_gpu} \
    --num_workers ${num_workers} \
    --data_path ${data_path} \
    --bbox_radius ${bbox_radius} \
    --output_image_res ${output_image_res} \
    --prediction_type ${prediction_type} \
    --image_num_per_batch ${image_num_per_batch} \
    --dataset_type "real_dataset" \
    --seed 9 \
    --input_image_res ${input_image_res} \
    --centralized_cropping \
    --save_mesh_video \
    --env_video_path "env.exr" \
    --render_mesh_mode "relighting" \
    --blender_bin ${blender_bin_for_path} \
    ${extra_args}
