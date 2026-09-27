# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
# Install the LSRM Python environment.
#
# Usage:
#     conda create -n lsrm python=3.10 -y
#     conda activate lsrm
#     bash install.sh
#
# Pinned to py3.10 + torch 2.4.0 + cu121 because both pytorch3d and flash-attn
# ship prebuilt wheels for this combo. nerfacc 0.5.3 has no prebuilt wheel for
# torch 2.4 and JIT-compiles on first use; we install nvcc + CUDA dev headers
# below so that build can succeed without a system CUDA toolkit.

set -e

if [ -z "${CONDA_PREFIX:-}" ]; then
    echo "ERROR: activate a conda env first, e.g.:"
    echo "    conda create -n lsrm python=3.10 -y && conda activate lsrm"
    exit 1
fi

echo "=== installing torch 2.4.0 + cu121 ==="
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121

echo "=== installing pytorch3d (prebuilt for py310_cu121_pyt240) ==="
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d \
    -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html

echo "=== installing nerfacc 0.5.3 + nvcc/CUDA headers (for first-use JIT compile) ==="
pip install nerfacc==0.5.3
pip install \
    nvidia-cuda-nvcc-cu12 \
    nvidia-cuda-cccl-cu12 \
    nvidia-cuda-runtime-cu12 \
    nvidia-cublas-cu12 \
    nvidia-cusparse-cu12 \
    nvidia-cusolver-cu12 \
    nvidia-curand-cu12 \
    nvidia-cufft-cu12

# nerfacc's JIT compile needs the actual `nvcc` binary; the pip
# `nvidia-cuda-nvcc-cu12` package only ships ptxas + headers. Install nvcc
# AND a matching gcc/g++ (<= 12, since nvcc 12.1 rejects gcc > 12) from
# conda-forge.
echo "=== installing nvcc + cuda-cccl + gcc 12 toolchain from conda-forge ==="
conda install -y -c conda-forge \
    cuda-nvcc=12.1 \
    cuda-cccl=12.1 \
    gcc_linux-64=12 \
    gxx_linux-64=12

echo "=== isolating env from base/global state (nvcc + torch JIT cache) ==="
# Robustness fixes baked into the env, applied on every `conda activate`:
#
# 1. NVCC_PREPEND_FLAGS: conda-forge's cuda-nvcc activate hook *appends* to
#    NVCC_PREPEND_FLAGS instead of resetting it. If the user's base conda env
#    also has cuda-nvcc + a different gcc installed, activating that env first
#    leaves a `-ccbin=<base-gcc>` entry in NVCC_PREPEND_FLAGS. nvcc honors the
#    *first* -ccbin, so the (possibly too-new) base gcc silently overrides our
#    env's gcc 12, causing nerfacc's JIT compile to fail with
#    "unsupported GNU version! gcc versions later than 12 are not supported".
#    We reset NVCC_PREPEND_FLAGS to a single, correct -ccbin.
#
# 2. TORCH_EXTENSIONS_DIR: torch's default JIT cache dir
#    ~/.cache/torch_extensions/py<X>_cu<Y>/ is keyed only by python + cuda
#    versions, so two envs with the same versions share (and corrupt) the
#    cache. We give each env its own cache dir under $CONDA_PREFIX.
#
# 3. TORCH_CUDA_ARCH_LIST: newer cuda-nvcc packages (12.6+) export a default
#    list that includes Blackwell archs (10.0, 10.1, 12.0). torch 2.4.0
#    doesn't know about those and aborts JIT builds with
#    "Unknown CUDA arch (10.0) or GPU not supported". We pin to the local
#    GPU's actual compute capability (with a broad fallback for envs where
#    torch isn't yet importable).
#
# 4. CPATH: nerfacc's JIT compile needs CUDA headers (cuda_runtime_api.h,
#    cusparse.h, ...). On boxes without a system CUDA toolkit, those headers
#    come from the pip nvidia-*-cu12 packages we installed above; we expose
#    their include dirs to the compiler via CPATH.
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d"
cat > "$CONDA_PREFIX/etc/conda/activate.d/zz_lsrm_env.sh" <<'EOF'
export NVCC_PREPEND_FLAGS="-ccbin=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export NVCC_CCBIN="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-c++"
export TORCH_EXTENSIONS_DIR="$CONDA_PREFIX/var/torch_extensions"
mkdir -p "$TORCH_EXTENSIONS_DIR"

# Pin TORCH_CUDA_ARCH_LIST to the local GPU's actual compute capability.
# Falls back to a broad list if torch/CUDA isn't ready (e.g. activating the
# env before install.sh finishes).
__cap="$(python -c 'import torch; m,n=torch.cuda.get_device_capability(0); print(f"{m}.{n}")' 2>/dev/null)"
export TORCH_CUDA_ARCH_LIST="${__cap:-8.0;8.6;8.9;9.0+PTX}"
unset __cap

# Make the pip nvidia-*-cu12 CUDA headers visible to nvcc / c++ on machines
# without a system CUDA toolkit (so JIT extensions can rebuild if cache misses).
__nvidia_pkg_dir="$(python -c 'import nvidia,os; print(os.path.dirname(nvidia.__file__))' 2>/dev/null)"
if [ -n "$__nvidia_pkg_dir" ]; then
    for __d in cublas cusparse cusolver curand cufft cuda_runtime cuda_cccl cuda_nvcc nvjitlink nvtx; do
        [ -d "$__nvidia_pkg_dir/$__d/include" ] && \
            CPATH="$__nvidia_pkg_dir/$__d/include:${CPATH:-}"
    done
    export CPATH
fi
unset __nvidia_pkg_dir __d
EOF
# Source the new hook for the rest of this install run.
source "$CONDA_PREFIX/etc/conda/activate.d/zz_lsrm_env.sh"

echo "=== installing flash-attn (prebuilt for torch 2.4 + cu122 + py310) ==="
pip install \
    https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.9.post1/flash_attn-2.5.9.post1+cu122torch2.4cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo "=== installing other Python dependencies ==="
pip install \
    'numpy<2' \
    trimesh \
    plyfile \
    einops \
    lpips \
    scikit-image \
    opencv-python \
    scipy \
    OpenEXR \
    ffmpeg-python \
    ninja \
    pyyaml \
    rich \
    timm \
    torchmetrics \
    pillow \
    tqdm \
    matplotlib \
    intel-cmplr-lib-rt   # provides libimf.so for blender on slimmer compute nodes

echo "=== pre-building nerfacc CUDA extension (so first run doesn't JIT) ==="
# Triggers nerfacc's first-use JIT build now, with the correct env vars set,
# so any toolchain breakage surfaces during install (not deep into a training
# run). The compiled .so lands in $TORCH_EXTENSIONS_DIR.
python - <<'PY'
import torch
from nerfacc import ray_aabb_intersect
rays_o = torch.zeros(1, 3, device="cuda")
rays_d = torch.tensor([[0.0, 0.0, 1.0]], device="cuda")
aabbs  = torch.tensor([[-1.0, -1.0, -1.0, 1.0, 1.0, 1.0]], device="cuda")
ray_aabb_intersect(rays_o, rays_d, aabbs)
print("nerfacc CUDA extension built OK")
PY

echo "=== verifying ==="
python - <<'PY'
import torch, nerfacc, flash_attn
from pytorch3d import _C
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("nerfacc:", nerfacc.__version__)
print("flash_attn:", flash_attn.__version__)
print("pytorch3d _C OK")
PY

echo "=== done ==="
# test_rgb.sh and test_brdf.sh source the activate hook themselves, so they
# work in this same shell without `conda deactivate && conda activate`. If
# you want the env vars (CPATH, TORCH_CUDA_ARCH_LIST, TORCH_EXTENSIONS_DIR,
# NVCC_PREPEND_FLAGS) for ad-hoc use elsewhere, re-activate the env.
