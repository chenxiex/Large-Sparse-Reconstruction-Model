# LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows

![Teaser](teaser.png)

Official code release for the paper
**LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows**. LSRM is a feed-forward, object-centric 3D reconstruction and inverse rendering model that reconstructs high-fidelity 3D assets from posed sparse multi-view images. The model is trained on synthetic data but comprehensively tested on real data.


* Project page: <https://lzqsd.github.io/LSRM.github.io/>
* arXiv: <https://arxiv.org/abs/2604.05182>
* Hugging Face: <https://huggingface.co/facebook/Large-Sparse-Reconstruction-Model>

All commands below assume your current working directory is the project root
(this folder).

## Setup

```sh
conda create -n lsrm python=3.10 -y
conda activate lsrm
bash install.sh
```

Tested on NVIDIA H200 (compute capability 9.0). `install.sh` pins to py3.10 +
torch 2.4.0 + cu121. Our inference code requires less than 40GB of GPU memory.

In addition, the following sibling clones / downloads are expected next to
this repo:

* **DINOv3** — model + weights (loaded via `torch.hub.load("dinov3", ...)`).
  The `dinov3_vith16plus` weights are gated; request access from the
  [DINOv3 repository](https://github.com/facebookresearch/dinov3) (see its
  README for the current download link / form), then:

  ```sh
  git clone https://github.com/facebookresearch/dinov3.git ../dinov3
  cp <path-to>/dinov3_vith16plus.pth ../dinov3/
  ln -s ../dinov3 dinov3
  ```

* **Blender** — used headlessly for mesh / texture rendering. Download the
  Linux build (we tested 4.5.3) into `../blender/blender-4.5.3-linux-x64/`.
  Override the path in test scripts via `--blender_bin` if needed.

## Checkpoints and testing data

Pre-trained weights and testing examples are hosted on Hugging Face at
[facebook/Large-Sparse-Reconstruction-Model](https://huggingface.co/facebook/Large-Sparse-Reconstruction-Model).

### Checkpoints

Download the checkpoint files and place them under `checkpoints/` with the
following layout (this is what the test scripts expect):

```
checkpoints/
├── rgb/                       # 3D reconstruction (GSO)
│   ├── dense/
│   │   ├── args.txt
│   │   └── checkpoints/last.pth
│   └── sparse.pth
└── brdf/                      # inverse rendering (ORB / DTC)
    ├── dense/
    │   ├── args.txt
    │   └── checkpoints/last.pth
    └── sparse.pth
```

### Testing data

Example datasets (GSO, ORB, DTC) and the default environment map `env.exr`
are also available from the same Hugging Face repo. Download them and place
under `datasets/` and the repo root respectively:

```
datasets/
├── rgb/
│   └── gso_example/           # GSO example (3D reconstruction)
└── brdf/
    ├── orb_example/           # ORB example (inverse rendering)
    └── dtc_example/           # DTC example (inverse rendering)

env.exr                       # default HDR environment map (repo root)
```

`env.exr` is the default HDR environment map used by Blender for re-rendering
predicted BRDFs / meshes. The test scripts pick it up automatically; you only
need to replace it if you want to re-light results under a different
illumination.

### Dataset format

To run on your own captures, mirror one of the example layouts. The two
formats are:

**RGB (GSO-style, for `test_rgb.sh`)**

Loaded by `data_loader/dtc_dataset.py`. Assumes the object is inside a
bounding sphere of radius **0.25**, which is scaled to **0.5** during data
loading (matching the `--bbox_radius 0.5` used in the test scripts).

```
datasets/rgb/<your_split>/
├── test.txt                     # one scene name per line
└── <scene_name>/
    ├── images/
    │   ├── CameraRig.json
    │   ├── image_process_info.json
    │   └── rgb/
    │       ├── rgb0000000.png … rgb000NNNN.png      # input views
    │       └── mask0000000.png … mask000NNNN.png    # foreground masks
    └── scene/
        ├── aria_trajectory.csv
        └── scene_info.json
```

**BRDF (ORB / DTC-style, for `test_brdf.sh`)**

Loaded by `data_loader/real_dataset.py`. Assumes the object is inside a
bounding sphere of radius **0.5** (matching the `--bbox_radius 0.5` used in
the test scripts).

```
datasets/brdf/<your_split>/
├── test.txt                     # one scene name per line
└── <scene_name>/
    ├── transforms_input.json    # camera poses for the input views
    ├── transforms_output.json   # camera poses for the eval views
    ├── scale_center.txt         # per-scene scene normalization
    ├── input/                   # input RGB
    ├── mask/                    # input foreground masks
    ├── output/                  # eval-view RGB (ground truth)
    ├── mask_output/             # eval-view foreground masks
    └── env/                     # environment maps (per scene)
```

Then point the corresponding test script at your split by editing `data_path`
inside the `.sh` file (or by overriding it on the command line).

## Running the tests

After downloading checkpoints and testing data from the Hugging Face repo
(see above), you can run the tests on the example datasets:

* [GSO](https://research.google/blog/scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/) — novel-view synthesis and 3D reconstruction.
* [ORB](https://stanfordorb.github.io/) — inverse rendering on real scenes.
* [DTC](https://www.projectaria.com/datasets/dtc/) — inverse rendering on real scenes.

```sh
conda activate lsrm

# Novel-view synthesis + 3D reconstruction (GSO).
bash test_rgb.sh

# Inverse rendering. Switch dataset via DATASET=orb (default) or DATASET=dtc.
bash test_brdf.sh
DATASET=dtc bash test_brdf.sh

# Re-render saved meshes into a video (no network inference).
bash test_brdf_video.sh
DATASET=dtc bash test_brdf_video.sh
```

Please see each script for the full set of flags.

### Outputs

Outputs land under `experiments/<exp_name>/<scene_name>/`. For RGB
reconstruction (`test_rgb.sh`) you get, per scene:

```
experiments/sparsergb_.../<scene_name>/
├── inputs_gt.png, inputs_mask.png      # input view montage
├── 0XX_rgb.png                         # rendered novel views (network output)
├── mesh.obj                            # extracted mesh (Marching Cubes)
└── mesh/
    ├── mesh_uv.obj, mesh_uv.mtl        # UV-unwrapped mesh + material
    ├── mesh_uv_rgb.png                 # baked RGB texture
    ├── mesh_uv_mask.png                # texture validity mask
    └── mesh_rendering/
        └── 0XX_rgb.png                 # Blender re-renders of the textured mesh
```

For inverse rendering (`test_brdf.sh`) you get, per scene:

```
experiments/sparsebrdf_.../<scene_name>/
├── inputs_gt.png, inputs_mask.png
├── 0XX_albedo.png                      # predicted albedo maps
├── 0XX_roughness.png                   # predicted roughness maps
├── 0XX_metallic.png                    # predicted metallic maps
├── mesh.obj                            # extracted mesh
├── mesh/
│   ├── mesh_uv.obj                     # UV-unwrapped mesh
│   ├── mesh_uv_albedo.png              # baked albedo texture
│   ├── mesh_uv_roughness.png           # baked roughness texture
│   ├── mesh_uv_metallic.png            # baked metallic texture
│   ├── mesh_uv_mask.png                # texture validity mask
│   └── mesh_rendering/
│       ├── 0XX_relight.png             # Blender re-renders (PNG)
│       └── 0XX_relight.exr             # Blender re-renders (EXR, HDR)
└── video/                              # populated by test_brdf_video.sh
    ├── 0XX_relight.png                 # video frames (PNG)
    └── 0XX_relight.exr                 # video frames (EXR, HDR)
```

`test_brdf_video.sh` re-renders the saved mesh into a rotating video (no
network inference) and populates the `video/` directory.

Top-level files in each `experiments/<exp_name>/`:

* `args.txt` — full CLI used for the run
* `weights.yaml` — resolved loss-weight config
* `volsdf.pth` — cached SDF/volume tensors
* `log.txt` — training/eval log
* `eval_<rank>.json` — per-rank metrics (when `--output_eval_json` is set)

## License

This project is licensed under the Creative Commons Attribution-NonCommercial
4.0 International (CC BY-NC 4.0) license. See [LICENSE.md](LICENSE.md) for
details.

## Acknowledgements

Our Triton implementation of native sparse attention builds on
[lucidrains/native-sparse-attention-pytorch](https://github.com/lucidrains/native-sparse-attention-pytorch)
and draws inspiration from
[DreamTechAI/Direct3D-S2](https://github.com/DreamTechAI/Direct3D-S2). We thank
the authors of both projects for releasing their work.

## Citation

If you find this work useful, please cite:

```bibtex
@article{lsrm,
    title   = {LSRM: High-Fidelity Object-Centric Reconstruction via Scaled Context Windows},
    author  = {Zhengqin Li and Cheng Zhang and Jakob Engel and Zhao Dong},
    journal = {arXiv preprint arXiv:2604.05182},
    year    = {2026}
}
```
