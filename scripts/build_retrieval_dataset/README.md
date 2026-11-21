# Build the retrieval dataset

This directory contains the three-stage pipeline used to construct the paper's
rendered vehicle retrieval datasets:

1. `render.py` renders the flooded reference vehicle in Blender from multiple viewpoints.
2. `crop.py` extracts the vehicle/water region and writes cropped images/masks.
3. `extract_features.py` computes VGG19 pool4 features in HDF5 parts.

`build.py` runs the complete pipeline and creates the directory structure read
by the repository-level `infer.py`.

## Input data

The separately distributed data bundle must contain:

```text
data/build_retrieval_dataset/
├── car_models/
│   └── 2022_peugeot_308.glb    # reference vehicle
└── hdri/
    └── pine_picnic_4k.hdr
```

`build.py` renders and processes only `2022_peugeot_308.glb`, the reference vehicle used to construct the retrieval dataset. Other GLB files in `car_models/` are ignored. The rendered images, GLB file, HDR texture, and generated VGG caches are not stored in Git.

## Requirements

- Blender 4.x available as `blender` (the rendering stage uses Blender's Python).
- The repository environment (`fd_retrieval_env`) for cropping and VGG19 extraction.
- A CUDA GPU is recommended for feature extraction.

```bash
conda activate fd_retrieval_env
pip install -r scripts/build_retrieval_dataset/requirements.txt
```

VGG19 ImageNet weights are downloaded by torchvision on first use if they are
not already cached.

## Build the main paper retrieval dataset

Run from the repository root:

```bash
conda activate fd_retrieval_env
python scripts/build_retrieval_dataset/build.py \
    --depth-levels 20 \
    --layers 10 \
    --cameras-per-layer 72 \
    --device cuda:0
```

This writes temporary Blender renders under `runs/build_retrieval_dataset/`
and the inference inputs under:

```text
datasets/retrieval/
├── images/10layer_72cam_20depths_hdr_light/
│   ├── no_bg_cropped/
│   ├── green_bg_cropped/
│   └── masks/
└── features/10layer_72cam_20depths_hdr_light/
    ├── image_names.pkl
    └── features_*.h5
```

The paper ablation datasets are produced by changing `--depth-levels` to `10`
or `20`. By default, the number of HDF5 parts equals the number of depth
levels; override it with `--feature-parts`.

## Resume individual stages

Use `--stages` to run only selected stages:

```bash
python scripts/build_retrieval_dataset/build.py --depth-levels 5 --stages crop features
python scripts/build_retrieval_dataset/build.py --depth-levels 5 --stages features
```

Existing render/depth folders are skipped. Pass `--overwrite` to regenerate
the selected outputs; when the render stage is selected, this also removes
stale renders from other vehicle models in the same dataset directory. Use
`--dry-run` to inspect all commands without running them.
