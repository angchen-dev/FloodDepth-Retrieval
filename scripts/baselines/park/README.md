# Park et al. baseline

This directory contains our unofficial, independent reimplementation of the method described by Park et al. It is not code released by the original authors, and the original implementation and dataset were not publicly available.

Original paper:

- [Computer Vision–Based Estimation of Flood Depth in Flooded-Vehicle Images](https://ascelibrary.org/doi/10.1061/%28ASCE%29CP.1943-5487.0000956), *Journal of Computing in Civil Engineering*, 35(2), 04020072 (2021). DOI: `10.1061/(ASCE)CP.1943-5487.0000956`.

## Environment

Create a separate environment for this baseline:

```bash
conda create -n park python=3.11 -y
conda activate park

pip install torch==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install numpy==2.2.6 pillow==12.3.0 opencv-python==4.12.0.88 \
    scikit-learn==1.9.0 matplotlib==3.10.6
```

The commands above install PyTorch 2.8.0 with CUDA 12.8 and torchvision 0.23.0. A compatible NVIDIA driver is required for GPU inference. By default, `infer.py` selects `cuda:0` when CUDA is available and otherwise uses the CPU; the device can also be selected explicitly with `--device`.

The pretrained Mask R-CNN and VGG19 weights used by the implementation are downloaded by torchvision on first use if they are not already cached.

Blender is not required for inference. It is needed only when using `render_car_images.py` to regenerate the reconstructed reference views.

## Reference data

The default reference directory is `data/baselines/park/target_images/`. It contains the rendered reference images and `cm_per_pixel.json` scale metadata constructed by us from the method description in the paper. This is our reconstruction rather than the original authors' dataset; see [`data/baselines/park/README.md`](../../../data/baselines/park/README.md) for details.

## Reproduce paper inference

Run the real-world evaluation from the repository root:

```bash
conda activate park

python scripts/baselines/park/infer.py \
    --input-dir datasets/evaluation/real_world/images \
    --ground-truth scripts/analyze_results_paper/gt/real_world_gt.json \
    --output-dir runs/paper/baselines/park/real_world
```

Run the synthetic evaluation:

```bash
python scripts/baselines/park/infer.py \
    --input-dir datasets/evaluation/synthetic/images \
    --ground-truth scripts/analyze_results_paper/gt/synthetic_gt.json \
    --output-dir runs/paper/baselines/park/synthetic
```

Each command processes every supported image in the input directory. The output contains `predictions.json`, candidate-filtering visualizations under `draw_bbox/`, and query/reference visualizations under `matched_pairs/`.

## Implementation files

- `infer.py`: automated folder-level inference entry used to reproduce the paper predictions.
- `utils.py`: vehicle detection, candidate filtering, feature extraction, image matching, and flood-depth estimation helpers.
- `render_car_images.py`: Blender utility used to generate the reconstructed reference views.

The original research notebook is not required for inference and is not included.
