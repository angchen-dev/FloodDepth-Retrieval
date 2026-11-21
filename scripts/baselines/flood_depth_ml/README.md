# FLOOD-DEPTH-ML baseline

This directory contains the unmodified upstream FLOOD-DEPTH-ML Streamlit application and the repository-specific automated inference entry used to reproduce the paper comparison.

## Install the environment

The complete Conda and pip dependency set was exported from the environment used for the paper experiments. Create the public `fd_ml_env` environment from the repository root:

```bash
conda env create -f scripts/baselines/flood_depth_ml/environment.yml
conda activate fd_ml_env
```

The exported environment uses Python 3.10.15, PyTorch 2.2.2, torchvision 0.17.2, and `pytorch-cuda=12.1`. A compatible NVIDIA driver is required for the paper's GPU inference setting.

## Reproduce paper inference

The paper experiment uses GPU inference (`device=0`). After activating `fd_ml_env`, run from the repository root:

```bash
conda activate fd_ml_env

python scripts/baselines/flood_depth_ml/infer.py \
    --input-dir datasets/evaluation/real_world/images \
    --ground-truth scripts/analyze_results_paper/gt/real_world_gt.json \
    --output-dir runs/paper/baselines/flood_depth_ml/real_world \
    --device 0
```

For the synthetic evaluation, change the input and GT paths to `datasets/evaluation/synthetic/images` and `scripts/analyze_results_paper/gt/synthetic_gt.json`.

The entry defaults to `--device 0`, matching the original experiment. CPU inference remains available with `--device cpu`, but it is not the paper's default execution mode.

The default checkpoint is `data/baselines/flood_depth_ml/best_car.pt`. Annotated images, YOLO labels, and `predictions.json` are saved under the output directory. Following the paper experiment, all detected vehicles are retained, and the candidate with the smallest absolute error to GT supplies the image-level prediction.

## Preserved research files

- `flooddepthml.py`: unmodified original Streamlit application from [mayankmi/FLOOD-DEPTH-ML](https://github.com/mayankmi/FLOOD-DEPTH-ML).
- `requirements.txt`: dependency list distributed with the original implementation.
- `environment.yml`: exported environment used for the paper experiments.
- `LICENSE`: license distributed with the source implementation.

Use `infer.py`, which was written for this repository, as the automated public entry for reproducing the paper predictions on an image directory.
