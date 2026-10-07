# Flood Depth Estimation Using Images of Vehicles

Official code and datasets for *Flood depth estimation using images of vehicles: a new image retrieval framework with annotated datasets*.

The proposed method first detects and segments vehicles with SAM3, then matches each segmented vehicle against a rendered retrieval dataset using VGG19 features. The final flood-depth estimate is selected from the detected vehicle candidates using retrieval similarity and vehicle area.

This repository supports three workflows:

1. [Estimate flood depth with the proposed method](#1-estimate-flood-depth-with-the-proposed-method)
2. [Reproduce the paper results](#2-reproduce-the-paper-results)
3. [Build the retrieval and real-world datasets](#3-build-the-datasets)

## Repository structure

```text
FloodDepth-Retrieval/
├── infer.py                         # Proposed-method inference entry
├── flooddepth_retrieval/            # Segmentation and retrieval pipeline
│   └── sam3/                        # Bundled SAM3 source
├── scripts/
│   ├── analyze_results_paper/       # Ground truth, metrics, and comparison
│   ├── ablation/                    # Complete paper ablation
│   ├── baselines/                   # Paper baseline reproduction
│   ├── build_retrieval_dataset/     # Render, crop, and feature pipeline
│   └── build_real_world_dataset/    # Participant annotation web app
├── datasets/                        # Evaluation and retrieval datasets
├── data/                            # Weights and build/runtime assets
└── runs/
    ├── paper/                       # Released paper results
    │   ├── ours/                    # Proposed-method predictions and metrics
    │   ├── baselines/               # Baseline predictions and GPT responses
    │   ├── ablation/                # Complete ablation study
    │   └── analysis/                # Paper comparison tables and figures
    ├── inference/                   # Generated independent inference outputs
    └── build_retrieval_dataset/     # Generated temporary retrieval renders
```

All commands below are run from the repository root.

## Data and released results

Download the [data and released-results bundle](https://telin.ugent.be/nextcloud/index.php/s/pPQcCwd2nFjqGLF), then extract it into the repository root. The bundle provides `datasets/`, `data/`, and `runs/paper/`, each with a separate responsibility:

```text
datasets/
├── evaluation/
│   ├── real_world/images/
│   └── synthetic/images/
└── retrieval/
│   ├── images/
│   └── features/

data/
├── baselines/                    # Baseline weights and reference assets
├── build_retrieval_dataset/      # GLB models and HDRI
└── build_real_world_dataset/     # Annotation images and web-app assets

runs/
├── paper/                        # Released results used in the paper
│   ├── ours/                     # Predictions and metrics for the proposed method
│   ├── baselines/                # Saved baseline predictions and GPT responses
│   ├── ablation/                 # Complete ablation-study outputs and summaries
│   └── analysis/                 # Comparison tables and figures
├── inference/                    # Created by independent inference
└── build_retrieval_dataset/      # Created for temporary retrieval-dataset renders
```

The `runs/paper/` directory contains the saved predictions, metrics, complete ablation outputs, tables, and figures used for the paper. `runs/inference/` and `runs/build_retrieval_dataset/` are generated only when users run independent inference or the retrieval-dataset builder. Dataset files, model weights, and large generated outputs are intentionally excluded from Git.

## 1. Estimate flood depth with the proposed method

### 1.1 Install the environment

Install Conda and create the environment supplied with this repository:

```bash
conda env create -f environment.yml
conda activate fd_retrieval_env
python -m pip install --no-deps pip==26.1.1
python -m pip install --no-deps -r requirements-sam3-lock.txt
python -m pip install --no-deps -e ./flooddepth_retrieval/sam3
```

The exported environment pins Python 3.12.13, PyTorch 2.10.0+cu128, torchvision 0.25.0+cu128, and NumPy 1.26.4. An NVIDIA GPU with a compatible CUDA driver is recommended.

### 1.2 Run inference on an image folder

`infer.py` processes every supported image in the input directory. It runs SAM3 segmentation and VGG19 retrieval as one pipeline; the two stages do not need to be launched separately.

The following example uses the paper's main retrieval dataset (`T=20`):

```bash
python infer.py \
    --input-dir datasets/evaluation/real_world/images \
    --retrieval-images-dir datasets/retrieval/images/10layer_72cam_20depths_hdr_light/no_bg_cropped \
    --retrieval-features-dir datasets/retrieval/features/10layer_72cam_20depths_hdr_light \
    --output-dir runs/inference/demo \
    --device cuda:0
```

To estimate another image collection, only change `--input-dir` and, if desired, `--output-dir`. The principal defaults are:

- SAM3 text prompt: `vehicle`
- candidate area weight: `0.2`
- reference vehicle height: `145.7` cm
- batch size: `16`
- device: `cuda:0`

Use `python infer.py --help` to view all options.

### 1.3 Inference output

```text
runs/inference/demo/
├── segmentation/
└── matching/
    ├── predictions.json
    ├── matched_info/
    ├── candidate_bboxes/
    ├── masks/
    └── save_pairs/
```

`matching/predictions.json` is the primary machine-readable result and contains one flood-depth estimate in centimeters for each successfully processed image. The JSON files in `matched_info/` retain all candidates, matching scores, area scores, and selected reference images.

## 2. Reproduce the paper results

Paper reproduction requires both evaluation datasets, all three retrieval datasets (`T=5,10,20`), and the baseline assets from the data bundle. The ground-truth files are included in [`scripts/analyze_results_paper/gt`](scripts/analyze_results_paper/gt).

### 2.1 Proposed method: main result

The main paper configuration uses `T=20` and `lambda_area=0.2`. Run it on both evaluation datasets:

```bash
python infer.py \
    --input-dir datasets/evaluation/real_world/images \
    --retrieval-images-dir datasets/retrieval/images/10layer_72cam_20depths_hdr_light/no_bg_cropped \
    --retrieval-features-dir datasets/retrieval/features/10layer_72cam_20depths_hdr_light \
    --output-dir runs/paper/ours/real_world \
    --area-weight 0.2

python infer.py \
    --input-dir datasets/evaluation/synthetic/images \
    --retrieval-images-dir datasets/retrieval/images/10layer_72cam_20depths_hdr_light/no_bg_cropped \
    --retrieval-features-dir datasets/retrieval/features/10layer_72cam_20depths_hdr_light \
    --output-dir runs/paper/ours/synthetic \
    --area-weight 0.2
```

Compute the paper error metrics for these predictions:

```bash
python scripts/analyze_results_paper/analyze_results.py \
    --run-dir runs/paper/ours \
    --output-dir runs/paper/ours/analysis
```

This reports per-image absolute errors, MAE, population standard deviation, RMSE, signed-error bias, and prediction availability. See the [analysis README](scripts/analyze_results_paper/README.md) for the formulas and output layout.

### 2.2 Ablation study

The complete ablation evaluates `T = 5, 10, 20` and `lambda_area = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0` on both evaluation datasets:

```bash
conda activate fd_retrieval_env
bash scripts/ablation/ablation.sh
```

SAM3 segmentation is cached once per evaluation dataset and reused for all 18 parameter configurations. The complete outputs and the final CSV, JSON, and Markdown summaries are written to `runs/paper/ablation/`. Completed stages are skipped, so the same command can resume an interrupted experiment.

Matching time depends on the CPU, BLAS implementation, thread count, and system load. Accuracy should be reproducible, but timing need not exactly match the paper unless hardware and runtime conditions are controlled. Full options and the output tree are documented in the [ablation README](scripts/ablation/README.md).

### 2.3 Baselines

The repository provides inference entries for our unofficial Park et al. reimplementation and FLOOD-DEPTH-ML. DSC-YOLOv8n is represented by the saved prediction JSON files under `runs/paper/baselines/` because its implementation, training dataset, and reproduction checkpoint are not redistributed. FloodDepth-GPT is represented by the raw saved responses and parsed result JSON files under `runs/paper/baselines/`; it does not require another API call.

Baseline dependencies differ from those of the proposed method. Use a separate environment for each executable baseline and consult its method documentation:

| Method                                        | Documentation or saved results                                                                                                                      | Required assets                                       |
| --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| Park et al. (our unofficial reimplementation) | [`scripts/baselines/park/README.md`](scripts/baselines/park/README.md)                                                                             | `data/baselines/park/target_images/`                |
| DSC-YOLOv8n                                   | [`scripts/baselines/dsc_yolov8n/README.md`](scripts/baselines/dsc_yolov8n/README.md); predictions under `runs/paper/baselines/dsc_yolov8n/`      | Original code/data must be requested from its authors |
| FLOOD-DEPTH-ML                                | [`scripts/baselines/flood_depth_ml/README.md`](scripts/baselines/flood_depth_ml/README.md)                                                         | `data/baselines/flood_depth_ml/`                    |
| FloodDepth-GPT                                | [`scripts/baselines/flooddepth_gpt/README.md`](scripts/baselines/flooddepth_gpt/README.md); results under `runs/paper/baselines/flooddepth_gpt/` | Saved responses and parsed results                    |

Run Park et al. on both datasets:

```bash
conda activate park

python scripts/baselines/park/infer.py \
    --input-dir datasets/evaluation/real_world/images \
    --ground-truth scripts/analyze_results_paper/gt/real_world_gt.json \
    --output-dir runs/paper/baselines/park/real_world

python scripts/baselines/park/infer.py \
    --input-dir datasets/evaluation/synthetic/images \
    --ground-truth scripts/analyze_results_paper/gt/synthetic_gt.json \
    --output-dir runs/paper/baselines/park/synthetic
```

The saved DSC-YOLOv8n predictions used in the paper are stored under `runs/paper/baselines/dsc_yolov8n/`; no DSC inference command is provided.

Run FLOOD-DEPTH-ML on both datasets:

```bash
conda env create -f scripts/baselines/flood_depth_ml/environment.yml
conda activate fd_ml_env

python scripts/baselines/flood_depth_ml/infer.py \
    --input-dir datasets/evaluation/real_world/images \
    --ground-truth scripts/analyze_results_paper/gt/real_world_gt.json \
    --output-dir runs/paper/baselines/flood_depth_ml/real_world \
    --device 0

python scripts/baselines/flood_depth_ml/infer.py \
    --input-dir datasets/evaluation/synthetic/images \
    --ground-truth scripts/analyze_results_paper/gt/synthetic_gt.json \
    --output-dir runs/paper/baselines/flood_depth_ml/synthetic \
    --device 0
```

The saved DSC-YOLOv8n results and the FLOOD-DEPTH-ML inference follow the paper protocol when multiple vehicles are detected: all candidates are recorded, and the candidate with the smallest absolute error to ground truth supplies the image-level prediction. Method-specific details are provided in the corresponding README files listed above.

### 2.4 Generate the paper comparison

After the proposed-method and baseline runs are present under `runs/paper/`, launch:

```bash
jupyter notebook scripts/analyze_results_paper/analysis_main_v3.ipynb
```

The notebook compares FloodDepth-GPT, Park et al., DSC-YOLOv8n, FLOOD-DEPTH-ML, and the proposed method. It reads the curated paper results from `runs/paper/` and writes comparison figures and tables to `runs/paper/analysis/`.

## 3. Build the datasets

Prebuilt datasets are provided in the downloadable data bundle. The following tools are included for researchers who want to reconstruct the retrieval dataset or reproduce the real-world annotation procedure.

### 3.1 Build the retrieval dataset

This pipeline renders the reference vehicle (`2022_peugeot_308.glb`) in Blender, creates cropped images and masks, and extracts VGG19 pool4 features:

```bash
conda activate fd_retrieval_env
python scripts/build_retrieval_dataset/build.py \
    --depth-levels 20 \
    --layers 10 \
    --cameras-per-layer 72 \
    --device cuda:0
```

The reference GLB model and HDRI must be available under `data/build_retrieval_dataset/`. Change `--depth-levels` to `5` or `10` to construct the other ablation datasets. Blender requirements, stage-resume commands, and output paths are documented in the [retrieval-dataset README](scripts/build_retrieval_dataset/README.md).

### 3.2 Build the real-world dataset

The real-world dataset was collected with a FastAPI/Three.js annotation app. Participants select a similar 3D vehicle and align a virtual water plane with the waterline in each image.

```bash
conda activate fd_retrieval_env
python -m pip install -r scripts/build_real_world_dataset/annotation_app/requirements.txt
python scripts/build_real_world_dataset/annotation_app/main.py
```

Open [http://127.0.0.1:8080](http://127.0.0.1:8080) after starting the server. Annotation images and web assets must be placed under `data/build_real_world_dataset/`. Responses are stored in a local SQLite database and excluded from Git. See the [real-world dataset README](scripts/build_real_world_dataset/README.md) and [annotation-app README](scripts/build_real_world_dataset/annotation_app/README.md) for the data layout and optional path overrides.

## License

Except for the third-party components listed below, the original code and documentation in this repository are licensed under the [Apache License 2.0](LICENSE).

Third-party components retain their original licenses:

- `flooddepth_retrieval/sam3/` is distributed under the [SAM License](flooddepth_retrieval/sam3/LICENSE).
- The original FLOOD-DEPTH-ML application preserved in `scripts/baselines/flood_depth_ml/flooddepthml.py` is distributed under its original [Apache License 2.0](scripts/baselines/flood_depth_ml/LICENSE).

The root license does not replace or modify third-party licenses. Dataset, model-weight, and other separately distributed assets may be subject to additional terms described in their corresponding README files. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for details.
