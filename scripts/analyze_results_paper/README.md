# Paper result analysis

This directory independently computes the flood-depth errors reported in the
paper. It does not import code from `flooddepth_retrieval`.

## Ground truth

- `gt/real_world_gt.json` contains the final aggregated depth annotation for
  each of the 200 real-world images. The `abs` field is the reference depth in
  centimeters.
- `gt/synthetic_gt.json` contains explicit ground truth for the 200 synthetic
  images. It is generated from each filename's vehicle name and `level-<ratio>`
  value using the corresponding real vehicle height.

To regenerate the synthetic file after changing the dataset:

```bash
python scripts/analyze_results_paper/generate_synthetic_gt.py
```

## Compute errors

Store each retrieval-dataset or ablation configuration in a separate run
directory. Each run directory must contain both evaluation datasets:

```text
runs/<experiment-name>/
├── real_world/matching/
└── synthetic/matching/
```

Each `matching/` directory may contain the full `matched_info/` output or only
the compact `predictions.json` summary. The latter is sufficient for the
downloadable paper-results bundle.

This makes it possible to analyze each retrieval-dataset configuration
independently during the ablation study. Use `--output-dir` only when a custom
analysis output location is needed.

Pass the run directory explicitly when computing its errors. For example:

```bash
python scripts/analyze_results_paper/analyze_results.py \
    --run-dir runs/paper/ablation/10layer_72cam_10depths_hdr_light/lambda_0.2
```

The script computes, independently for each dataset:

```text
absolute_error = abs(estimated_depth_cm - ground_truth_depth_cm)
MAE            = mean(absolute_error)
STD            = population standard deviation of absolute_error (ddof=0)
```

It also reports prediction availability, RMSE, and signed-error bias. By
default, outputs are written inside the selected run:

```text
runs/<experiment-name>/paper_analysis/
├── summary.json
├── real_world/
│   ├── metrics.json
│   ├── per_image_errors.csv
│   └── per_image_errors.json
└── synthetic/
    ├── metrics.json
    ├── per_image_errors.csv
    └── per_image_errors.json
```

The main paper results use the retrieval dataset with `T=20` and
`lambda_area=0.2`. Their canonical release location is `runs/paper/ours/`;
complete parameter sweeps are stored under `runs/paper/ablation/`.

## Compare all paper methods

Run `analysis_main_v3.ipynb` to compare FloodDepth-GPT, Park et al.,
DSC-YOLOv8n, FLOOD-DEPTH-ML, and the proposed method. The notebook reads all
curated inputs from:

```text
runs/paper/
```

Launch Jupyter from the repository root or any subdirectory. The notebook
resolves the repository root automatically and contains no machine-specific
paths. Figures and comparison tables are written under
`runs/paper/analysis/`.
