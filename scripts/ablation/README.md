# Paper ablation study

This directory reproduces the paper ablation over:

- flood-depth intervals: `T = 5, 10, 20`
- candidate area weights: `lambda_area = 0.0, 0.2, 0.4, 0.6, 0.8, 1.0`
- evaluation datasets: real-world and synthetic

The full study contains 18 parameter configurations and 36 retrieval runs.
SAM3 segmentation is computed only once per evaluation dataset and reused by
all retrieval runs. This is equivalent to the original experiment structure
and avoids repeating identical segmentation work.

## Run

From the repository root, activate the project environment and run:

```bash
conda activate fd_retrieval_env
bash scripts/ablation/ablation.sh
```

To inspect all commands without starting GPU inference:

```bash
bash scripts/ablation/ablation.sh --dry-run
```

Completed segmentation and matching outputs are skipped automatically, so an
interrupted study can be resumed with the same command. Use
`--rerun-segmentation` or `--rerun-matching` when those stages must be replaced.
Additional arguments such as `--device cuda:1` and `--batch-size 8` are passed
to the Python runner. The launcher uses the `python` executable from the
currently activated environment; set `PYTHON` explicitly when needed.

The shell launcher runs `run_ablation.py`, which performs all stages in
sequence: it creates or reuses the SAM3 segmentation cache, runs matching for
every parameter configuration, calls
`scripts/analyze_results_paper/analyze_results.py` for each configuration,
and finally calls `summarize_ablation.py` to generate the combined CSV, JSON,
and Markdown summaries.

## Output layout

```text
runs/paper/ablation/
├── _segmentation/
│   ├── real_world/
│   └── synthetic/
├── 10layer_72cam_5depths_hdr_light/
│   ├── lambda_0.0/
│   │   ├── real_world/matching/
│   │   ├── synthetic/matching/
│   │   └── paper_analysis/
│   └── ...
├── 10layer_72cam_10depths_hdr_light/
├── 10layer_72cam_20depths_hdr_light/
├── ablation_summary.json
├── ablation_summary.csv
└── ablation_summary.md
```

For every retrieval run, `matching/analysis/processing_time.json` records only
the cosine-similarity matrix time and its average per candidate query. The
summary reports the mean matching time across the six area-weight runs for
each value of `T`, matching the definition used in the paper table.

Matching time depends on the CPU, BLAS implementation, thread count, and
system load. Re-running the study is expected to reproduce the accuracy table,
but not necessarily the exact timing measured for the paper. Timing results
should therefore be compared only under controlled hardware and runtime
conditions.

The scripts call the existing segmentation, retrieval, and paper-analysis
implementations directly. The public `infer.py` entry point is unchanged.
