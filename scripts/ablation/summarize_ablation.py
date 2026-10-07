#!/usr/bin/env python3
"""Aggregate ablation metrics and matching times into paper-ready tables."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_T_VALUES = (5, 10, 20)
DEFAULT_AREA_WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "paper" / "ablation",
    )
    parser.add_argument(
        "--t-values", type=int, nargs="+", default=list(DEFAULT_T_VALUES)
    )
    parser.add_argument(
        "--area-weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_AREA_WEIGHTS),
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def run_dir(runs_dir: Path, t_value: int, area_weight: float) -> Path:
    name = f"10layer_72cam_{t_value}depths_hdr_light"
    return runs_dir / name / f"lambda_{area_weight:.1f}"


def main() -> None:
    args = parse_args()
    runs_dir = args.runs_dir.resolve()
    rows: list[dict[str, object]] = []
    missing: list[Path] = []

    for t_value in args.t_values:
        for area_weight in args.area_weights:
            current_run = run_dir(runs_dir, t_value, area_weight)
            summary_path = current_run / "paper_analysis" / "summary.json"
            timing_paths = {
                dataset: current_run
                / dataset
                / "matching"
                / "analysis"
                / "processing_time.json"
                for dataset in ("real_world", "synthetic")
            }
            required = [summary_path, *timing_paths.values()]
            absent = [path for path in required if not path.is_file()]
            if absent:
                missing.extend(absent)
                continue

            summary = read_json(summary_path)
            real = summary["real_world"]
            synthetic = summary["synthetic"]
            real_timing = read_json(timing_paths["real_world"])
            synthetic_timing = read_json(timing_paths["synthetic"])
            rows.append(
                {
                    "t": t_value,
                    "lambda_area": area_weight,
                    "real_world_mae_cm": real["mae_cm"],
                    "real_world_std_cm": real["absolute_error_std_cm"],
                    "real_world_valid_predictions": real["valid_predictions"],
                    "synthetic_mae_cm": synthetic["mae_cm"],
                    "synthetic_std_cm": synthetic["absolute_error_std_cm"],
                    "synthetic_valid_predictions": synthetic[
                        "valid_predictions"
                    ],
                    "real_world_matching_ms_per_query": round(
                        float(
                            real_timing[
                                "average_matching_time_per_candidate_query_seconds"
                            ]
                        ),
                        6,
                    )
                    * 1000.0,
                    "synthetic_matching_ms_per_query": round(
                        float(
                            synthetic_timing[
                                "average_matching_time_per_candidate_query_seconds"
                            ]
                        ),
                        6,
                    )
                    * 1000.0,
                }
            )

    if missing:
        preview = "\n".join(f"  - {path}" for path in missing[:12])
        suffix = "\n  ..." if len(missing) > 12 else ""
        raise SystemExit(
            "Ablation is incomplete; required outputs are missing:\n"
            f"{preview}{suffix}"
        )
    if not rows:
        raise SystemExit(f"No ablation results found under {runs_dir}")

    timing_by_t: dict[int, dict[str, float]] = {}
    for t_value in args.t_values:
        t_rows = [row for row in rows if row["t"] == t_value]
        timing_by_t[t_value] = {
            "real_world_mean_matching_ms_per_query": statistics.fmean(
                float(row["real_world_matching_ms_per_query"])
                for row in t_rows
            ),
            "synthetic_mean_matching_ms_per_query": statistics.fmean(
                float(row["synthetic_matching_ms_per_query"])
                for row in t_rows
            ),
        }

    for row in rows:
        means = timing_by_t[int(row["t"])]
        row.update(means)

    output = {
        "t_values": args.t_values,
        "area_weights": args.area_weights,
        "matching_time_definition": (
            "Cosine-similarity matrix time averaged per candidate query; "
            "the per-T mean is averaged across area-weight runs."
        ),
        "mean_matching_time_by_t": timing_by_t,
        "results": rows,
    }
    json_path = runs_dir / "ablation_summary.json"
    csv_path = runs_dir / "ablation_summary.csv"
    markdown_path = runs_dir / "ablation_summary.md"
    runs_dir.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# Ablation summary",
        "",
        "> **Timing note:** Matching time is hardware- and runtime-dependent. ",
        "> CPU, BLAS implementation, thread count, memory state, and system ",
        "> load can change measured values even when the Conda environment ",
        "> and code are identical. Accuracy should be reproducible, but exact ",
        "> timing values are not expected to match the paper across runs.",
        "",
        "| T | lambda_area | Real-world MAE ± STD (cm) | "
        "Synthetic MAE ± STD (cm) | Real matching (ms/query) | "
        "Synthetic matching (ms/query) |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        means = timing_by_t[int(row["t"])]
        markdown.append(
            f"| {row['t']} | {float(row['lambda_area']):.1f} | "
            f"{float(row['real_world_mae_cm']):.2f} ± "
            f"{float(row['real_world_std_cm']):.2f} | "
            f"{float(row['synthetic_mae_cm']):.2f} ± "
            f"{float(row['synthetic_std_cm']):.2f} | "
            f"{means['real_world_mean_matching_ms_per_query']:.2f} | "
            f"{means['synthetic_mean_matching_ms_per_query']:.2f} |"
        )
    markdown_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(f"Saved {len(rows)} ablation configurations to {runs_dir}")


if __name__ == "__main__":
    main()
