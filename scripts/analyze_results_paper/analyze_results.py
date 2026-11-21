#!/usr/bin/env python3
"""Compute the paper's flood-depth errors for real and synthetic results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_REAL_GT = SCRIPT_DIR / "gt" / "real_world_gt.json"
DEFAULT_SYNTHETIC_GT = SCRIPT_DIR / "gt" / "synthetic_gt.json"


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_estimated_depth(result: dict[str, object]) -> float:
    final_estimate = result.get("final_estimate") or {}
    depth_cm = final_estimate.get("flood_depth_cm")
    if depth_cm is None:
        depth_cm = result.get("flood_depth_cm")
    if depth_cm is None:
        raise ValueError("Estimated flood depth is missing")
    return float(depth_cm)


def analyze_dataset(
    *,
    dataset_name: str,
    result_dir: Path,
    ground_truth_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    matched_info_dir = result_dir / "matched_info"
    predictions_path = result_dir / "predictions.json"
    if matched_info_dir.is_dir():
        result_records = [
            (path.name, read_json(path))
            for path in sorted(matched_info_dir.glob("*.json"))
        ]
    elif predictions_path.is_file():
        payload = read_json(predictions_path)
        predictions = payload.get("predictions") if isinstance(payload, dict) else None
        if not isinstance(predictions, list):
            raise ValueError(f"Invalid predictions file: {predictions_path}")
        result_records = [
            (f"{record.get('image_name', index)}", record)
            for index, record in enumerate(predictions)
            if isinstance(record, dict)
        ]
    else:
        raise FileNotFoundError(
            "Neither matched_info nor predictions.json exists under "
            f"{result_dir}"
        )
    if not ground_truth_path.is_file():
        raise FileNotFoundError(f"Ground truth does not exist: {ground_truth_path}")

    ground_truth = read_json(ground_truth_path)
    if not isinstance(ground_truth, dict) or not ground_truth:
        raise ValueError(f"Invalid ground-truth file: {ground_truth_path}")

    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    predicted_names: set[str] = set()

    for result_name, result in result_records:
        try:
            image_name = str(result["image_name"])
            if result.get("status") not in (None, "success"):
                raise ValueError(str(result.get("reason", result["status"])))
            if image_name in predicted_names:
                raise ValueError(f"Duplicate prediction for {image_name}")
            predicted_names.add(image_name)
            if image_name not in ground_truth:
                raise KeyError(f"Ground truth not found for {image_name}")

            gt_depth_cm = float(ground_truth[image_name]["abs"])
            estimated_depth_cm = get_estimated_depth(result)
            signed_error_cm = estimated_depth_cm - gt_depth_cm
            absolute_error_cm = abs(signed_error_cm)
            rows.append(
                {
                    "image_name": image_name,
                    "gt_depth_cm": round(gt_depth_cm, 4),
                    "estimated_depth_cm": round(estimated_depth_cm, 4),
                    "signed_error_cm": round(signed_error_cm, 4),
                    "absolute_error_cm": round(absolute_error_cm, 4),
                }
            )
        except Exception as exc:
            skipped.append(
                {"result_file": result_name, "reason": str(exc)}
            )

    if not rows:
        raise RuntimeError(f"No valid {dataset_name} predictions were found")

    rows.sort(key=lambda row: str(row["image_name"]))
    absolute_errors = [float(row["absolute_error_cm"]) for row in rows]
    signed_errors = [float(row["signed_error_cm"]) for row in rows]
    gt_names = set(ground_truth)
    valid_names = {str(row["image_name"]) for row in rows}
    missing_predictions = sorted(gt_names - valid_names)

    metrics = {
        "dataset": dataset_name,
        "ground_truth_images": len(ground_truth),
        "result_files": len(result_records),
        "valid_predictions": len(rows),
        "prediction_availability_percent": round(
            100.0 * len(rows) / len(ground_truth), 4
        ),
        "mae_cm": round(statistics.fmean(absolute_errors), 4),
        "absolute_error_std_cm": round(
            statistics.pstdev(absolute_errors), 4
        ),
        "rmse_cm": round(
            statistics.fmean(error * error for error in signed_errors) ** 0.5,
            4,
        ),
        "bias_cm": round(statistics.fmean(signed_errors), 4),
        "missing_predictions": len(missing_predictions),
        "skipped_result_files": len(skipped),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "per_image_errors.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "missing_predictions.json").write_text(
        json.dumps(missing_predictions, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "skipped_results.json").write_text(
        json.dumps(skipped, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "per_image_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"{dataset_name}: {len(rows)}/{len(ground_truth)} predictions, "
        f"MAE={metrics['mae_cm']:.2f} cm, "
        f"STD={metrics['absolute_error_std_cm']:.2f} cm"
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help=(
            "Ablation run directory containing real_world/matching and "
            "synthetic/matching."
        ),
    )
    parser.add_argument("--real-gt", type=Path, default=DEFAULT_REAL_GT)
    parser.add_argument(
        "--synthetic-gt", type=Path, default=DEFAULT_SYNTHETIC_GT
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <run-dir>/paper_analysis).",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else run_dir / "paper_analysis"
    )

    real_metrics = analyze_dataset(
        dataset_name="real_world",
        result_dir=run_dir / "real_world" / "matching",
        ground_truth_path=args.real_gt.resolve(),
        output_dir=output_dir / "real_world",
    )
    synthetic_metrics = analyze_dataset(
        dataset_name="synthetic",
        result_dir=run_dir / "synthetic" / "matching",
        ground_truth_path=args.synthetic_gt.resolve(),
        output_dir=output_dir / "synthetic",
    )
    try:
        recorded_run_dir = str(run_dir.relative_to(PROJECT_ROOT))
    except ValueError:
        recorded_run_dir = str(run_dir)

    summary = {
        "run_dir": recorded_run_dir,
        "real_world": real_metrics,
        "synthetic": synthetic_metrics,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved paper-error analysis to {output_dir}")


if __name__ == "__main__":
    main()
