#!/usr/bin/env python3
"""Run the paper ablation over depth intervals and area weights."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS = {
    "real_world": PROJECT_ROOT / "datasets/evaluation/real_world/images",
    "synthetic": PROJECT_ROOT / "datasets/evaluation/synthetic/images",
}
DEFAULT_T_VALUES = (5, 10, 20)
DEFAULT_AREA_WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=PROJECT_ROOT / "runs" / "paper" / "ablation",
        help="Ablation output root (default: runs/paper/ablation).",
    )
    parser.add_argument(
        "--t-values",
        type=int,
        nargs="+",
        default=list(DEFAULT_T_VALUES),
    )
    parser.add_argument(
        "--area-weights",
        type=float,
        nargs="+",
        default=list(DEFAULT_AREA_WEIGHTS),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--prompt", default="vehicle")
    parser.add_argument("--sam-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam-resolution", type=int, default=1008)
    parser.add_argument("--reference-vehicle-height-cm", type=float, default=145.7)
    parser.add_argument("--save-logits", action="store_true")
    parser.add_argument(
        "--rerun-segmentation",
        action="store_true",
        help="Run SAM3 again even when a complete cache is present.",
    )
    parser.add_argument(
        "--rerun-matching",
        action="store_true",
        help="Run retrieval again even when its outputs are present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print commands without running them.",
    )
    return parser.parse_args()


def format_weight(value: float) -> str:
    return f"{value:.1f}"


def retrieval_name(t_value: int) -> str:
    return f"10layer_72cam_{t_value}depths_hdr_light"


def run_command(command: list[str], *, dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def segmentation_is_complete(path: Path) -> bool:
    metadata_dir = path / "metadata"
    return (
        (path / "summary.json").is_file()
        and metadata_dir.is_dir()
        and any(metadata_dir.glob("*.json"))
    )


def matching_is_complete(path: Path) -> bool:
    return (
        (path / "predictions.json").is_file()
        and (path / "analysis" / "processing_time.json").is_file()
    )


def validate_args(args: argparse.Namespace) -> None:
    if not args.t_values or any(value <= 0 for value in args.t_values):
        raise SystemExit("--t-values must contain positive integers")
    if not args.area_weights or any(
        value < 0.0 or value > 1.0 for value in args.area_weights
    ):
        raise SystemExit("--area-weights must be between 0 and 1")
    if args.batch_size <= 0 or args.sam_resolution <= 0:
        raise SystemExit("--batch-size and --sam-resolution must be positive")

    missing = [str(path) for path in DATASETS.values() if not path.is_dir()]
    for t_value in args.t_values:
        name = retrieval_name(t_value)
        for path in (
            PROJECT_ROOT
            / "datasets/retrieval/images"
            / name
            / "no_bg_cropped",
            PROJECT_ROOT / "datasets/retrieval/features" / name,
        ):
            if not path.exists():
                missing.append(str(path))
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise SystemExit(f"Required ablation inputs are missing:\n{details}")


def main() -> None:
    args = parse_args()
    validate_args(args)
    runs_dir = args.runs_dir.resolve()
    python = sys.executable

    segmentation_dirs: dict[str, Path] = {}
    for dataset, input_dir in DATASETS.items():
        segmentation_dir = runs_dir / "_segmentation" / dataset
        segmentation_dirs[dataset] = segmentation_dir
        if args.rerun_segmentation or not segmentation_is_complete(
            segmentation_dir
        ):
            command = [
                python,
                "-m",
                "flooddepth_retrieval.segmentation",
                "--in_dir",
                str(input_dir),
                "--out_dir",
                str(segmentation_dir),
                "--prompt",
                args.prompt,
                "--device",
                args.device,
                "--sam_confidence_threshold",
                str(args.sam_confidence_threshold),
                "--sam_resolution",
                str(args.sam_resolution),
            ]
            if args.save_logits:
                command.append("--save_logits")
            run_command(command, dry_run=args.dry_run)
        else:
            print(f"Skipping complete segmentation cache: {segmentation_dir}")

    for t_value in args.t_values:
        name = retrieval_name(t_value)
        retrieval_images = (
            PROJECT_ROOT
            / "datasets/retrieval/images"
            / name
            / "no_bg_cropped"
        )
        retrieval_features = (
            PROJECT_ROOT / "datasets/retrieval/features" / name
        )

        for area_weight in args.area_weights:
            weight = format_weight(area_weight)
            run_dir = runs_dir / name / f"lambda_{weight}"
            print(f"\n=== T={t_value}, lambda_area={weight} ===", flush=True)

            for dataset in DATASETS:
                matching_dir = run_dir / dataset / "matching"
                if args.rerun_matching or not matching_is_complete(matching_dir):
                    run_command(
                        [
                            python,
                            "scripts/ablation/timed_retrieval.py",
                            "--seg_results_dir",
                            str(segmentation_dirs[dataset]),
                            "--retrieval_dataset_dir",
                            str(retrieval_images),
                            "--vgg_feature_h5_dir",
                            str(retrieval_features),
                            "--out_dir",
                            str(matching_dir),
                            "--device",
                            args.device,
                            "--mask_score_weight",
                            weight,
                            "--batch_size",
                            str(args.batch_size),
                            "--flood_height_cm",
                            str(args.reference_vehicle_height_cm),
                        ],
                        dry_run=args.dry_run,
                    )
                else:
                    print(f"Skipping complete matching run: {matching_dir}")

            run_command(
                [
                    python,
                    "scripts/analyze_results_paper/analyze_results.py",
                    "--run-dir",
                    str(run_dir),
                ],
                dry_run=args.dry_run,
            )

if __name__ == "__main__":
    main()
