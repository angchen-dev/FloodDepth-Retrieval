#!/usr/bin/env python3
"""Estimate flood depth for every image in an input directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from flooddepth_retrieval import InferenceConfig, run_inference
from flooddepth_retrieval.pipeline import PROJECT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing input JPG, PNG, or WebP images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "runs",
        help="Output directory (default: runs).",
    )
    parser.add_argument(
        "--retrieval-images-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "retrieval_dataset" / "images",
        help="Directory containing rendered retrieval images.",
    )
    parser.add_argument(
        "--retrieval-features-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "retrieval_dataset" / "features",
        help="Directory containing precomputed VGG19 HDF5 features.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="vehicle")
    parser.add_argument("--sam-confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam-resolution", type=int, default=1008)
    parser.add_argument(
        "--area-weight",
        type=float,
        default=0.2,
        help="Candidate-selection area weight (paper setting: 0.2).",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--reference-vehicle-height-cm",
        type=float,
        default=145.7,
    )
    parser.add_argument("--save-logits", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        run_inference(
            InferenceConfig(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                retrieval_images_dir=args.retrieval_images_dir,
                retrieval_features_dir=args.retrieval_features_dir,
                device=args.device,
                prompt=args.prompt,
                sam_confidence_threshold=args.sam_confidence_threshold,
                sam_resolution=args.sam_resolution,
                area_weight=args.area_weight,
                batch_size=args.batch_size,
                reference_vehicle_height_cm=(
                    args.reference_vehicle_height_cm
                ),
                save_logits=args.save_logits,
            )
        )
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"Error: {exc}") from exc


if __name__ == "__main__":
    main()
