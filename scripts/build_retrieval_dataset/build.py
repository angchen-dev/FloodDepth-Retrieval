#!/usr/bin/env python3
"""Build a retrieval dataset by rendering, cropping, and extracting VGG19 features."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_INPUT_DIR = REPO_ROOT / "data/build_retrieval_dataset"
REFERENCE_MODEL = "2022_peugeot_308.glb"
REFERENCE_MODEL_STEM = Path(REFERENCE_MODEL).stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth-levels", type=int, default=20)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--cameras-per-layer", type=int, default=72)
    parser.add_argument("--dataset-name")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_INPUT_DIR / "car_models")
    parser.add_argument("--hdri-path", type=Path, default=DEFAULT_INPUT_DIR / "hdri/pine_picnic_4k.hdr")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-parts", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--without-hdr-light", action="store_true")
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("render", "crop", "features"),
        default=("render", "crop", "features"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def run(command: list[str], dry_run: bool) -> None:
    print(f"+ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def main() -> None:
    args = parse_args()
    for name, value in (
        ("depth-levels", args.depth_levels),
        ("layers", args.layers),
        ("cameras-per-layer", args.cameras_per_layer),
        ("batch-size", args.batch_size),
        ("workers", args.workers),
    ):
        if value <= 0:
            raise SystemExit(f"--{name} must be positive")

    suffix = "wo_light" if args.without_hdr_light else "hdr_light"
    dataset_name = args.dataset_name or (
        f"{args.layers}layer_{args.cameras_per_layer}cam_"
        f"{args.depth_levels}depths_{suffix}"
    )
    parts = args.feature_parts or args.depth_levels
    render_dir = REPO_ROOT / "runs/build_retrieval_dataset/renders" / dataset_name
    images_dir = REPO_ROOT / "datasets/retrieval/images" / dataset_name
    features_dir = REPO_ROOT / "datasets/retrieval/features" / dataset_name

    if args.overwrite and not args.dry_run:
        if "render" in args.stages and render_dir.is_dir():
            shutil.rmtree(render_dir)
        if "crop" in args.stages and images_dir.is_dir():
            shutil.rmtree(images_dir)
        if "features" in args.stages and features_dir.is_dir():
            shutil.rmtree(features_dir)

    if "render" in args.stages:
        command = [
            args.blender,
            "--background",
            "--python",
            str(SCRIPT_DIR / "render.py"),
            "--",
            "--models-dir",
            str(args.models_dir.resolve()),
            "--models",
            REFERENCE_MODEL,
            "--hdri-path",
            str(args.hdri_path.resolve()),
            "--output-dir",
            str(render_dir),
            "--layers",
            str(args.layers),
            "--cameras-per-layer",
            str(args.cameras_per_layer),
            "--depth-levels",
            str(args.depth_levels),
        ]
        if args.without_hdr_light:
            command.append("--without-hdr-light")
        if args.overwrite:
            command.append("--overwrite")
        run(command, args.dry_run)

    if "crop" in args.stages:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "crop.py"),
            "--render-dir",
            str(render_dir),
            "--output-dir",
            str(images_dir),
            "--model",
            REFERENCE_MODEL_STEM,
        ]
        if args.overwrite:
            command.append("--overwrite")
        run(command, args.dry_run)

    if "features" in args.stages:
        command = [
            sys.executable,
            str(SCRIPT_DIR / "extract_features.py"),
            "--images-dir",
            str(images_dir),
            "--output-dir",
            str(features_dir),
            "--device",
            args.device,
            "--batch-size",
            str(args.batch_size),
            "--parts",
            str(parts),
            "--workers",
            str(args.workers),
        ]
        if args.overwrite:
            command.append("--overwrite")
        run(command, args.dry_run)

    print(f"Retrieval images: {images_dir}")
    print(f"VGG19 features: {features_dir}")


if __name__ == "__main__":
    main()
