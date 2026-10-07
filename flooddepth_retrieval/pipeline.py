"""End-to-end SAM3 and VGG19 flood-depth inference pipeline."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _run_command(command: list[str]) -> None:
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


@dataclass(frozen=True)
class InferenceConfig:
    """Paths and paper-default parameters for one directory inference run."""

    input_dir: Path
    output_dir: Path = PROJECT_ROOT / "runs"
    retrieval_images_dir: Path = (
        PROJECT_ROOT / "data" / "retrieval_dataset" / "images"
    )
    retrieval_features_dir: Path = (
        PROJECT_ROOT / "data" / "retrieval_dataset" / "features"
    )
    device: str = "cuda:0"
    prompt: str = "vehicle"
    sam_confidence_threshold: float = 0.5
    sam_resolution: int = 1008
    area_weight: float = 0.2
    batch_size: int = 16
    reference_vehicle_height_cm: float = 145.7
    save_logits: bool = False


def _validate(config: InferenceConfig) -> None:
    for label, path in (
        ("input directory", config.input_dir),
        ("retrieval image directory", config.retrieval_images_dir),
    ):
        if not path.is_dir():
            raise ValueError(f"Missing {label}: {path}")
    if not config.retrieval_features_dir.exists():
        raise ValueError(
            "Missing retrieval feature path: "
            f"{config.retrieval_features_dir}"
        )

    images = [
        path
        for path in config.input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not images:
        raise ValueError(f"No supported images found in {config.input_dir}")
    retrieval_images = [
        path
        for path in config.retrieval_images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    if not retrieval_images:
        raise ValueError(
            "No retrieval images found in "
            f"{config.retrieval_images_dir}"
        )
    feature_path = config.retrieval_features_dir
    feature_files = (
        [feature_path]
        if feature_path.is_file() and feature_path.suffix.lower() == ".h5"
        else list(feature_path.glob("*.h5"))
        if feature_path.is_dir()
        else []
    )
    if not feature_files:
        raise ValueError(f"No HDF5 retrieval features found at {feature_path}")
    if not 0.0 <= config.sam_confidence_threshold <= 1.0:
        raise ValueError("SAM3 confidence threshold must be between 0 and 1")
    if not 0.0 <= config.area_weight <= 1.0:
        raise ValueError("Area weight must be between 0 and 1")
    if config.sam_resolution <= 0 or config.batch_size <= 0:
        raise ValueError("SAM3 resolution and batch size must be positive")


def run_inference(config: InferenceConfig) -> Path:
    """Run segmentation followed by retrieval for an image directory."""
    config = replace(
        config,
        input_dir=config.input_dir.resolve(),
        output_dir=config.output_dir.resolve(),
        retrieval_images_dir=config.retrieval_images_dir.resolve(),
        retrieval_features_dir=config.retrieval_features_dir.resolve(),
    )
    _validate(config)

    segmentation_dir = config.output_dir / "segmentation"
    matching_dir = config.output_dir / "matching"
    config.output_dir.mkdir(parents=True, exist_ok=True)

    print("[1/2] Segmenting vehicles with SAM3...", flush=True)
    segmentation_command = [
        sys.executable,
        "-m",
        "flooddepth_retrieval.segmentation",
        "--in_dir",
        str(config.input_dir),
        "--out_dir",
        str(segmentation_dir),
        "--prompt",
        config.prompt,
        "--device",
        config.device,
        "--sam_confidence_threshold",
        str(config.sam_confidence_threshold),
        "--sam_resolution",
        str(config.sam_resolution),
    ]
    if config.save_logits:
        segmentation_command.append("--save_logits")
    _run_command(segmentation_command)

    print(
        "[2/2] Matching vehicles with the VGG19 retrieval dataset...",
        flush=True,
    )
    _run_command(
        [
            sys.executable,
            "-m",
            "flooddepth_retrieval.retrieval",
            "--seg_results_dir",
            str(segmentation_dir),
            "--retrieval_dataset_dir",
            str(config.retrieval_images_dir),
            "--vgg_feature_h5_dir",
            str(config.retrieval_features_dir),
            "--out_dir",
            str(matching_dir),
            "--device",
            config.device,
            "--mask_score_weight",
            str(config.area_weight),
            "--batch_size",
            str(config.batch_size),
            "--flood_height_cm",
            str(config.reference_vehicle_height_cm),
        ]
    )
    print(f"Inference complete. Results: {config.output_dir}", flush=True)
    return config.output_dir
