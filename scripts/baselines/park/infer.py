#!/usr/bin/env python3
"""Run the Park et al. baseline on every image in a directory."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_REFERENCE_DIR = REPO_ROOT / "data" / "baselines" / "park" / "target_images"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "runs" / "paper" / "baselines" / "park"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}

sys.path.insert(0, str(SCRIPT_DIR))
from utils import (  # noqa: E402
    DFSSImageRetrieval,
    apply_maskrcnn,
    crop_and_resize_with_margin,
    filter_candidates_by_wh_ratio,
    filter_candidates_on_edge,
    filter_occlusion,
    load_image,
    load_mask_rcnn,
    save_draw_bbox,
    save_matched_result,
    select_largest_vehicle,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="JSON mapping each input image name to an object containing abs (cm).",
    )
    parser.add_argument("--reference-dir", type=Path, default=DEFAULT_REFERENCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device, for example cuda:0 or cpu (default: auto).",
    )
    parser.add_argument("--score-threshold", type=float, default=0.9)
    parser.add_argument("--mask-threshold", type=float, default=0.2)
    parser.add_argument(
        "--ratio-threshold",
        type=float,
        default=1.42,
        help="Minimum vehicle mask width/height ratio used in the paper.",
    )
    return parser.parse_args()


def image_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def require_directory(path: Path, label: str) -> Path:
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def recorded_path(path: Path) -> str:
    """Use a repository-relative path when writing portable result JSON."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    args = parse_args()
    input_dir = require_directory(args.input_dir, "Input directory")
    ground_truth_path = args.ground_truth.resolve()
    if not ground_truth_path.is_file():
        raise FileNotFoundError(
            f"Ground-truth file does not exist: {ground_truth_path}"
        )
    with ground_truth_path.open(encoding="utf-8") as handle:
        ground_truth = json.load(handle)

    reference_dir = require_directory(args.reference_dir, "Reference directory")
    reference_images_dir = require_directory(
        reference_dir / "images", "Reference image directory"
    )
    scale_path = reference_dir / "cm_per_pixel.json"
    if not scale_path.is_file():
        raise FileNotFoundError(f"Reference scale file does not exist: {scale_path}")

    inputs = image_paths(input_dir)
    references = image_paths(reference_images_dir)
    if not inputs:
        raise RuntimeError(f"No supported images found in {input_dir}")
    if not references:
        raise RuntimeError(f"No reference images found in {reference_images_dir}")
    missing_ground_truth = [path.name for path in inputs if path.name not in ground_truth]
    if missing_ground_truth:
        raise KeyError(
            f"Ground truth is missing for: {missing_ground_truth[:5]}"
        )

    with scale_path.open(encoding="utf-8") as handle:
        cm_per_pixel = json.load(handle)

    missing_scales = [path.name for path in references if path.name not in cm_per_pixel]
    if missing_scales:
        raise KeyError(f"Missing cm-per-pixel values for: {missing_scales[:5]}")

    device = (
        "cuda:0" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )
    output_dir = args.output_dir.resolve()
    bbox_dir = output_dir / "draw_bbox"
    pair_dir = output_dir / "matched_pairs"
    bbox_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Mask R-CNN and VGG19 on {device}...")
    maskrcnn_model = load_mask_rcnn(device)
    retrieval_system = DFSSImageRetrieval(device=device)
    reference_arrays = [np.asarray(Image.open(path).convert("RGB")) for path in references]
    print(f"Extracting features for {len(references)} reference images...")
    reference_features = [
        retrieval_system._extract_features(image) for image in reference_arrays
    ]

    predictions: list[dict[str, object]] = []
    for index, image_path in enumerate(inputs, start=1):
        print(f"[{index}/{len(inputs)}] {image_path.name}")
        gt_entry = ground_truth[image_path.name]
        gt_depth_cm = float(gt_entry["abs"] if isinstance(gt_entry, dict) else gt_entry)
        record: dict[str, object] = {
            "image_name": image_path.name,
            "status": "failed",
            "ground_truth_cm": gt_depth_cm,
        }
        try:
            image, image_tensor = load_image(str(image_path))
            detections = apply_maskrcnn(maskrcnn_model, image_tensor)
            filtered = filter_occlusion(
                image,
                detections,
                mask_threshold=args.mask_threshold,
                score_thresh=args.score_threshold,
            )
            if not filtered:
                record["reason"] = "no_unoccluded_vehicle"
                predictions.append(record)
                continue

            candidates, draw_all, draw_occlusion, _ = filtered
            candidates, draw_edge = filter_candidates_on_edge(candidates, image)
            candidates, draw_ratio = filter_candidates_by_wh_ratio(
                candidates,
                image,
                ratio_thresh=args.ratio_threshold,
            )
            candidates, draw_final = select_largest_vehicle(candidates, image)
            save_draw_bbox(
                draw_all,
                draw_occlusion,
                draw_edge,
                draw_ratio,
                draw_final,
                bbox_dir / image_path.name,
            )
            if not candidates:
                record["reason"] = "no_vehicle_after_filtering"
                predictions.append(record)
                continue

            query_array, _, _ = crop_and_resize_with_margin(
                image, candidates[0]["mask"]
            )
            indices, similarities, pixel_difference = retrieval_system.image_retrieval(
                query_array,
                reference_arrays,
                reference_features,
                top_k=1,
            )
            if not indices:
                record["reason"] = "retrieval_similarity_below_threshold"
                predictions.append(record)
                continue

            reference_path = references[indices[0]]
            depth_cm = max(
                0.0,
                round(
                    float(pixel_difference)
                    * float(cm_per_pixel[reference_path.name]),
                    4,
                ),
            )
            record.update(
                {
                    "status": "success",
                    "flood_depth_cm": depth_cm,
                    "matched_reference_image": reference_path.name,
                    "similarity": round(float(similarities[0]), 6),
                    "absolute_error_cm": round(abs(depth_cm - gt_depth_cm), 4),
                }
            )
            save_matched_result(
                query_array,
                reference_arrays[indices[0]],
                f"Estimated flood depth: {depth_cm} cm",
                pair_dir / image_path.name,
            )
        except Exception as exc:
            record["reason"] = f"{type(exc).__name__}: {exc}"
        predictions.append(record)

    successful = sum(item["status"] == "success" for item in predictions)
    payload = {
        "method": "Park et al. reimplementation",
        "input_dir": recorded_path(input_dir),
        "reference_dir": recorded_path(reference_dir),
        "ground_truth": recorded_path(ground_truth_path),
        "device": device,
        "number_of_images": len(inputs),
        "number_of_predictions": successful,
        "predictions": predictions,
    }
    output_path = output_dir / "predictions.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {successful}/{len(inputs)} predictions to {output_path}")


if __name__ == "__main__":
    main()
