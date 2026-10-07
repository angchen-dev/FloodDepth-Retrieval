#!/usr/bin/env python3
"""Run FLOOD-DEPTH-ML inference on every image in a directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
DEFAULT_WEIGHTS = (
    REPO_ROOT / "data" / "baselines" / "flood_depth_ml" / "best_car.pt"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "runs" / "paper" / "baselines" / "flood_depth_ml"
)

# Representative depths used in the paper comparison: midpoints for Levels
# 0--3 and the lower bound for the open-ended Level 4 range (>105 cm).
LEVEL_DEPTH_CM = {0: 0.0, 1: 17.5, 2: 55.0, 3: 90.0, 4: 105.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=True,
        help="JSON mapping each input image name to an object containing abs (cm).",
    )
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="0", help="Ultralytics device, e.g. 0 or cpu.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--save-images",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save annotated images and YOLO label files.",
    )
    return parser.parse_args()


def require_path(path: Path, *, directory: bool, label: str) -> Path:
    path = path.resolve()
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
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
    from ultralytics import YOLO

    input_dir = require_path(args.input_dir, directory=True, label="Input directory")
    ground_truth_path = require_path(
        args.ground_truth, directory=False, label="Ground-truth file"
    )
    with ground_truth_path.open(encoding="utf-8") as handle:
        ground_truth = json.load(handle)
    weights = require_path(args.weights, directory=False, label="Model weights")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    results = model.predict(
        source=str(input_dir),
        imgsz=args.imgsz,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        rect=False,
        stream=True,
        verbose=False,
        save=args.save_images,
        save_txt=args.save_images,
        save_conf=args.save_images,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
    )

    predictions: list[dict[str, object]] = []
    for result in results:
        image_name = Path(result.path).name
        if image_name not in ground_truth:
            raise KeyError(f"Ground truth is missing for {image_name}")
        gt_entry = ground_truth[image_name]
        gt_depth_cm = float(gt_entry["abs"] if isinstance(gt_entry, dict) else gt_entry)
        class_ids = result.boxes.cls.int().cpu().tolist()
        confidences = result.boxes.conf.cpu().tolist()
        candidates = [
            {
                "class_id": class_id,
                "confidence": round(float(confidence), 6),
                "flood_depth_cm": LEVEL_DEPTH_CM[class_id],
            }
            for class_id, confidence in zip(class_ids, confidences)
            if class_id in LEVEL_DEPTH_CM
        ]
        record: dict[str, object] = {
            "image_name": image_name,
            "status": "no_detection",
            "ground_truth_cm": gt_depth_cm,
            "candidates": candidates,
        }
        if candidates:
            selected_index = min(
                range(len(candidates)),
                key=lambda index: abs(
                    float(candidates[index]["flood_depth_cm"]) - gt_depth_cm
                ),
            )
            selected = candidates[selected_index]
            record.update(
                {
                    "status": "success",
                    "flood_depth_cm": selected["flood_depth_cm"],
                    "selected_candidate_index": selected_index,
                    "selected_class_id": selected["class_id"],
                    "confidence": selected["confidence"],
                    "absolute_error_cm": round(
                        abs(float(selected["flood_depth_cm"]) - gt_depth_cm), 4
                    ),
                }
            )
        predictions.append(record)
        print(f"{image_name}: {record['status']}")

    successful = sum(item["status"] == "success" for item in predictions)
    payload = {
        "method": "FLOOD-DEPTH-ML",
        "input_dir": recorded_path(input_dir),
        "weights": recorded_path(weights),
        "ground_truth": recorded_path(ground_truth_path),
        "level_depth_cm": LEVEL_DEPTH_CM,
        "selection_rule": "candidate with minimum absolute error to ground truth (paper protocol)",
        "number_of_images": len(predictions),
        "number_of_predictions": successful,
        "predictions": predictions,
    }
    output_path = output_dir / "predictions.json"
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {successful}/{len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
