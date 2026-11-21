#!/usr/bin/env python3
"""Generate explicit synthetic ground truth from the dataset filenames."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_IMAGE_DIR = (
    PROJECT_ROOT / "data" / "evaluation_dataset" / "synthetic" / "images"
)
DEFAULT_OUTPUT = SCRIPT_DIR / "gt" / "synthetic_gt.json"
REFERENCE_HEIGHT_CM = 145.7
CAR_HEIGHT_CM = {
    "2021_hyundai_encino_ev": 158.0,
    "2020_skoda_kamiq_gt": 153.1,
    "bmw_m6_gran_coupe": 139.5,
    "2021_volkswagen_passat_330_tsi": 151.6,
    "2021_volkwagen_id.3": 156.2,
    "2020_nissan_sentra_sylphy": 165.9,
    "2021_volkswagen_cross_touran_l_280_tsi": 144.5,
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FILENAME_PATTERN = re.compile(
    r"^10layer_18cam_(.+?)_level-([0-9]*\.?[0-9]+)(?:_|\.|$)"
)


def flood_level(depth_cm: float, intervals: int = 10) -> int:
    if depth_cm <= 0:
        return 0
    interval_cm = REFERENCE_HEIGHT_CM / intervals
    return min(math.ceil(depth_cm / interval_cm), intervals - 1)


def parse_ground_truth(image_name: str) -> dict[str, object]:
    match = FILENAME_PATTERN.match(image_name)
    if match is None:
        raise ValueError(
            "Filename must match "
            "10layer_18cam_<vehicle>_level-<ratio>_...: "
            f"{image_name}"
        )
    vehicle = match.group(1)
    if vehicle not in CAR_HEIGHT_CM:
        raise ValueError(f"Unknown synthetic vehicle: {vehicle}")
    ratio = float(match.group(2))
    height_cm = CAR_HEIGHT_CM[vehicle]
    depth_cm = height_cm * ratio
    return {
        "abs": round(depth_cm, 4),
        "ratio": ratio,
        "flood_level": flood_level(depth_cm),
        "vehicle": vehicle,
        "vehicle_height_cm": height_cm,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.image_dir.is_dir():
        parser.error(f"Image directory does not exist: {args.image_dir}")
    images = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        parser.error(f"No supported images found in {args.image_dir}")

    ground_truth = {
        path.name: parse_ground_truth(path.name)
        for path in images
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(ground_truth, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Saved {len(ground_truth)} annotations to {args.output}")


if __name__ == "__main__":
    main()
