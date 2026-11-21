#!/usr/bin/env python3
"""Crop Blender renders and create masks used by the retrieval dataset."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm


def create_vehicle_mask(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([80, 255, 255]))
    red_1 = cv2.inRange(hsv, np.array([0, 80, 80]), np.array([10, 255, 255]))
    red_2 = cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    non_red = cv2.bitwise_not(cv2.bitwise_or(red_1, red_2))
    background = cv2.bitwise_and(green, non_red)
    return cv2.bitwise_not(background)


def process_pair(with_cube: Path, without_cube: Path, output_dir: Path, name: str) -> bool:
    mask = create_vehicle_mask(with_cube)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    contour = max(contours, key=cv2.contourArea)
    canvas = np.zeros_like(mask)
    cv2.drawContours(canvas, [contour], -1, 255, cv2.FILLED)
    x, y, width, height = cv2.boundingRect(contour)
    if width <= 0 or height <= 0:
        return False

    image = cv2.imread(str(without_cube))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {without_cube}")
    no_background = cv2.bitwise_and(image, image, mask=canvas)
    green_background = np.full_like(image, (0, 255, 0))
    green_background[canvas == 255] = no_background[canvas == 255]
    crop = np.s_[y : y + height, x : x + width]

    cv2.imwrite(str(output_dir / "no_bg_cropped" / name), no_background[crop], [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(output_dir / "green_bg_cropped" / name), green_background[crop], [cv2.IMWRITE_JPEG_QUALITY, 100])
    cv2.imwrite(str(output_dir / "masks" / name), canvas[crop], [cv2.IMWRITE_JPEG_QUALITY, 100])
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", help="Only process renders for this model directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_dir = args.render_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not render_dir.is_dir():
        raise FileNotFoundError(f"Render directory not found: {render_dir}")
    if args.overwrite and output_dir.is_dir():
        shutil.rmtree(output_dir)
    for subdirectory in ("no_bg_cropped", "green_bg_cropped", "masks"):
        (output_dir / subdirectory).mkdir(parents=True, exist_ok=True)

    model_dirs = [render_dir / args.model] if args.model else sorted(render_dir.iterdir())
    level_dirs = sorted(
        path
        for model_dir in model_dirs
        if model_dir.is_dir()
        for path in model_dir.iterdir()
        if path.is_dir() and (path / "w_cube").is_dir()
    )
    if not level_dirs:
        raise RuntimeError(f"No rendered vehicle/depth directories found in {render_dir}")

    written = 0
    skipped = 0
    for level_dir in tqdm(level_dirs, desc="Cropping renders"):
        car_name = level_dir.parent.name
        flood_level = level_dir.name
        for with_cube in sorted((level_dir / "w_cube").glob("*.png")):
            without_cube = level_dir / "wo_cube" / with_cube.name
            if not without_cube.is_file():
                raise FileNotFoundError(f"Missing render pair: {without_cube}")
            output_name = f"{car_name}_level{flood_level}_cam{with_cube.stem}.jpg"
            if process_pair(with_cube, without_cube, output_dir, output_name):
                written += 1
            else:
                skipped += 1
    print(f"Saved {written} cropped retrieval images to {output_dir} (skipped {skipped})")


if __name__ == "__main__":
    main()
