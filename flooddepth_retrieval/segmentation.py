#!/usr/bin/env python3
"""Segment vehicles in an image directory with SAM3."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Use the complete SAM3 source bundled with this repository.
_SAM3_REPO = Path(__file__).resolve().parent / "sam3"
if (_SAM3_REPO / "sam3").is_dir():
    sys.path.insert(0, str(_SAM3_REPO))

import cv2
import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MASK_COLORS_BGR = (
    (0, 0, 255),
    (0, 180, 0),
    (255, 80, 0),
    (0, 180, 255),
    (180, 0, 180),
    (255, 180, 0),
)


def natural_sort_key(path):
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", Path(path).name)
    ]


def list_images(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"Input directory does not exist: {directory}")
    images = sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=natural_sort_key,
    )
    stems = [path.stem for path in images]
    duplicate_stems = sorted(
        stem for stem in set(stems) if stems.count(stem) > 1
    )
    if duplicate_stems:
        raise ValueError(
            "Input images must have unique filename stems; duplicates: "
            + ", ".join(duplicate_stems)
        )
    return images


def resize_like_estimate_flood_depth(image):
    """Reproduce the resize policy used by utils/get_mask.py."""
    width, height = image.size
    if width > 960:
        new_width = 960
        new_height = int((new_width / width) * height)
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )
    if height > 540:
        new_height = 540
        new_width = int((new_height / height) * width)
        image = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS,
        )
    return image


def tensor_to_numpy(value, dtype):
    if value is None:
        return None
    return value.detach().cpu().numpy().astype(dtype, copy=False)


def masks_to_numpy(masks, height, width):
    if masks is None or masks.numel() == 0:
        return np.empty((0, height, width), dtype=np.uint8)
    masks_np = tensor_to_numpy(masks, np.uint8)
    if masks_np.ndim == 4 and masks_np.shape[1] == 1:
        masks_np = masks_np[:, 0]
    if masks_np.ndim != 3:
        raise ValueError(f"Unexpected SAM3 mask shape: {masks_np.shape}")
    return masks_np


def binary_mask_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]


def save_visualization(image_rgb, masks, boxes, save_path):
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    visualization = image_bgr.astype(np.float32)

    for index, mask in enumerate(masks):
        color = np.asarray(
            MASK_COLORS_BGR[index % len(MASK_COLORS_BGR)],
            dtype=np.float32,
        )
        foreground = mask > 0
        visualization[foreground] = (
            0.55 * visualization[foreground] + 0.45 * color
        )

    visualization = np.clip(visualization, 0, 255).astype(np.uint8)
    for index, mask in enumerate(masks):
        color = MASK_COLORS_BGR[index % len(MASK_COLORS_BGR)]
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(visualization, contours, -1, color, 2)

        if index < len(boxes):
            x1, y1, x2, y2 = np.rint(boxes[index]).astype(int)
        else:
            bbox = binary_mask_bbox(mask)
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
        cv2.rectangle(visualization, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            visualization,
            str(index),
            (x1 + 4, max(24, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            visualization,
            str(index),
            (x1 + 4, max(24, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
            cv2.LINE_AA,
        )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(save_path), visualization):
        raise RuntimeError(f"Failed to save visualization: {save_path}")


def run_one(processor, image_path, output_dirs, prompt, save_logits):
    with Image.open(image_path) as source:
        original_size = source.size
        resized_image = resize_like_estimate_flood_depth(
            source.convert("RGB")
        )
    image_rgb = np.asarray(resized_image)
    height, width = image_rgb.shape[:2]

    state = processor.set_image(resized_image)
    state = processor.set_text_prompt(state=state, prompt=prompt)
    masks = masks_to_numpy(state.get("masks"), height, width)
    boxes = tensor_to_numpy(state.get("boxes"), np.float32)
    scores = tensor_to_numpy(state.get("scores"), np.float32)
    boxes = (
        boxes.reshape(-1, 4)
        if boxes is not None
        else np.empty((0, 4), dtype=np.float32)
    )
    scores = (
        scores.reshape(-1)
        if scores is not None
        else np.zeros((len(masks),), dtype=np.float32)
    )

    stem = image_path.stem
    resized_path = output_dirs["resized_images"] / image_path.name
    resized_image.save(resized_path)

    image_mask_dir = output_dirs["masks"] / stem
    image_mask_dir.mkdir(parents=True, exist_ok=True)
    for stale_mask_path in image_mask_dir.glob("mask_*.png"):
        stale_mask_path.unlink()
    image_crop_dir = output_dirs["crops"] / stem
    image_crop_dir.mkdir(parents=True, exist_ok=True)
    for stale_crop_path in image_crop_dir.glob("vehicle_*.png"):
        stale_crop_path.unlink()

    instances = []
    for index, mask in enumerate(masks):
        mask_path = image_mask_dir / f"mask_{index:03d}.png"
        if not cv2.imwrite(str(mask_path), mask * 255):
            raise RuntimeError(f"Failed to save mask: {mask_path}")
        bbox = binary_mask_bbox(mask)
        crop_path = None
        crop_aspect_ratio = None
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            masked_image = np.zeros_like(image_rgb)
            masked_image[mask > 0] = image_rgb[mask > 0]
            # Keep the same exclusive maximum-coordinate crop convention as
            # estimate_flood_depth.py and the current retrieval scripts.
            crop = masked_image[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = image_crop_dir / f"vehicle_{index:03d}.png"
                Image.fromarray(crop).save(crop_path)
                crop_aspect_ratio = float(crop.shape[1] / crop.shape[0])
        model_box = (
            [round(float(value), 4) for value in boxes[index]]
            if index < len(boxes)
            else None
        )
        instances.append(
            {
                "instance_index": index,
                "score": (
                    round(float(scores[index]), 6)
                    if index < len(scores)
                    else None
                ),
                "model_bbox_xyxy": model_box,
                "mask_bbox_xyxy": bbox,
                "mask_area_pixels": int(mask.sum()),
                "area_ratio": round(
                    float(mask.sum()) / float(height * width),
                    6,
                ),
                "mask_path": os.path.relpath(mask_path, output_dirs["root"]),
                "crop_path": (
                    os.path.relpath(crop_path, output_dirs["root"])
                    if crop_path is not None
                    else None
                ),
                "crop_aspect_ratio": (
                    round(crop_aspect_ratio, 6)
                    if crop_aspect_ratio is not None
                    else None
                ),
            }
        )

    raw_path = output_dirs["raw"] / f"{stem}.npz"
    np.savez_compressed(
        raw_path,
        masks=masks,
        boxes=boxes,
        scores=scores,
    )
    if save_logits:
        logits = tensor_to_numpy(state.get("masks_logits"), np.float16)
        if logits is not None and logits.ndim == 4 and logits.shape[1] == 1:
            logits = logits[:, 0]
        np.savez_compressed(
            output_dirs["raw_logits"] / f"{stem}.npz",
            masks_logits=logits,
        )

    visualization_path = output_dirs["visualizations"] / image_path.name
    save_visualization(image_rgb, masks, boxes, visualization_path)

    metadata = {
        "image_name": image_path.name,
        "prompt": prompt,
        "original_width": original_size[0],
        "original_height": original_size[1],
        "resized_width": width,
        "resized_height": height,
        "number_of_instances": len(masks),
        "raw_npz_path": os.path.relpath(raw_path, output_dirs["root"]),
        "resized_image_path": os.path.relpath(
            resized_path,
            output_dirs["root"],
        ),
        "visualization_path": os.path.relpath(
            visualization_path,
            output_dirs["root"],
        ),
        "instances": instances,
    }
    metadata_path = output_dirs["metadata"] / f"{stem}.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return len(masks)


def make_output_dirs(out_dir, save_logits):
    root = Path(out_dir).resolve()
    output_dirs = {
        "root": root,
        "resized_images": root / "resized_images",
        "masks": root / "masks",
        "crops": root / "crops",
        "raw": root / "raw",
        "metadata": root / "metadata",
        "visualizations": root / "visualizations",
    }
    if save_logits:
        output_dirs["raw_logits"] = root / "raw_logits"
    for path in output_dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return output_dirs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in_dir",
        "--original_dir",
        dest="in_dir",
        required=True,
        help="Directory containing the original images",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Parent directory for raw SAM3 results",
    )
    parser.add_argument(
        "--prompt",
        default="vehicle",
        help="SAM3 text prompt; default: vehicle",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--sam_confidence_threshold",
        type=float,
        default=0.5,
        help="SAM3 confidence threshold; default: 0.5",
    )
    parser.add_argument(
        "--sam_resolution",
        type=int,
        default=1008,
        help="Square SAM3 inference resolution; default: 1008",
    )
    parser.add_argument(
        "--save_logits",
        action="store_true",
        help="Also save the raw float16 mask logits (requires more storage)",
    )
    args = parser.parse_args()

    if not 0.0 <= args.sam_confidence_threshold <= 1.0:
        parser.error("--sam_confidence_threshold must be between 0 and 1")
    if args.sam_resolution <= 0:
        parser.error("--sam_resolution must be positive")

    try:
        image_paths = list_images(args.in_dir)
    except ValueError as exc:
        parser.error(str(exc))
    if not image_paths:
        parser.error(f"No supported images found in {args.in_dir}")
    output_dirs = make_output_dirs(args.out_dir, args.save_logits)

    model = build_sam3_image_model().to(args.device)
    processor = Sam3Processor(
        model,
        resolution=args.sam_resolution,
        device=args.device,
        confidence_threshold=args.sam_confidence_threshold,
    )

    total_instances = 0
    successful_images = 0
    for index, image_path in enumerate(image_paths, 1):
        try:
            count = run_one(
                processor,
                image_path,
                output_dirs,
                args.prompt,
                args.save_logits,
            )
            total_instances += count
            successful_images += 1
            print(
                f"[{index}/{len(image_paths)}] {image_path.name}: "
                f"{count} instance(s)"
            )
        except Exception as exc:
            print(
                f"[{index}/{len(image_paths)}] WARN {image_path.name}: {exc}"
            )

    summary = {
        "input_directory": str(Path(args.in_dir).resolve()),
        "output_directory": str(output_dirs["root"]),
        "prompt": args.prompt,
        "sam_confidence_threshold": args.sam_confidence_threshold,
        "sam_resolution": args.sam_resolution,
        "images_found": len(image_paths),
        "images_processed": successful_images,
        "images_failed": len(image_paths) - successful_images,
        "instances_saved": total_instances,
        "raw_mask_processing": (
            "none: no morphology, component filtering, deduplication, "
            "merging, or final-instance selection"
        ),
    }
    with (output_dirs["root"] / "summary.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(summary, handle, indent=2)

    if successful_images == 0:
        raise SystemExit("SAM3 failed for every input image")
    print(
        f"Done: saved {total_instances} raw instance(s) from "
        f"{successful_images}/{len(image_paths)} image(s) to "
        f"{output_dirs['root']}"
    )


if __name__ == "__main__":
    main()
