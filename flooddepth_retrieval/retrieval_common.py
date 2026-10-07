"""Shared I/O and visualization helpers for the retrieval pipeline."""

import json
import os
import re
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def _resolve_segmentation_path(root, relative_path, field, metadata_path):
    if not relative_path:
        raise ValueError(f"Missing {field} in {metadata_path}")
    path = Path(relative_path)
    if not path.is_absolute():
        path = root / path
    if not path.is_file():
        raise FileNotFoundError(f"{field} does not exist: {path}")
    return str(path.resolve())


def load_seg_results(seg_results_dir):
    """Load standardized instance crops produced by a segmentation exporter."""
    root = Path(seg_results_dir).resolve()
    metadata_dir = root / "metadata"
    if not metadata_dir.is_dir():
        raise ValueError(
            f"Missing segmentation metadata directory: {metadata_dir}"
        )
    metadata_paths = sorted(
        metadata_dir.glob("*.json"),
        key=lambda path: [
            int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", path.name)
        ],
    )
    if not metadata_paths:
        raise ValueError(f"No metadata JSON files found in {metadata_dir}")

    image_records = []
    for metadata_path in metadata_paths:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        image_name = metadata.get("image_name")
        if not image_name:
            print(f"WARN: missing image_name in {metadata_path}")
            continue
        width = int(metadata.get("resized_width", 0))
        height = int(metadata.get("resized_height", 0))
        if width <= 0 or height <= 0:
            print(f"WARN: invalid resized dimensions in {metadata_path}")
            continue

        candidates = []
        for instance in metadata.get("instances", []):
            try:
                candidate_path = _resolve_segmentation_path(
                    root,
                    instance.get("crop_path"),
                    "crop_path",
                    metadata_path,
                )
                candidate_mask_path = _resolve_segmentation_path(
                    root,
                    instance.get("mask_path"),
                    "mask_path",
                    metadata_path,
                )
            except (ValueError, FileNotFoundError) as exc:
                print(f"WARN: {exc}")
                continue

            candidate_index = int(
                instance.get("instance_index", len(candidates))
            )
            mask_area = int(instance.get("mask_area_pixels", 0))
            area_ratio = instance.get("area_ratio")
            if area_ratio is None:
                area_ratio = mask_area / float(width * height)
            candidates.append(
                {
                    "candidate_index": candidate_index,
                    "sam_mask_index": candidate_index,
                    "sam_score": float(instance.get("score") or 0.0),
                    "bbox_xyxy": instance.get("mask_bbox_xyxy"),
                    "mask_area_pixels": mask_area,
                    "area_ratio": float(area_ratio),
                    "crop_aspect_ratio": instance.get("crop_aspect_ratio"),
                    "candidate_path": candidate_path,
                    "candidate_mask_path": candidate_mask_path,
                }
            )
        if not candidates:
            print(f"WARN: no usable vehicle crops for {image_name}")
            continue

        try:
            resized_original_path = _resolve_segmentation_path(
                root,
                metadata.get("resized_image_path"),
                "resized_image_path",
                metadata_path,
            )
            candidate_visualization_path = _resolve_segmentation_path(
                root,
                metadata.get("visualization_path"),
                "visualization_path",
                metadata_path,
            )
        except (ValueError, FileNotFoundError) as exc:
            print(f"WARN: {exc}")
            continue

        image_records.append(
            {
                "image_name": image_name,
                "original_path": None,
                "image_path": resized_original_path,
                "resized_original_path": resized_original_path,
                "candidate_visualization_path": candidate_visualization_path,
                "image_width": width,
                "image_height": height,
                "sam_masks_detected": int(
                    metadata.get("number_of_instances", len(candidates))
                ),
                "duplicate_masks_removed": 0,
                "segmentation_metadata_path": str(metadata_path.resolve()),
                "candidates": candidates,
            }
        )

    if not image_records:
        raise ValueError(
            f"No usable segmentation image records found in {metadata_dir}"
        )
    return image_records


def _save_image_in_destination_format(source_path, destination_path):
    with Image.open(source_path) as image:
        image.convert("RGB").save(destination_path)


def materialize_segmentation_visuals(
    image_records,
    candidates_dir,
    resized_originals_dir,
):
    os.makedirs(candidates_dir, exist_ok=True)
    os.makedirs(resized_originals_dir, exist_ok=True)
    for record in image_records:
        candidate_visualization_path = os.path.join(
            candidates_dir,
            record["image_name"],
        )
        _save_image_in_destination_format(
            record["candidate_visualization_path"],
            candidate_visualization_path,
        )
        resized_original_path = os.path.join(
            resized_originals_dir,
            record["image_name"],
        )
        _save_image_in_destination_format(
            record["resized_original_path"],
            resized_original_path,
        )
        record["candidate_visualization_path"] = candidate_visualization_path
        record["image_path"] = resized_original_path
        record["resized_original_path"] = resized_original_path


def save_candidate_bbox_visualization(record, save_path):
    """Save the selected and rejected candidate boxes plus a score table."""
    original = cv2.imread(record["image_path"])
    if original is None:
        raise RuntimeError(
            "Failed to read original image for visualization: "
            f"{record['image_path']}"
        )

    canvas_width = 1920
    canvas_height = 1080
    image_region_height = 700
    image_margin = 20
    canvas = np.full(
        (canvas_height, canvas_width, 3), 255, dtype=np.uint8
    )

    original_h, original_w = original.shape[:2]
    scale = min(
        (canvas_width - 2 * image_margin) / original_w,
        (image_region_height - 2 * image_margin) / original_h,
    )
    resized_w = max(1, round(original_w * scale))
    resized_h = max(1, round(original_h * scale))
    resized = cv2.resize(
        original, (resized_w, resized_h), interpolation=cv2.INTER_AREA
    )
    offset_x = (canvas_width - resized_w) // 2
    offset_y = (image_region_height - resized_h) // 2
    canvas[
        offset_y : offset_y + resized_h,
        offset_x : offset_x + resized_w,
    ] = resized

    for candidate in record["candidates"]:
        x1, y1, x2, y2 = candidate["bbox_xyxy"]
        x1 = round(x1 * scale) + offset_x
        y1 = round(y1 * scale) + offset_y
        x2 = round(x2 * scale) + offset_x
        y2 = round(y2 * scale) + offset_y
        is_final = candidate["used_as_final_vehicle"]
        color = (0, 0, 255) if is_final else (0, 200, 0)
        thickness = 4 if is_final else 2
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label = str(candidate["candidate_index"])
        text_x = max(2, x1 + 5)
        text_y = max(32, y1 - 8)
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 0),
            5,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            color,
            2,
            cv2.LINE_AA,
        )

    table_top = 715
    canvas[table_top - 2 : table_top + 2, :] = (180, 180, 180)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(
        canvas,
        "Candidate results",
        (35, table_top + 30),
        font,
        0.72,
        (30, 30, 30),
        2,
        cv2.LINE_AA,
    )

    columns = [
        ("ID", 55),
        ("BBox [x1,y1,x2,y2]", 245),
        ("Area", 105),
        ("Norm area", 115),
        ("Similarity", 115),
        ("Score", 105),
        ("Lambda", 100),
        ("Best match image", 510),
        ("Ratio", 100),
        ("Depth cm", 115),
        ("Status", 110),
    ]
    table_width = sum(width for _, width in columns)
    table_x = (canvas_width - table_width) // 2
    header_y = table_top + 48
    table_bottom = canvas_height - 20
    candidate_count = len(record["candidates"])
    row_height = max(
        16,
        min(38, (table_bottom - header_y) // max(1, candidate_count + 1)),
    )
    font_scale = max(0.3, min(0.52, row_height / 58))

    def fit_text(text, width):
        text = str(text)
        if cv2.getTextSize(text, font, font_scale, 1)[0][0] <= width - 10:
            return text
        shortened = text
        while shortened:
            candidate_text = shortened + "..."
            if (
                cv2.getTextSize(candidate_text, font, font_scale, 1)[0][0]
                <= width - 10
            ):
                return candidate_text
            shortened = shortened[:-1]
        return ""

    cv2.rectangle(
        canvas,
        (table_x, header_y),
        (table_x + table_width, header_y + row_height),
        (225, 225, 225),
        -1,
    )
    current_x = table_x
    for heading, width in columns:
        cv2.rectangle(
            canvas,
            (current_x, header_y),
            (current_x + width, header_y + row_height),
            (170, 170, 170),
            1,
        )
        cv2.putText(
            canvas,
            fit_text(heading, width),
            (current_x + 5, header_y + row_height - 9),
            font,
            font_scale,
            (25, 25, 25),
            1,
            cv2.LINE_AA,
        )
        current_x += width

    for row_index, candidate in enumerate(record["candidates"]):
        row_y = header_y + (row_index + 1) * row_height
        is_final = candidate["used_as_final_vehicle"]
        if is_final:
            row_color = (235, 235, 255)
            text_color = (0, 0, 180)
        else:
            shade = 250 if row_index % 2 == 0 else 240
            row_color = (shade, shade, shade)
            text_color = (25, 25, 25)

        bbox_text = "[" + ",".join(
            str(value) for value in candidate["bbox_xyxy"]
        ) + "]"
        ratio = candidate["flood_depth_ratio"]
        depth_cm = candidate["flood_depth_cm"]
        values = [
            candidate["candidate_index"],
            bbox_text,
            f"{candidate['area_ratio']:.4f}",
            f"{candidate['normalized_area_ratio']:.4f}",
            f"{candidate['similarity']:.4f}",
            f"{candidate['final_score']:.4f}",
            f"{record['mask_score_weight']:.2f}",
            candidate["best_match_image"],
            f"{ratio:.2f}" if ratio is not None else "N/A",
            f"{depth_cm:.2f}" if depth_cm is not None else "N/A",
            "FINAL" if is_final else "",
        ]
        current_x = table_x
        for value, (_, width) in zip(values, columns):
            cv2.rectangle(
                canvas,
                (current_x, row_y),
                (current_x + width, row_y + row_height),
                row_color,
                -1,
            )
            cv2.rectangle(
                canvas,
                (current_x, row_y),
                (current_x + width, row_y + row_height),
                (195, 195, 195),
                1,
            )
            cv2.putText(
                canvas,
                fit_text(value, width),
                (current_x + 5, row_y + row_height - 9),
                font,
                font_scale,
                text_color,
                2 if is_final else 1,
                cv2.LINE_AA,
            )
            current_x += width

    if not cv2.imwrite(str(save_path), canvas):
        raise RuntimeError(f"Failed to save bbox visualization: {save_path}")

