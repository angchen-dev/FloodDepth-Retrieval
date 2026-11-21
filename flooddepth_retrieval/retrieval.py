#!/usr/bin/env python3
"""Estimate flood depth by VGG19 retrieval and candidate selection."""

import argparse
import json
import os
import re
import shutil
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RETRIEVAL_DATASET_DIR = (
    PROJECT_ROOT / "data" / "retrieval_dataset" / "images"
).resolve()
DEFAULT_VGG_FEATURE_H5_DIR = (
    PROJECT_ROOT / "data" / "retrieval_dataset" / "features"
).resolve()
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19

from .retrieval_common import (
    load_seg_results,
    materialize_segmentation_visuals,
    save_candidate_bbox_visualization,
)

def normalize_rows(features):
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norms, 1e-12)


def cosine_similarity_matrix(database_features, query_features):
    """Return pairwise cosine similarities with shape database x query."""
    normalized_database = normalize_rows(database_features)
    normalized_queries = normalize_rows(query_features)
    return normalized_database @ normalized_queries.T


def load_vgg_cache(vgg_feature_h5_dir, retrieval_dataset_dir):
    feature_path = Path(vgg_feature_h5_dir)
    cache_paths = (
        [feature_path]
        if feature_path.is_file()
        else sorted(feature_path.glob("*.h5"))
    )
    features, ratios, names = [], [], []
    for cache_path in cache_paths:
        with h5py.File(cache_path, "r") as handle:
            required = {"features", "ratios", "image_names"}
            if not required.issubset(handle.keys()):
                continue
            features.append(handle["features"][:])
            ratios.append(handle["ratios"][:])
            names.extend(
                os.fsdecode(name) for name in handle["image_names"][:]
            )
    if not features:
        raise RuntimeError(
            "No VGG HDF5 files containing features, ratios, and image_names "
            f"were found at {feature_path}"
        )
    database_paths = [
        name
        if os.path.isabs(name)
        else os.path.join(retrieval_dataset_dir, name)
        for name in names
    ]
    return (
        normalize_rows(np.concatenate(features, axis=0)),
        np.concatenate(ratios, axis=0),
        database_paths,
    )


def extract_vgg_queries(image_paths, device, batch_size):
    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:28]
    model = model.eval().to(device)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    features = []
    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        batch = torch.stack(
            [transform(Image.open(path).convert("RGB")) for path in batch_paths]
        ).to(device)
        with torch.inference_mode():
            output = model(batch).flatten(1).cpu().numpy()
        features.append(output)
    del model
    return normalize_rows(np.vstack(features))


def concat_pair(query_path, retrieval_path, output_path):
    query = Image.open(query_path).convert("RGB")
    match = Image.open(retrieval_path).convert("RGB")
    height = max(query.height, match.height)

    def resize(image):
        width = max(1, round(image.width * height / image.height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    query, match = resize(query), resize(match)
    merged = Image.new("RGB", (query.width + match.width + 20, height))
    merged.paste(query, (0, 0))
    merged.paste(match, (query.width + 20, 0))
    merged.save(output_path)


def retrieve_and_select_candidates(
    image_records,
    candidates,
    query_features,
    database_features,
    database_paths,
    result_dir,
    method,
    flood_height_cm,
    database_ratios,
    mask_score_weight,
):
    matched_dir = Path(result_dir) / "matched_info"
    pair_dir = Path(result_dir) / "save_pairs"
    final_image_dir = Path(result_dir) / "images"
    final_query_dir = Path(result_dir) / "vgg_queries"
    final_mask_dir = Path(result_dir) / "masks"
    bbox_vis_dir = Path(result_dir) / "candidate_bboxes"
    matched_dir.mkdir(parents=True, exist_ok=True)
    pair_dir.mkdir(parents=True, exist_ok=True)
    final_image_dir.mkdir(parents=True, exist_ok=True)
    final_query_dir.mkdir(parents=True, exist_ok=True)
    final_mask_dir.mkdir(parents=True, exist_ok=True)
    bbox_vis_dir.mkdir(parents=True, exist_ok=True)
    predictions = []
    similarities = cosine_similarity_matrix(
        database_features,
        query_features,
    )

    for query_index, candidate in enumerate(candidates):
        ranked = np.argsort(-similarities[:, query_index])
        query_ratio = candidate.get("crop_aspect_ratio")
        if query_ratio is None:
            with Image.open(candidate["candidate_path"]) as query_image:
                query_ratio = query_image.width / query_image.height
            candidate["crop_aspect_ratio"] = query_ratio
        if query_ratio > 1.0:
            valid = [
                int(index)
                for index in ranked
                if database_ratios[index] >= query_ratio
            ]
        else:
            valid = [
                int(index)
                for index in ranked
                if database_ratios[index] <= query_ratio
            ]
        database_index = (
            valid[0]
            if valid
            else int(np.argmin(np.abs(database_ratios - query_ratio)))
        )
        database_path = database_paths[database_index]
        if not os.path.isfile(database_path):
            raise FileNotFoundError(database_path)
        similarity = float(similarities[database_index, query_index])
        match = re.search(
            r"level([0-9]*\.?[0-9]+)_cam", os.path.basename(database_path)
        )
        ratio = float(match.group(1)) if match else None
        depth_cm = round(ratio * flood_height_cm, 2) if ratio is not None else None
        candidate.update(
            {
                "best_match_path": database_path,
                "best_match_image": os.path.basename(database_path),
                "similarity": similarity,
                "flood_depth_ratio": ratio,
                "flood_depth_cm": depth_cm,
            }
        )

    for record in image_records:
        record["mask_score_weight"] = mask_score_weight
        record_candidates = record["candidates"]
        max_area_ratio = max(
            candidate["area_ratio"] for candidate in record_candidates
        )
        for candidate in record_candidates:
            normalized_area = (
                candidate["area_ratio"] / max_area_ratio
                if max_area_ratio > 0
                else 0.0
            )
            candidate["normalized_area_ratio"] = normalized_area
            candidate["final_score"] = (
                mask_score_weight * normalized_area
                + (1.0 - mask_score_weight) * candidate["similarity"]
            )

        final_candidate = max(
            record_candidates,
            key=lambda candidate: candidate["final_score"],
        )
        final_index = final_candidate["candidate_index"]
        image_name = record["image_name"]
        stem = Path(image_name).stem
        candidate_pair_dir = pair_dir / stem
        candidate_pair_dir.mkdir(parents=True, exist_ok=True)

        for candidate in record_candidates:
            candidate["used_as_final_vehicle"] = (
                candidate["candidate_index"] == final_index
            )
            candidate_pair_path = (
                candidate_pair_dir
                / f"candidate_{candidate['candidate_index']:03d}.png"
            )
            concat_pair(
                candidate["candidate_path"],
                candidate["best_match_path"],
                candidate_pair_path,
            )
            candidate["pair_path"] = os.path.relpath(
                candidate_pair_path,
                result_dir,
            )

        final_image_path = final_image_dir / image_name
        with Image.open(final_candidate["candidate_path"]) as image:
            image.convert("RGB").save(final_image_path)
        final_query_path = final_query_dir / f"{stem}.png"
        shutil.copyfile(
            final_candidate["candidate_path"],
            final_query_path,
        )
        final_mask_path = final_mask_dir / f"{stem}.png"
        shutil.copyfile(
            final_candidate["candidate_mask_path"],
            final_mask_path,
        )
        concat_pair(
            final_candidate["candidate_path"],
            final_candidate["best_match_path"],
            pair_dir / image_name,
        )
        save_candidate_bbox_visualization(
            record,
            str(bbox_vis_dir / image_name),
        )

        serialized_candidates = []
        for candidate in record_candidates:
            serialized_candidates.append(
                {
                    "candidate_index": candidate["candidate_index"],
                    "sam_mask_index": candidate["sam_mask_index"],
                    "sam_score": round(candidate["sam_score"], 6),
                    "bbox_xyxy": candidate["bbox_xyxy"],
                    "mask_area_pixels": candidate["mask_area_pixels"],
                    "area_ratio": round(candidate["area_ratio"], 6),
                    "normalized_area_ratio": round(
                        candidate["normalized_area_ratio"],
                        6,
                    ),
                    "crop_aspect_ratio": round(
                        candidate["crop_aspect_ratio"],
                        6,
                    ),
                    "similarity": round(candidate["similarity"], 6),
                    "final_score": round(candidate["final_score"], 6),
                    "best_match_image": candidate["best_match_image"],
                    "flood_depth_ratio": candidate["flood_depth_ratio"],
                    "flood_depth_cm": candidate["flood_depth_cm"],
                    "used_as_final_vehicle": candidate[
                        "used_as_final_vehicle"
                    ],
                    "pair_path": candidate["pair_path"],
                }
            )

        result = {
            "image_name": image_name,
            "ablation_method": method,
            "similarity_metric": "cosine",
            "image_width": record["image_width"],
            "image_height": record["image_height"],
            "sam_masks_detected": record["sam_masks_detected"],
            "number_of_candidates": len(record_candidates),
            "mask_score_weight": mask_score_weight,
            "final_vehicle_index": final_index,
            "candidate_visualization_path": os.path.relpath(
                record["candidate_visualization_path"],
                result_dir,
            ),
            "bbox_xyxy": final_candidate["bbox_xyxy"],
            "mask_area_pixels": final_candidate["mask_area_pixels"],
            "area_ratio": round(final_candidate["area_ratio"], 6),
            "crop_aspect_ratio": round(
                final_candidate["crop_aspect_ratio"],
                6,
            ),
            "best_match_image": final_candidate["best_match_image"],
            "similarity": round(final_candidate["similarity"], 6),
            "flood_depth_ratio": final_candidate["flood_depth_ratio"],
            "flood_depth_cm": final_candidate["flood_depth_cm"],
            "candidates": serialized_candidates,
            "final_estimate": {
                "candidate_index": final_index,
                "mask_path": os.path.relpath(final_mask_path, result_dir),
                "best_match_image": final_candidate["best_match_image"],
                "flood_depth_ratio": final_candidate["flood_depth_ratio"],
                "flood_depth_cm": final_candidate["flood_depth_cm"],
            },
        }
        with open(
            matched_dir / f"{stem}.json",
            "w",
            encoding="utf-8",
        ) as handle:
            json.dump(result, handle, indent=2)
        predictions.append(
            {
                "image_name": image_name,
                "status": "success",
                "flood_depth_cm": final_candidate["flood_depth_cm"],
                "flood_depth_ratio": final_candidate["flood_depth_ratio"],
                "selected_candidate_index": final_index,
                "best_match_image": final_candidate["best_match_image"],
                "similarity": round(final_candidate["similarity"], 6),
                "final_score": round(final_candidate["final_score"], 6),
            }
        )
        print(
            f"  {image_name}: candidate {final_index} selected "
            f"(score={final_candidate['final_score']:.4f}, "
            f"match={final_candidate['best_match_image']})"
        )
    return predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seg_results_dir",
        dest="seg_results_dir",
        required=True,
        help=(
            "Result root produced by a compatible segmentation exporter; "
            "must contain metadata/, "
            "crops/, masks/, resized_images/, and visualizations/"
        ),
    )
    parser.add_argument(
        "--retrieval_dataset_dir",
        "--database_dir",
        dest="retrieval_dataset_dir",
        default=str(DEFAULT_RETRIEVAL_DATASET_DIR),
        help="Directory containing the retrieval reference images",
    )
    parser.add_argument(
        "--vgg_feature_h5_dir",
        "--vgg_feature_h5_path",
        "--vgg_feature_dir",
        dest="vgg_feature_h5_dir",
        default=str(DEFAULT_VGG_FEATURE_H5_DIR),
        help="Directory containing VGG feature HDF5 shards",
    )
    parser.add_argument(
        "--out_dir",
        "--output_root",
        dest="out_dir",
        default="./matching_vgg19_ablation",
        help="Directory in which matching results are saved",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mask_score_weight",
        type=float,
        default=0.2,
        help=(
            "Weight of normalized mask area in final candidate selection; "
            "VGG19 cosine similarity receives the remaining weight"
        ),
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--flood_height_cm", type=float, default=145.7)
    args = parser.parse_args()

    for path in (
        args.seg_results_dir,
        args.retrieval_dataset_dir,
        args.vgg_feature_h5_dir,
    ):
        if not os.path.exists(path):
            parser.error(f"Path does not exist: {path}")
    if not 0.0 <= args.mask_score_weight <= 1.0:
        parser.error("--mask_score_weight must be between 0 and 1")
    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")

    result_dir = Path(args.out_dir).resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    processing_start = time.perf_counter()

    print("[1/3] Loading all candidates from segmentation results...")
    stage_start = time.perf_counter()
    try:
        image_records = load_seg_results(args.seg_results_dir)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to load segmentation results: {exc}") from exc
    materialize_segmentation_visuals(
        image_records,
        result_dir / "candidates",
        result_dir / "resized_originals",
    )
    candidates = [
        candidate
        for record in image_records
        for candidate in record["candidates"]
    ]
    print(
        f"Loaded {len(candidates)} vehicle candidates from "
        f"{len(image_records)} images"
    )
    load_candidates_seconds = time.perf_counter() - stage_start

    print("[2/3] Loading VGG19 database features...")
    stage_start = time.perf_counter()
    database_features, database_ratios, database_paths = load_vgg_cache(
        vgg_feature_h5_dir=args.vgg_feature_h5_dir,
        retrieval_dataset_dir=args.retrieval_dataset_dir,
    )
    query_features = extract_vgg_queries(
        image_paths=[
            candidate["candidate_path"] for candidate in candidates
        ],
        device=args.device,
        batch_size=args.batch_size,
    )
    if query_features.shape[0] != len(candidates):
        raise ValueError(
            f"VGG query count mismatch: features={query_features.shape[0]}, "
            f"candidates={len(candidates)}"
        )
    if query_features.shape[1] != database_features.shape[1]:
        raise ValueError(
            f"VGG feature dimension mismatch: query={query_features.shape[1]}, "
            f"database={database_features.shape[1]}"
        )
    feature_extraction_seconds = time.perf_counter() - stage_start

    print("[3/3] Matching candidates and selecting one vehicle per image...")
    stage_start = time.perf_counter()
    predictions = retrieve_and_select_candidates(
        image_records=image_records,
        candidates=candidates,
        query_features=query_features,
        database_features=database_features,
        database_paths=database_paths,
        result_dir=result_dir,
        method="SAM3+VGG19",
        flood_height_cm=args.flood_height_cm,
        database_ratios=database_ratios,
        mask_score_weight=args.mask_score_weight,
    )
    with (result_dir / "predictions.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            {
                "method": "SAM3+VGG19",
                "reference_vehicle_height_cm": args.flood_height_cm,
                "area_weight": args.mask_score_weight,
                "number_of_predictions": len(predictions),
                "predictions": predictions,
            },
            handle,
            indent=2,
        )
    matching_seconds = time.perf_counter() - stage_start
    total_processing_seconds = time.perf_counter() - processing_start
    average_seconds_per_image = (
        total_processing_seconds / len(image_records) if image_records else 0.0
    )
    timing_result = {
        "number_of_images": len(image_records),
        "number_of_vehicle_candidates": len(candidates),
        "total_processing_time_seconds": round(total_processing_seconds, 6),
        "average_processing_time_per_image_seconds": round(
            average_seconds_per_image,
            6,
        ),
        "processing_scope": (
            "candidate loading/materialization, VGG19 feature extraction, "
            "retrieval matching, representative-vehicle selection, and "
            "result saving"
        ),
        "stage_times_seconds": {
            "load_and_materialize_candidates": round(
                load_candidates_seconds,
                6,
            ),
            "load_database_and_extract_query_features": round(
                feature_extraction_seconds,
                6,
            ),
            "match_select_and_save_results": round(matching_seconds, 6),
        },
    }
    timing_json_path = result_dir / "processing_time.json"
    with timing_json_path.open("w", encoding="utf-8") as handle:
        json.dump(timing_result, handle, indent=2)
    with (result_dir / "processing_time.txt").open(
        "w",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "Average processing time per image: "
            f"{average_seconds_per_image:.6f} sec\n"
        )
        handle.write(
            f"Total processing time: {total_processing_seconds:.6f} sec\n"
        )
        handle.write(f"Number of images: {len(image_records)}\n")
    print(
        "Average processing time per image: "
        f"{average_seconds_per_image:.6f} sec"
    )

    print(f"Done: {result_dir}")


if __name__ == "__main__":
    main()
