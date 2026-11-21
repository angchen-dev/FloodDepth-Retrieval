#!/usr/bin/env python3
"""Run retrieval with the paper source code's matching-time definition."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flooddepth_retrieval import retrieval  # noqa: E402


def get_output_dir(argv: list[str]) -> Path:
    for option in ("--out_dir", "--output_root"):
        if option in argv:
            index = argv.index(option)
            if index + 1 >= len(argv):
                raise SystemExit(f"{option} requires a value")
            return Path(argv[index + 1]).resolve()
        prefix = f"{option}="
        for argument in argv:
            if argument.startswith(prefix):
                return Path(argument[len(prefix) :]).resolve()
    return (PROJECT_ROOT / "matching_vgg19_ablation").resolve()


def main() -> None:
    output_dir = get_output_dir(sys.argv[1:])
    original = retrieval.cosine_similarity_matrix
    measurements: list[dict[str, float | int]] = []

    def timed_cosine_similarity(database_features, query_features):
        # This reproduces matching_vgg19.py: time only normalization and the
        # candidate-query/reference-feature cosine-similarity matrix. Feature
        # extraction, ranking, selection, and result saving are excluded.
        matching_start = time.perf_counter()
        result = original(database_features, query_features)
        matching_seconds = time.perf_counter() - matching_start
        measurements.append(
            {
                "number_of_reference_features": int(
                    database_features.shape[0]
                ),
                "number_of_vehicle_candidates": int(query_features.shape[0]),
                "total_matching_time_seconds": matching_seconds,
            }
        )
        return result

    retrieval.cosine_similarity_matrix = timed_cosine_similarity
    retrieval.main()

    if len(measurements) != 1:
        raise RuntimeError(
            "Expected one cosine-similarity operation, found "
            f"{len(measurements)}"
        )
    measurement = measurements[0]
    candidate_count = int(measurement["number_of_vehicle_candidates"])
    matching_seconds = float(measurement["total_matching_time_seconds"])
    average_seconds = (
        matching_seconds / candidate_count if candidate_count else 0.0
    )
    timing = {
        "number_of_reference_features": measurement[
            "number_of_reference_features"
        ],
        "number_of_vehicle_candidates": candidate_count,
        "total_matching_time_seconds": round(matching_seconds, 6),
        "average_matching_time_per_candidate_query_seconds": round(
            average_seconds,
            6,
        ),
        "matching_scope": (
            "cosine-similarity matrix computation between candidate-query "
            "features and reference features only; excludes feature "
            "extraction, ranking, aspect-ratio filtering, candidate selection, "
            "visualization, and result saving"
        ),
    }

    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "processing_time.json").write_text(
        json.dumps(timing, indent=2) + "\n",
        encoding="utf-8",
    )
    (analysis_dir / "processing_time.txt").write_text(
        "Average matching time per candidate query: "
        f"{average_seconds:.6f} sec\n"
        f"Total matching time: {matching_seconds:.6f} sec\n"
        f"Number of vehicle candidates: {candidate_count}\n",
        encoding="utf-8",
    )
    print(
        "Average matching time per candidate query: "
        f"{average_seconds:.6f} sec"
    )


if __name__ == "__main__":
    main()
