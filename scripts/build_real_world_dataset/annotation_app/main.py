#!/usr/bin/env python3
"""Run the participant web app used to annotate the real-world dataset."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parents[2]
DEFAULT_IMAGES_DIR = REPO_ROOT / "data/build_real_world_dataset/images"
DEFAULT_ASSETS_DIR = REPO_ROOT / "data/build_real_world_dataset/assets"
DEFAULT_DB_PATH = APP_DIR / "results/results.db"
GROUPS_PATH = APP_DIR / "annotation_groups.json"
TRIAL_IMAGE_NAMES = ("trial_049.jpg", "trial_050.jpg", "trial_201.jpg")
ANNOTATIONS_PER_IMAGE = 3
VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def configured_path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, default)).expanduser().resolve()


IMAGES_DIR = configured_path("FLOOD_APP_IMAGES_DIR", DEFAULT_IMAGES_DIR)
ASSETS_DIR = configured_path("FLOOD_APP_ASSETS_DIR", DEFAULT_ASSETS_DIR)
DB_PATH = configured_path("FLOOD_APP_DB_PATH", DEFAULT_DB_PATH)


def load_groups() -> list[list[str]]:
    with GROUPS_PATH.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    groups = [payload[key] for key in sorted(payload, key=lambda value: int(value))]
    missing = [name for group in groups for name in group if not (IMAGES_DIR / name).is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"Annotation images are missing from {IMAGES_DIR}: {preview}")
    return groups


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT NOT NULL,
                image_path TEXT NOT NULL,
                car_model TEXT,
                flood_depth REAL,
                timestamp TEXT NOT NULL,
                phrase TEXT NOT NULL,
                consent_date TEXT
            )
            """
        )


def save_result(payload: dict[str, Any]) -> None:
    try:
        result = payload["result"]
        row = (
            str(payload["user"]),
            str(result["image"]),
            str(result["car_model"]),
            float(result["flood_depth"]),
            str(result.get("date_time") or datetime.now(timezone.utc).isoformat()),
            str(result["phrase"]),
            str(payload.get("consentDate", "")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=f"Invalid result payload: {error}") from error

    with sqlite3.connect(DB_PATH) as connection:
        connection.execute(
            """
            INSERT INTO results (
                user, image_path, car_model, flood_depth,
                timestamp, phrase, consent_date
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )


def asset_files(subdirectory: str, extensions: set[str]) -> list[str]:
    directory = ASSETS_DIR / subdirectory
    return sorted(
        f"/assets/{subdirectory}/{path.name}"
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def image_url(name: str) -> str:
    return f"/evaluation-images/{name}"


for required in (
    IMAGES_DIR,
    ASSETS_DIR / "car_images",
    ASSETS_DIR / "car_models",
    ASSETS_DIR / "instruction",
    ASSETS_DIR / "textures",
):
    if not required.is_dir():
        raise RuntimeError(
            f"Required directory does not exist: {required}. "
            "See annotation_app/README.md for the expected data layout."
        )

IMAGE_GROUPS = load_groups()
init_db()

app = FastAPI(title="Flood-depth annotation app")
app.mount("/static", StaticFiles(directory=APP_DIR / "static"), name="static")
app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")
app.mount(
    "/evaluation-images",
    StaticFiles(directory=IMAGES_DIR),
    name="evaluation-images",
)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(APP_DIR / "index.html")


@app.get("/health/")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/trial_images/")
def get_trial_images() -> dict[str, Any]:
    names = [name for name in TRIAL_IMAGE_NAMES if (IMAGES_DIR / name).is_file()]
    data = [image_url(name) for name in names]
    return {"data": data, "count": len(data)}


@app.get("/anno_images/")
def get_annotation_images(user: str | None = Query(default=None)) -> dict[str, Any]:
    if not user or not user.strip():
        return {"data": [], "count": 0, "message": "Participant ID is required."}

    with sqlite3.connect(DB_PATH) as connection:
        user_images = {
            row[0]
            for row in connection.execute(
                "SELECT image_path FROM results WHERE user = ? AND phrase = 'real'",
                (user,),
            )
        }
        counts = dict(
            connection.execute(
                """
                SELECT image_path, COUNT(*) FROM results
                WHERE phrase = 'real' GROUP BY image_path
                """
            )
        )

    assigned_index = next(
        (
            index
            for index, group in enumerate(IMAGE_GROUPS)
            if any(image_url(name) in user_images for name in group)
        ),
        None,
    )
    if assigned_index is not None:
        remaining = [
            image_url(name)
            for name in IMAGE_GROUPS[assigned_index]
            if image_url(name) not in user_images
        ]
        message = None if remaining else "This participant has completed the assigned group."
        return {
            "data": remaining,
            "count": len(remaining),
            "group": assigned_index + 1,
            **({"message": message} if message else {}),
        }

    for index, group in enumerate(IMAGE_GROUPS):
        if all(counts.get(image_url(name), 0) < ANNOTATIONS_PER_IMAGE for name in group):
            data = [image_url(name) for name in group]
            return {"data": data, "count": len(data), "group": index + 1}

    return {"data": [], "count": 0, "message": "No annotation group is available."}


@app.get("/car_images/")
def get_car_images() -> dict[str, Any]:
    data = [
        path
        for path in asset_files("car_images", VALID_IMAGE_EXTENSIONS)
        if not path.endswith("/all_cars.png")
    ]
    return {"data": data, "count": len(data)}


@app.get("/car_models/")
def get_car_models() -> dict[str, Any]:
    data = asset_files("car_models", {".glb"})
    return {"data": data, "count": len(data)}


@app.get("/instruction/")
def get_instruction() -> list[str]:
    instruction_path = APP_DIR / "static/instruction.html"
    soup = BeautifulSoup(instruction_path.read_text(encoding="utf-8"), "html.parser")
    content = soup.find("div", {"id": "inst-msg"})
    return [str(content)] if content else ["Instruction message not found"]


@app.post("/submit_result/")
def submit_result(payload: dict[str, Any]) -> dict[str, str]:
    save_result(payload)
    return {"status": "success", "message": "Result saved"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    uvicorn.run(app, host=arguments.host, port=arguments.port)
