#!/usr/bin/env python3
"""Extract VGG19 pool4 features for cropped retrieval images."""

from __future__ import annotations

import argparse
import pickle
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import VGG19_Weights, vgg19
from tqdm import tqdm


class RetrievalImageDataset(Dataset):
    def __init__(self, paths: list[Path]):
        self.paths = paths
        self.transform = transforms.Compose(
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        image = Image.open(path).convert("RGB")
        ratio = image.width / image.height
        return self.transform(image), ratio, path.name


def extract_part(paths, model, device, batch_size, workers):
    loader = DataLoader(
        RetrievalImageDataset(paths),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    features, ratios, names = [], [], []
    with torch.inference_mode():
        for images, batch_ratios, batch_names in loader:
            output = model(images.to(device)).flatten(1).cpu().numpy()
            features.append(output)
            ratios.append(batch_ratios.numpy())
            names.extend(batch_names)
    return np.vstack(features), np.concatenate(ratios), names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--parts", type=int, default=20)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = args.images_dir.expanduser().resolve() / "no_bg_cropped"
    output_dir = args.output_dir.expanduser().resolve()
    paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not paths:
        raise RuntimeError(f"No cropped retrieval images found in {source_dir}")
    if args.parts <= 0 or args.batch_size <= 0:
        raise ValueError("parts and batch-size must be positive")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    if args.overwrite and output_dir.is_dir():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "image_names.pkl").open("wb") as handle:
        pickle.dump([path.name for path in paths], handle)

    model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).features[:28]
    model = model.eval().to(device)
    indices = [part for part in np.array_split(np.arange(len(paths)), min(args.parts, len(paths))) if len(part)]
    string_dtype = h5py.string_dtype(encoding="utf-8")
    for part_index, part in enumerate(tqdm(indices, desc="Extracting VGG19 features"), start=1):
        part_paths = [paths[int(index)] for index in part]
        part_features, part_ratios, part_names = extract_part(
            part_paths, model, device, args.batch_size, args.workers
        )
        with h5py.File(output_dir / f"features_{part_index}.h5", "w") as handle:
            handle.create_dataset("ratios", data=part_ratios)
            handle.create_dataset("features", data=part_features, compression="gzip", compression_opts=9)
            handle.create_dataset("image_names", data=np.asarray(part_names, dtype=object), dtype=string_dtype)
    print(f"Saved {len(paths)} image features in {len(indices)} parts to {output_dir}")


if __name__ == "__main__":
    main()
