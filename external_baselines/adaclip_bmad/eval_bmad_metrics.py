#!/usr/bin/env python3
"""Compute BMAD AC/AS AUROC from AdaCLIP prediction files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import DATASET_BY_KEY  # noqa: E402


def load_heatmap(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    return np.asarray(Image.open(path).convert("F"), dtype=np.float32)


def load_mask(path: Path) -> np.ndarray:
    return (np.asarray(Image.open(path).convert("L")) > 0).astype(np.uint8)


def resize_heatmap(heatmap: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    heatmap = np.asarray(heatmap, dtype=np.float32)
    if heatmap.ndim == 3:
        heatmap = np.squeeze(heatmap)
    image = Image.fromarray(heatmap.astype(np.float32), mode="F")
    return np.asarray(image.resize((shape[1], shape[0]), resample=Image.BILINEAR), dtype=np.float32)


def read_predictions(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_metrics(dataset_key: str, predictions: list[dict[str, str]]) -> dict:
    spec = DATASET_BY_KEY[dataset_key]
    labels = np.asarray([int(row["label"]) for row in predictions])
    image_scores = np.asarray([float(row["image_score"]) for row in predictions], dtype=np.float32)
    metrics = {
        "dataset_key": dataset_key,
        "dataset": spec.table_name,
        "ac_auroc": float(roc_auc_score(labels, image_scores)),
        "as_auroc": None,
        "num_images": int(len(predictions)),
        "num_as_images": 0,
        "ignored_as_images": 0,
    }

    if spec.has_pixel_masks:
        y_true = []
        y_score = []
        ignored = 0
        for row in predictions:
            mask_path = row.get("mask_path") or ""
            heatmap_path = row.get("heatmap_path") or ""
            if not mask_path or not heatmap_path:
                ignored += 1
                continue
            mask = load_mask(Path(mask_path))
            heatmap = resize_heatmap(load_heatmap(Path(heatmap_path)), mask.shape)
            y_true.append(mask.reshape(-1))
            y_score.append(heatmap.reshape(-1))
        if y_true:
            metrics["as_auroc"] = float(roc_auc_score(np.concatenate(y_true), np.concatenate(y_score)))
            metrics["num_as_images"] = int(len(y_true))
        metrics["ignored_as_images"] = int(ignored)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=sorted(DATASET_BY_KEY))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    metrics = compute_metrics(args.dataset, read_predictions(args.predictions))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
