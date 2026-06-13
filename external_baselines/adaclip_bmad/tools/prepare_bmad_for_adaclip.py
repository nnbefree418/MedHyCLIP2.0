#!/usr/bin/env python3
"""Create BMAD leave-one-out JSONL indices for AdaCLIP wrappers.

The script only reads the existing BMAD data tree and writes new index files
under external_baselines/adaclip_bmad.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import DATASET_BY_KEY, DATASETS, default_data_root, source_keys_for  # noqa: E402

IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def image_files(path: Path) -> list[Path]:
    if not path.exists():
        raise FileNotFoundError(f"Missing image directory: {path}")
    return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def mask_path_for(image_path: Path) -> Path:
    return image_path.parent.parent / "anomaly_mask" / image_path.name


def records_for_dataset(data_root: Path, key: str, split: str) -> list[dict]:
    spec = DATASET_BY_KEY[key]
    dataset_root = data_root / f"{spec.bmad_name}_AD" / "test"
    records: list[dict] = []

    for label_name, label in (("good", 0), ("Ungood", 1)):
        for image_path in image_files(dataset_root / label_name / "img"):
            mask_path = None
            if spec.has_pixel_masks and label == 1:
                mask_path = str(mask_path_for(image_path).resolve())
            records.append(
                {
                    "dataset_key": spec.key,
                    "dataset": spec.table_name,
                    "bmad_name": spec.bmad_name,
                    "split": split,
                    "label": label,
                    "image_path": str(image_path.resolve()),
                    "mask_path": mask_path,
                    "has_pixel_mask": bool(mask_path),
                    "case_id": image_path.stem,
                }
            )
    return records


def stratified_source_split(records: list[dict], val_fraction: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    train: list[dict] = []
    val: list[dict] = []
    grouped: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["dataset_key"], int(record["label"])), []).append(record)

    for group in grouped.values():
        group = list(group)
        rng.shuffle(group)
        val_count = max(1, int(round(len(group) * val_fraction))) if len(group) > 1 else 0
        val.extend(group[:val_count])
        train.extend(group[val_count:])

    for record in train:
        record["split"] = "source_train"
    for record in val:
        record["split"] = "source_val"
    return train, val


def limit_by_group(records: list[dict], per_group: int) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["dataset_key"], int(record["label"])), []).append(record)
    limited: list[dict] = []
    for key in sorted(grouped):
        limited.extend(grouped[key][:per_group])
    return limited


def write_jsonl(path: Path, records: list[dict], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to regenerate")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def make_target_index(
    target_key: str,
    data_root: Path,
    output_dir: Path,
    val_fraction: float,
    seed: int,
    overwrite: bool,
    dryrun: bool,
) -> dict:
    source_records: list[dict] = []
    for key in source_keys_for(target_key):
        source_records.extend(records_for_dataset(data_root, key, "source_train"))
    source_train, source_val = stratified_source_split(source_records, val_fraction, seed)
    target_test = records_for_dataset(data_root, target_key, "target_test")

    if dryrun:
        source_train = limit_by_group(source_train, per_group=2)
        source_val = limit_by_group(source_val, per_group=1)
        normal = [record for record in target_test if record["label"] == 0][:4]
        abnormal = [record for record in target_test if record["label"] == 1][:4]
        target_test = normal + abnormal

    target_dir = output_dir / target_key
    write_jsonl(target_dir / "source_train.jsonl", source_train, overwrite)
    write_jsonl(target_dir / "source_val.jsonl", source_val, overwrite)
    write_jsonl(target_dir / "target_test.jsonl", target_test, overwrite)
    write_jsonl(target_dir / "all.jsonl", source_train + source_val + target_test, overwrite)

    return {
        "target": target_key,
        "source_train": len(source_train),
        "source_val": len(source_val),
        "target_test": len(target_test),
        "source_datasets": source_keys_for(target_key),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "indices",
    )
    parser.add_argument("--target", choices=[spec.key for spec in DATASETS] + ["all"], default="all")
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=111)
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    targets = [spec.key for spec in DATASETS] if args.target == "all" else [args.target]
    summary = [
        make_target_index(
            target,
            args.data_root,
            args.output_dir,
            args.val_fraction,
            args.seed,
            args.overwrite,
            args.dryrun,
        )
        for target in targets
    ]
    summary_path = args.output_dir / ("dryrun_summary.json" if args.dryrun else "summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
