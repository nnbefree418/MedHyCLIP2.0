#!/usr/bin/env python3
"""Validate BMAD JSONL indices for source/target leakage and path health."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import DATASET_BY_KEY, DATASETS  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def validate(target: str, index_dir: Path) -> tuple[dict, list[str]]:
    errors: list[str] = []
    target_dir = index_dir / target
    records = []
    for name in ("source_train", "source_val", "target_test"):
        path = target_dir / f"{name}.jsonl"
        if not path.exists():
            errors.append(f"missing index file: {path}")
            continue
        records.extend(read_jsonl(path))

    counts = Counter((record.get("split"), record.get("dataset_key"), record.get("label")) for record in records)
    target_keys_in_source = {
        record["dataset_key"]
        for record in records
        if record.get("split") in {"source_train", "source_val"} and record.get("dataset_key") == target
    }
    non_target_in_target = {
        record["dataset_key"]
        for record in records
        if record.get("split") == "target_test" and record.get("dataset_key") != target
    }
    if target_keys_in_source:
        errors.append(f"target dataset appears in source split: {sorted(target_keys_in_source)}")
    if non_target_in_target:
        errors.append(f"non-target datasets appear in target_test: {sorted(non_target_in_target)}")

    source_cases = {
        (record.get("dataset_key"), record.get("case_id"))
        for record in records
        if record.get("split") in {"source_train", "source_val"}
    }
    target_cases = {
        (record.get("dataset_key"), record.get("case_id"))
        for record in records
        if record.get("split") == "target_test"
    }
    overlap = source_cases & target_cases
    if overlap:
        errors.append(f"case_id overlap between source and target: {sorted(list(overlap))[:10]}")

    missing_images = []
    missing_masks = []
    invalid_labels = []
    unexpected_masks = []
    for record in records:
        if record.get("label") not in {0, 1}:
            invalid_labels.append(record)
        if not Path(record["image_path"]).exists():
            missing_images.append(record["image_path"])
        mask_path = record.get("mask_path")
        spec = DATASET_BY_KEY[record["dataset_key"]]
        if mask_path:
            if not spec.has_pixel_masks:
                unexpected_masks.append(mask_path)
            if not Path(mask_path).exists():
                missing_masks.append(mask_path)
    if invalid_labels:
        errors.append(f"invalid labels: {len(invalid_labels)}")
    if missing_images:
        errors.append(f"missing images: {len(missing_images)}; first={missing_images[0]}")
    if missing_masks:
        errors.append(f"missing masks: {len(missing_masks)}; first={missing_masks[0]}")
    if unexpected_masks:
        errors.append(f"unexpected masks for AC-only datasets: {len(unexpected_masks)}")

    report = {
        "target": target,
        "status": "pass" if not errors else "fail",
        "num_records": len(records),
        "counts": {str(key): value for key, value in sorted(counts.items())},
        "errors": errors,
    }
    return report, errors


def write_markdown(reports: list[dict], output: Path) -> None:
    lines = ["# BMAD Index Validation Report", ""]
    for report in reports:
        lines.extend(
            [
                f"## {report['target']}",
                "",
                f"- Status: {report['status']}",
                f"- Records: {report['num_records']}",
                "- Errors:",
            ]
        )
        if report["errors"]:
            lines.extend(f"  - {error}" for error in report["errors"])
        else:
            lines.append("  - None")
        lines.append("")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-dir", type=Path, default=Path(__file__).resolve().parents[1] / "indices")
    parser.add_argument("--target", choices=[spec.key for spec in DATASETS] + ["all"], default="all")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/dryrun/index_validation.md",
    )
    args = parser.parse_args()

    targets = [spec.key for spec in DATASETS] if args.target == "all" else [args.target]
    reports = []
    any_errors = False
    for target in targets:
        report, errors = validate(target, args.index_dir)
        reports.append(report)
        any_errors = any_errors or bool(errors)
    write_markdown(reports, args.report)
    print(json.dumps(reports, indent=2, sort_keys=True))
    if any_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
