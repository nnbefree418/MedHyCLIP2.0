#!/usr/bin/env python3
"""Collect BMAD AdaCLIP metrics into CSV and a paper table row."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from constants import DATASETS  # noqa: E402


def fmt(value: float | None) -> str:
    return "--" if value is None else f"{value * 100:.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-dir", type=Path, required=True)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--table-row", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for spec in DATASETS:
        path = args.metrics_dir / f"{spec.key}_metrics.json"
        if not path.exists():
            raise FileNotFoundError(path)
        metrics = json.loads(path.read_text(encoding="utf-8"))
        rows.append(metrics)

    ac_avg = sum(row["ac_auroc"] for row in rows) / len(rows)
    as_values = [row["as_auroc"] for row in rows if row["as_auroc"] is not None]
    as_avg = sum(as_values) / len(as_values) if as_values else None

    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "ac_auroc", "as_auroc", "num_images", "num_as_images", "ignored_as_images"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "ac_auroc": row["ac_auroc"],
                    "as_auroc": row["as_auroc"] if row["as_auroc"] is not None else "--",
                    "num_images": row["num_images"],
                    "num_as_images": row["num_as_images"],
                    "ignored_as_images": row["ignored_as_images"],
                }
            )
        writer.writerow(
            {
                "dataset": "Average",
                "ac_auroc": ac_avg,
                "as_auroc": as_avg if as_avg is not None else "--",
                "num_images": "",
                "num_as_images": "",
                "ignored_as_images": "",
            }
        )

    cells = [f"{fmt(row['ac_auroc'])} / {fmt(row['as_auroc'])}" for row in rows]
    cells.append(f"{fmt(ac_avg)} / {fmt(as_avg)}")
    args.table_row.write_text(
        "AdaCLIP~\\cite{cao2024adaclip} & " + " & ".join(cells) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {args.summary_csv}")
    print(f"Wrote {args.table_row}")


if __name__ == "__main__":
    main()
