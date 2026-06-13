#!/usr/bin/env python3
"""Collect lightweight Euclidean-vs-hyperbolic benchmark results.

The parser consumes only wrapper-generated logs and nvidia-smi samples. It does
not import or modify the training code, which keeps the benchmark independent
from the experiment implementation.
"""

import csv
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CLIP_BACKBONE_PARAMS = 427_650_305
EUCLID_ADAPTER_PARAMS = 12_582_912
HYPER_ADAPTER_PARAMS = 12_597_248

METHOD_PARAMS = {
    "euclid": {
        "label": "MVFA (Euclidean)",
        "total": CLIP_BACKBONE_PARAMS + EUCLID_ADAPTER_PARAMS,
        "trainable": EUCLID_ADAPTER_PARAMS,
        "extra": 0,
    },
    "hyper": {
        "label": "Hyper-MVFA",
        "total": CLIP_BACKBONE_PARAMS + HYPER_ADAPTER_PARAMS,
        "trainable": HYPER_ADAPTER_PARAMS,
        "extra": HYPER_ADAPTER_PARAMS - EUCLID_ADAPTER_PARAMS,
    },
}

TS_RE = re.compile(r"^\[(?P<ts>[^\]]+)\]\s(?P<msg>.*)$")
EPOCH_RE = re.compile(r"\bepoch\s+(\d+)\s*:", re.IGNORECASE)
TQDM_RE = re.compile(
    r"(?P<done>\d+)\s*/\s*(?P<total>\d+)\s*"
    r"\[(?P<elapsed>[0-9:]+)<[^,]*,\s*"
    r"(?P<rate>[0-9.]+)\s*(?P<unit>it/s|s/it)\]"
)

REPO_ROOT = Path(__file__).resolve().parent.parent


class ManifestRow:
    def __init__(
        self,
        task,
        method,
        seed,
        phase,
        dataset,
        tag,
        gpu_ids,
        log_path,
        memory_path,
        start_epoch,
        end_epoch,
        status,
    ):
        self.task = task
        self.method = method
        self.seed = seed
        self.phase = phase
        self.dataset = dataset
        self.tag = tag
        self.gpu_ids = gpu_ids
        self.log_path = log_path
        self.memory_path = memory_path
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.status = status


def parse_iso_timestamp(value):
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+0000"
    elif len(normalized) >= 6 and normalized[-3] == ":" and normalized[-6] in ("+", "-"):
        normalized = normalized[:-3] + normalized[-2:]

    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(normalized, fmt).timestamp()
        except ValueError:
            pass
    return None


def parse_duration_seconds(value):
    parts = value.split(":")
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 1:
        return float(nums[0])
    if len(nums) == 2:
        return float(nums[0] * 60 + nums[1])
    if len(nums) == 3:
        return float(nums[0] * 3600 + nums[1] * 60 + nums[2])
    return None


def read_manifest(result_dir):
    manifest = result_dir / "manifest.tsv"
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    rows = []
    with manifest.open("r", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for item in reader:
            rows.append(
                ManifestRow(
                    task=item["task"],
                    method=item["method"],
                    seed=item["seed"],
                    phase=item["phase"],
                    dataset=item["dataset"],
                    tag=item["tag"],
                    gpu_ids=item["gpu_ids"],
                    log_path=Path(item["log_path"]),
                    memory_path=Path(item["memory_path"]),
                    start_epoch=float(item["start_epoch"]),
                    end_epoch=float(item["end_epoch"]),
                    status=int(item["status"]),
                )
            )
    return rows


def iter_timestamped_messages(log_path):
    if not log_path.exists():
        return
    with log_path.open("r", errors="replace") as handle:
        for line in handle:
            match = TS_RE.match(line.rstrip("\n"))
            if not match:
                continue
            ts = parse_iso_timestamp(match.group("ts"))
            if ts is None:
                continue
            yield ts, match.group("msg")


def training_time_per_epoch(row):
    epoch_starts = []
    seen_epochs = set()
    for ts, msg in iter_timestamped_messages(row.log_path):
        match = EPOCH_RE.search(msg)
        if not match:
            continue
        epoch_idx = int(match.group(1))
        if epoch_idx in seen_epochs:
            continue
        seen_epochs.add(epoch_idx)
        epoch_starts.append(ts)

    if not epoch_starts:
        return None

    durations = []
    for current_ts, next_ts in zip(epoch_starts, epoch_starts[1:]):
        if next_ts > current_ts:
            durations.append(next_ts - current_ts)
    if row.end_epoch > epoch_starts[-1]:
        durations.append(row.end_epoch - epoch_starts[-1])

    durations = [value for value in durations if value > 0]
    if not durations:
        return None
    return statistics.mean(durations)


def count_test_images(dataset):
    dataset_dir = REPO_ROOT / "data" / "{}_AD".format(dataset) / "test"
    total = 0
    for split in ("good", "Ungood"):
        image_dir = dataset_dir / split / "img"
        if not image_dir.exists():
            continue
        total += sum(1 for item in image_dir.iterdir() if item.is_file())
    return total or None


def inference_time_per_image(row):
    last_match = None
    for _, msg in iter_timestamped_messages(row.log_path):
        match = TQDM_RE.search(msg)
        if match and match.group("done") == match.group("total"):
            last_match = match

    if last_match is not None:
        total = int(last_match.group("total"))
        rate = float(last_match.group("rate"))
        unit = last_match.group("unit")
        if unit == "s/it":
            return rate, total
        if rate > 0:
            return 1.0 / rate, total

    totals = []
    elapsed_values = []
    for _, msg in iter_timestamped_messages(row.log_path):
        match = TQDM_RE.search(msg)
        if not match:
            continue
        totals.append(int(match.group("total")))
        elapsed = parse_duration_seconds(match.group("elapsed"))
        if elapsed is not None:
            elapsed_values.append(elapsed)

    if totals:
        total = max(totals)
        elapsed = max(elapsed_values) if elapsed_values else row.end_epoch - row.start_epoch
        if total > 0 and elapsed > 0:
            return elapsed / total, total

    wall = row.end_epoch - row.start_epoch
    image_count = count_test_images(row.dataset)
    if wall > 0 and image_count:
        return wall / image_count, image_count
    return (wall, None) if wall > 0 else (None, None)


def peak_memory_mib(row):
    if not row.memory_path.exists():
        return None

    max_memory = None
    with row.memory_path.open("r", errors="replace") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith("timestamp") or "not found" in stripped:
                continue
            parts = [part.strip() for part in stripped.split(",")]
            if len(parts) < 3:
                continue
            try:
                memory = float(parts[-1])
            except ValueError:
                continue
            max_memory = memory if max_memory is None else max(max_memory, memory)
    return max_memory


def mean(values):
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.mean(clean) if clean else None


def stdev(values):
    clean = [value for value in values if value is not None and not math.isnan(value)]
    return statistics.stdev(clean) if len(clean) > 1 else None


def fmt_millions(value):
    if value is None:
        return "NA"
    return f"{value / 1_000_000:.3f}"


def fmt_seconds(value):
    if value is None:
        return "NA"
    if value >= 3600:
        return f"{value / 3600:.2f} h"
    if value >= 60:
        return f"{value / 60:.2f} min"
    return f"{value:.3f} s"


def fmt_seconds_with_std(value, deviation):
    if value is None:
        return "NA"
    if deviation is None:
        return fmt_seconds(value)
    return f"{fmt_seconds(value)} +/- {fmt_seconds(deviation)}"


def fmt_fps(value):
    if value is None:
        return "NA"
    return f"{value:.2f}"


def fmt_memory(value):
    if value is None:
        return "NA"
    return f"{value / 1024:.2f} GB"


def collect(result_dir):
    rows = read_manifest(result_dir)
    detail_rows = []

    train_times = {}
    infer_times = {}
    peak_memory = {}

    for row in rows:
        train_time = training_time_per_epoch(row) if row.phase == "train" else None
        infer_time, image_count = inference_time_per_image(row) if row.phase == "test" else (None, None)
        memory = peak_memory_mib(row)

        key = (row.method, row.task)
        if row.phase == "train":
            train_times.setdefault(key, []).append(train_time)
        if row.phase == "test":
            infer_times.setdefault(key, []).append(infer_time)
        peak_memory.setdefault(key, []).append(memory)

        detail_rows.append(
            {
                "task": row.task,
                "method": row.method,
                "seed": row.seed,
                "phase": row.phase,
                "dataset": row.dataset,
                "status": str(row.status),
                "train_time_per_epoch_s": "" if train_time is None else f"{train_time:.6f}",
                "infer_time_per_image_s": "" if infer_time is None else f"{infer_time:.6f}",
                "image_count": "" if image_count is None else str(image_count),
                "peak_memory_mib": "" if memory is None else f"{memory:.1f}",
                "log_path": str(row.log_path),
            }
        )

    summary_rows = []
    for method in ("euclid", "hyper"):
        params = METHOD_PARAMS[method]
        zero_train = mean(train_times.get((method, "zero"), []))
        few_train = mean(train_times.get((method, "few"), []))
        zero_infer = mean(infer_times.get((method, "zero"), []))
        few_infer = mean(infer_times.get((method, "few"), []))
        zero_fps = (1.0 / zero_infer) if zero_infer and zero_infer > 0 else None
        few_fps = (1.0 / few_infer) if few_infer and few_infer > 0 else None
        zero_memory = mean(peak_memory.get((method, "zero"), []))
        few_memory = mean(peak_memory.get((method, "few"), []))

        summary_rows.append(
            {
                "Method": params["label"],
                "Total Params (M)": fmt_millions(params["total"]),
                "Trainable Params (M)": fmt_millions(params["trainable"]),
                "Extra Params (M)": fmt_millions(params["extra"]),
                "Zero-shot Train Time / Epoch": fmt_seconds_with_std(
                    zero_train, stdev(train_times.get((method, "zero"), []))
                ),
                "Few-shot Train Time / Epoch": fmt_seconds_with_std(
                    few_train, stdev(train_times.get((method, "few"), []))
                ),
                "Zero-shot Infer Time / Image": fmt_seconds_with_std(
                    zero_infer, stdev(infer_times.get((method, "zero"), []))
                ),
                "Few-shot Infer Time / Image": fmt_seconds_with_std(
                    few_infer, stdev(infer_times.get((method, "few"), []))
                ),
                "FPS": f"zero {fmt_fps(zero_fps)} / few {fmt_fps(few_fps)}",
                "Peak Memory": f"zero {fmt_memory(zero_memory)} / few {fmt_memory(few_memory)}",
            }
        )
    return summary_rows, detail_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows):
    if not rows:
        return ""
    headers = list(rows[0].keys())
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row[header] for header in headers) + " |")
    return "\n".join(lines)


def write_markdown(path, summary_rows, detail_rows):
    failed = [row for row in detail_rows if row["status"] != "0"]
    lines = [
        "# Lightweight Euclidean vs Hyperbolic Benchmark",
        "",
        "## Summary",
        "",
        markdown_table(summary_rows),
        "",
        "## Notes",
        "",
        "- `Trainable Params` counts only `seg_adapters` and `det_adapters`, matching the optimizer target in the existing scripts.",
        "- `Extra Params` is measured relative to MVFA (Euclidean).",
        "- `Training Time / Epoch` is measured from wrapper wall-clock timestamps and includes the current scripts' epoch-end evaluation.",
        "- `Inference Time / Image` is parsed from `tqdm` test throughput when available, otherwise from test wall time.",
        "- `Peak Memory` is averaged over runs and reported as max memory used on a single monitored GPU.",
        "",
        "## Failed Or Incomplete Runs",
        "",
    ]
    if failed:
        lines.extend(f"- `{row['task']}/{row['method']}/seed={row['seed']}/{row['phase']}` exited with status `{row['status']}`." for row in failed)
    else:
        lines.append("- None detected in the manifest.")
    lines.extend(
        [
            "",
            "## Detail CSV",
            "",
            f"- `benchmark_details.csv` contains per-run parsed values and log paths.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main(argv):
    if len(argv) != 2:
        print("Usage: collect_lightweight_benchmark.py <result_dir>", file=sys.stderr)
        return 2

    result_dir = Path(argv[1]).resolve()
    summary_rows, detail_rows = collect(result_dir)

    write_csv(result_dir / "benchmark_summary.csv", summary_rows)
    write_csv(result_dir / "benchmark_details.csv", detail_rows)
    write_markdown(result_dir / "benchmark_summary.md", summary_rows, detail_rows)

    print(markdown_table(summary_rows))
    print(f"\nWrote: {result_dir / 'benchmark_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
