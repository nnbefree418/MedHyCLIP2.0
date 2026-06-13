#!/usr/bin/env python3
"""AdaCLIP runner boundary for BMAD LOSO.

This wrapper intentionally avoids official train.py/test.py entry points so
that BMAD source/target boundaries, checkpoint policy, and metric computation
stay under our control.
"""

from __future__ import annotations

import argparse
import csv
import os
import json
from pathlib import Path
import sys
import types
from contextlib import contextmanager

import numpy as np

WRAPPER_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WRAPPER_ROOT))

from adapters.bmad_dataset import BMADAdaCLIPDataset  # noqa: E402


def write_blocked_report(output: Path, reason: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                "# AdaCLIP Runner Feasibility Report",
                "",
                "- Status: blocked",
                f"- Reason: {reason}",
                "- Main table result generated: no",
                "",
                "The BMAD index, prompt mapping, leakage validation, and metric",
                "pipeline can be prepared independently, but official AdaCLIP",
                "training/adaptation and inference require the official repository",
                "and weights/code dependencies to be available.",
                "",
            ]
        ),
        encoding="utf-8",
    )


@contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_prompt_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    current_key = None
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        if line.startswith("  ") and line.strip().endswith(":") and not line.startswith("    "):
            current_key = line.strip()[:-1]
        elif current_key and "adaclip_category:" in line:
            value = line.split(":", 1)[1].strip().strip('"')
            mapping[current_key] = value
    return mapping


def load_official_trainer(adaclip_root: Path):
    sys.path.insert(0, str(adaclip_root))
    # method.trainer imports `tools` only for evaluation/logging helpers. The
    # dry-run wrapper bypasses official metrics, so provide a tiny stub to avoid
    # pulling tensorboard through official tools/training_tools.py.
    if "tools" not in sys.modules:
        tools_stub = types.ModuleType("tools")
        visualization_stub = types.SimpleNamespace(plot_sample_cv2=lambda *args, **kwargs: None)
        tools_stub.visualization = visualization_stub
        tools_stub.calculate_metric = lambda *args, **kwargs: {}
        tools_stub.calculate_average_metric = lambda *args, **kwargs: {}
        sys.modules["tools"] = tools_stub
    from method import AdaCLIP_Trainer  # type: ignore

    return AdaCLIP_Trainer


def build_model(args, adaclip_root: Path):
    with pushd(adaclip_root):
        config_path = adaclip_root / "model_configs" / f"{args.backbone}.json"
        model_configs = json.loads(config_path.read_text(encoding="utf-8"))
        n_layers = model_configs["vision_cfg"]["layers"]
        substage = n_layers // 4
        features_list = [substage, substage * 2, substage * 3, substage * 4]

        import torch

        trainer_cls = load_official_trainer(adaclip_root)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = trainer_cls(
            backbone=args.backbone,
            feat_list=features_list,
            input_dim=model_configs["vision_cfg"]["width"],
            output_dim=model_configs["embed_dim"],
            learning_rate=0.0,
            device=device,
            image_size=args.image_size,
            prompting_depth=args.prompting_depth,
            prompting_length=args.prompting_length,
            prompting_branch=args.prompting_branch,
            prompting_type=args.prompting_type,
            use_hsf=args.use_hsf,
            k_clusters=args.k_clusters,
        ).to(device)
        model.eval()
        return model


def write_predictions(model, dataset, output_dir: Path, predictions_csv: Path) -> int:
    import torch
    from torch.utils.data import DataLoader

    output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    rows = []
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            image = batch["img"].to(model.device)
            cls_name = list(batch["cls_name"])
            anomaly_map, anomaly_score = model.clip_model(image, cls_name, aggregation=True)
            heatmap = anomaly_map[0].detach().cpu().numpy().astype(np.float32)
            score = float(anomaly_score[0].detach().cpu().item())
            heatmap_path = heatmap_dir / f"{idx:04d}.npy"
            np.save(heatmap_path, heatmap)
            rows.append(
                {
                    "dataset_key": batch["dataset_key"][0],
                    "dataset": batch["dataset"][0],
                    "image_path": batch["image_path"][0],
                    "mask_path": batch["mask_path"][0],
                    "label": int(batch["label"][0].item()),
                    "image_score": score,
                    "heatmap_path": str(heatmap_path.resolve()),
                    "cls_name": cls_name[0],
                }
            )

    predictions_csv.parent.mkdir(parents=True, exist_ok=True)
    with predictions_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_protocol_manifest(args, output: Path, status: str, predictions: Path | None = None) -> None:
    manifest = {
        "status": status,
        "target": args.target,
        "adaclip_root": str(args.adaclip_root),
        "official_train_py_used": False,
        "official_test_py_used": False,
        "official_default_training_data_disabled": True,
        "official_default_testing_data_disabled": True,
        "source_train_index": str(args.source_train_index),
        "source_val_index": str(args.source_val_index),
        "target_test_index": str(args.target_test_index),
        "training_performed": False,
        "prompt_adaptation_performed": False,
        "checkpoint_policy": "none_for_dryrun_forward_only",
        "checkpoint_selection_uses_target": False,
        "threshold_selection_uses_target": False,
        "target_used_for_training": False,
        "target_used_for_validation": False,
        "target_used_for_prompt_adaptation": False,
        "target_used_for_checkpoint_selection": False,
        "target_used_for_threshold_selection": False,
        "outputs_raw_predictions_only": True,
        "metrics_owner": "external_baselines/adaclip_bmad/eval_bmad_metrics.py",
        "main_table_result_generated": False,
        "predictions_csv": str(predictions) if predictions else None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaclip-root", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--source-train-index", type=Path, required=True)
    parser.add_argument("--source-val-index", type=Path, required=True)
    parser.add_argument("--target-test-index", type=Path, required=True)
    parser.add_argument("--prompt-mapping", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=240)
    parser.add_argument("--backbone", default="ViT-L-14-336")
    parser.add_argument("--prompting-depth", type=int, default=4)
    parser.add_argument("--prompting-length", type=int, default=5)
    parser.add_argument("--prompting-type", default="SD", choices=["", "S", "D", "SD"])
    parser.add_argument("--prompting-branch", default="VL", choices=["", "V", "L", "VL"])
    parser.add_argument("--use-hsf", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--k-clusters", type=int, default=20)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--dryrun", action="store_true")
    args = parser.parse_args()

    if not args.adaclip_root.exists():
        write_blocked_report(
            args.output_dir / "feasibility_report.md",
            f"official AdaCLIP repository not found at {args.adaclip_root}",
        )
        raise SystemExit(2)

    predictions_csv = args.predictions_csv or args.output_dir / "dryrun_predictions.csv"
    protocol_manifest = args.protocol_manifest or args.output_dir / "protocol_manifest.json"
    try:
        prompt_mapping = load_prompt_mapping(args.prompt_mapping)
        model = build_model(args, args.adaclip_root)
        dataset = BMADAdaCLIPDataset(
            args.target_test_index,
            image_size=args.image_size,
            transform=model.preprocess,
            target_transform=model.transform,
            class_name_map=prompt_mapping,
        )
        num_predictions = write_predictions(model, dataset, args.output_dir, predictions_csv)
    except Exception as exc:
        write_protocol_manifest(args, protocol_manifest, status="blocked", predictions=predictions_csv)
        write_blocked_report(args.output_dir / "feasibility_report.md", f"AdaCLIP dry-run failed: {exc}")
        raise

    write_protocol_manifest(args, protocol_manifest, status="predictions_exported", predictions=predictions_csv)
    metadata = {
        "status": "predictions_exported",
        "target": args.target,
        "adaclip_root": str(args.adaclip_root),
        "source_train_index": str(args.source_train_index),
        "source_val_index": str(args.source_val_index),
        "target_test_index": str(args.target_test_index),
        "prompt_mapping": str(args.prompt_mapping),
        "image_size": args.image_size,
        "backbone": args.backbone,
        "dryrun": args.dryrun,
        "predictions_csv": str(predictions_csv),
        "num_predictions": num_predictions,
        "protocol_manifest": str(protocol_manifest),
        "note": "Official train.py/test.py were not used; raw predictions were exported for wrapper-owned AUROC evaluation.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "runner_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
