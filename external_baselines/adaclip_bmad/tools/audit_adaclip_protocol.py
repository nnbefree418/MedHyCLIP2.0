#!/usr/bin/env python3
"""Audit official AdaCLIP code for protocol behaviors relevant to BMAD LOSO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


PATTERNS = {
    "default_external_training_data": [
        "mvtec",
        "colondb",
        "visa",
        "clinicdb",
        "training_data",
        "testing_data",
    ],
    "target_or_test_data_use": [
        "testing_data",
        "test_data",
        "class_name",
        "class_names",
        "normal",
        "good",
    ],
    "threshold_selection": [
        "threshold",
        "precision_recall_curve",
        "best",
        "optimal",
    ],
    "checkpoint_selection": [
        "save_model",
        "checkpoint",
        "best",
        "validation",
        "val_",
    ],
    "test_statistics_or_calibration": [
        "min()",
        "max()",
        "mean()",
        "std()",
        "normalize",
        "calibration",
    ],
}

TEXT_EXTENSIONS = {".py", ".sh", ".txt", ".md", ".json"}


def iter_text_files(repo: Path) -> list[Path]:
    return [
        path
        for path in repo.rglob("*")
        if path.is_file()
        and path.suffix.lower() in TEXT_EXTENSIONS
        and ".git" not in path.parts
        and "__pycache__" not in path.parts
    ]


def audit_repo(repo: Path) -> dict:
    if not repo.exists():
        return {
            "status": "blocked",
            "reason": f"official AdaCLIP repository not found at {repo}",
            "main_table_allowed": False,
            "findings": [],
            "required_actions": [
                "Place the official AdaCLIP repository at external_baselines/AdaCLIP.",
                "Rerun this audit before any training or inference.",
                "Do not generate main-table results until target/test-data behaviors have been reviewed and disabled.",
            ],
        }

    findings = []
    for path in iter_text_files(repo):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError as exc:
            findings.append({"file": str(path), "category": "read_error", "match": str(exc)})
            continue
        lowered = text.lower()
        for category, terms in PATTERNS.items():
            for term in terms:
                if term.lower() in lowered:
                    findings.append(
                        {
                            "file": str(path.relative_to(repo)),
                            "category": category,
                            "match": term,
                        }
                    )
                    break

    return {
        "status": "review_required" if findings else "pass",
        "reason": "static keyword audit completed",
        "main_table_allowed": False,
        "findings": findings,
        "required_actions": [
            "Disable all non-BMAD default training_data/testing_data settings.",
            "Use only source BMAD datasets for train/adaptation/validation.",
            "Use held-out BMAD target only for final inference and AUROC.",
            "Do not use target test thresholds, statistics, or checkpoint selection.",
        ],
    }


REQUIRED_MANIFEST_FLAGS = {
    "official_train_py_used": False,
    "official_test_py_used": False,
    "official_default_training_data_disabled": True,
    "official_default_testing_data_disabled": True,
    "target_used_for_training": False,
    "target_used_for_validation": False,
    "target_used_for_prompt_adaptation": False,
    "target_used_for_checkpoint_selection": False,
    "target_used_for_threshold_selection": False,
    "checkpoint_selection_uses_target": False,
    "threshold_selection_uses_target": False,
    "outputs_raw_predictions_only": True,
    "main_table_result_generated": False,
}


def apply_manifest_gate(report: dict, manifest_path: Path | None) -> dict:
    if manifest_path is None:
        return report
    if not manifest_path.exists():
        report["status"] = "blocked"
        report["reason"] = f"protocol manifest not found: {manifest_path}"
        return report

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for key, expected in REQUIRED_MANIFEST_FLAGS.items():
        if manifest.get(key) != expected:
            failures.append({"key": key, "expected": expected, "actual": manifest.get(key)})
    if manifest.get("metrics_owner") != "external_baselines/adaclip_bmad/eval_bmad_metrics.py":
        failures.append(
            {
                "key": "metrics_owner",
                "expected": "external_baselines/adaclip_bmad/eval_bmad_metrics.py",
                "actual": manifest.get("metrics_owner"),
            }
        )
    if manifest.get("status") not in {"predictions_exported", "metrics_computed"}:
        failures.append({"key": "status", "expected": "predictions_exported or metrics_computed", "actual": manifest.get("status")})

    report["protocol_manifest"] = str(manifest_path)
    report["manifest_failures"] = failures
    if failures:
        report["status"] = "review_required"
        report["reason"] = "static audit completed, but manifest gate failed"
        report["main_table_allowed"] = False
    else:
        report["status"] = "pass"
        report["reason"] = "official defaults are bypassed by wrapper protocol manifest"
        report["main_table_allowed"] = False
    return report


def write_markdown(report: dict, output: Path) -> None:
    lines = [
        "# AdaCLIP Protocol Audit",
        "",
        f"- Status: {report['status']}",
        f"- Main table allowed before manual resolution: {report['main_table_allowed']}",
        f"- Reason: {report['reason']}",
        "",
        "## Findings",
        "",
    ]
    findings = report.get("findings", [])
    if findings:
        for item in findings[:200]:
            lines.append(f"- `{item.get('file', '-')}`: {item.get('category')} -> `{item.get('match')}`")
        if len(findings) > 200:
            lines.append(f"- ... {len(findings) - 200} additional findings omitted")
    else:
        lines.append("- None")
    lines.extend(["", "## Required Actions", ""])
    for action in report.get("required_actions", []):
        lines.append(f"- {action}")
    if report.get("protocol_manifest"):
        lines.extend(["", "## Protocol Manifest Gate", "", f"- Manifest: `{report['protocol_manifest']}`"])
        failures = report.get("manifest_failures", [])
        if failures:
            lines.append("- Failures:")
            for failure in failures:
                lines.append(
                    f"  - `{failure['key']}` expected `{failure['expected']}`, got `{failure['actual']}`"
                )
        else:
            lines.append("- Failures: none")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adaclip-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "AdaCLIP",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/dryrun/adaclip_protocol_audit.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results/dryrun/adaclip_protocol_audit.md",
    )
    parser.add_argument("--protocol-manifest", type=Path, default=None)
    args = parser.parse_args()

    report = audit_repo(args.adaclip_root)
    report = apply_manifest_gate(report, args.protocol_manifest)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] == "blocked":
        raise SystemExit(2)
    if report["status"] == "review_required":
        raise SystemExit(3)


if __name__ == "__main__":
    main()
