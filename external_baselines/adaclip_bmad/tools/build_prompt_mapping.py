#!/usr/bin/env python3
"""Build BMAD-to-AdaCLIP prompt/category mapping from current project prompts."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import DATASETS  # noqa: E402


def read_real_name(prompt_py: Path) -> dict[str, str]:
    tree = ast.parse(prompt_py.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "REAL_NAME":
                    value = ast.literal_eval(node.value)
                    if not isinstance(value, dict):
                        raise TypeError("REAL_NAME is not a dict")
                    return {str(k): str(v) for k, v in value.items()}
    raise ValueError(f"REAL_NAME not found in {prompt_py}")


def yaml_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def build_mapping(real_name: dict[str, str]) -> str:
    lines = [
        "# Auto-generated BMAD prompt mapping for AdaCLIP.",
        "# Source priority: MedHyCLIP/MVFA prompt.py REAL_NAME.",
        "datasets:",
    ]
    for spec in DATASETS:
        prompt_name = real_name.get(spec.bmad_name, spec.medhyclip_name)
        lines.extend(
            [
                f"  {spec.key}:",
                f"    table_name: {yaml_quote(spec.table_name)}",
                f"    bmad_name: {yaml_quote(spec.bmad_name)}",
                f"    adaclip_category: {yaml_quote(prompt_name)}",
                f"    adaclip_object_name: {yaml_quote(prompt_name)}",
                f"    normal_prompt_name: {yaml_quote(prompt_name)}",
                f"    abnormal_prompt_name: {yaml_quote(prompt_name)}",
                f"    has_pixel_masks: {str(spec.has_pixel_masks).lower()}",
                "    source: \"prompt.py REAL_NAME\"",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-py", type=Path, default=ROOT / "prompt.py")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "external_baselines/adaclip_bmad/configs/prompt_mapping.yaml",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite to regenerate")

    real_name = read_real_name(args.prompt_py)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_mapping(real_name), encoding="utf-8")
    print(f"Wrote prompt mapping: {args.output}")


if __name__ == "__main__":
    main()
