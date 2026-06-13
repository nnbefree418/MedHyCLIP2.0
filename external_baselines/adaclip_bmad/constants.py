"""Shared constants for the BMAD AdaCLIP leave-one-out wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    bmad_name: str
    table_name: str
    has_pixel_masks: bool
    medhyclip_name: str


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("his", "Histopathology", "HIS", False, "histopathological image"),
    DatasetSpec("chestxray", "Chest", "ChestXray", False, "Chest X-ray film"),
    DatasetSpec("oct17", "Retina_OCT2017", "OCT17", False, "retinal OCT"),
    DatasetSpec("brainmri", "Brain", "BrainMRI", True, "Brain"),
    DatasetSpec("liverct", "Liver", "LiverCT", True, "Liver"),
    DatasetSpec("resc", "Retina_RESC", "RESC", True, "retinal OCT"),
)

DATASET_BY_KEY = {spec.key: spec for spec in DATASETS}
DATASET_BY_BMAD_NAME = {spec.bmad_name: spec for spec in DATASETS}

DEFAULT_IMAGE_SIZE = 240
DEFAULT_BACKBONE = "ViT-L-14-336"
DEFAULT_PRETRAIN = "openai"


def repo_root_from_this_file() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_root() -> Path:
    return repo_root_from_this_file() / "data"


def source_keys_for(target_key: str) -> list[str]:
    if target_key not in DATASET_BY_KEY:
        raise KeyError(f"Unknown target dataset key: {target_key}")
    return [spec.key for spec in DATASETS if spec.key != target_key]
