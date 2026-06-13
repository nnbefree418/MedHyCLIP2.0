"""BMAD JSONL dataset adapter for AdaCLIP wrapper code."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

try:
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
except ImportError:  # pragma: no cover - lets validation scripts import metadata without torch.
    torch = None
    Dataset = object
    transforms = None


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


class BMADAdaCLIPDataset(Dataset):
    """A minimal PyTorch dataset over generated BMAD AdaCLIP index files."""

    def __init__(
        self,
        index_path: str | Path,
        image_size: int = 240,
        transform: Any | None = None,
        target_transform: Any | None = None,
        class_name_map: dict[str, str] | None = None,
    ) -> None:
        if torch is None or transforms is None:
            raise ImportError("torch and torchvision are required for BMADAdaCLIPDataset")
        self.index_path = Path(index_path)
        self.records = read_jsonl(self.index_path)
        self.image_size = image_size
        self.image_transform = transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size), Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )
        self.mask_transform = target_transform or transforms.Compose(
            [
                transforms.Resize((image_size, image_size), Image.NEAREST),
                transforms.ToTensor(),
            ]
        )
        self.class_name_map = class_name_map or {}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        image = Image.open(record["image_path"]).convert("RGB")
        image_tensor = self.image_transform(image)
        mask_path = record.get("mask_path")
        if mask_path:
            mask = Image.open(mask_path).convert("L")
            mask_tensor = self.mask_transform(mask)
            mask_tensor = (mask_tensor > 0.5).float()
        else:
            mask_tensor = torch.zeros((1, self.image_size, self.image_size), dtype=torch.float32)

        label = torch.tensor(int(record["label"]), dtype=torch.long)
        cls_name = self.class_name_map.get(record["dataset_key"], record["dataset"])
        return {
            "img": image_tensor,
            "img_mask": mask_tensor,
            "cls_name": cls_name,
            "anomaly": label,
            "image_path": record["image_path"],
            "img_path": record["image_path"],
            "mask_path": mask_path or "",
            "dataset_key": record["dataset_key"],
            "dataset": record["dataset"],
            "split": record["split"],
            "case_id": record.get("case_id", ""),
            "image": image_tensor,
            "label": label,
            "mask": mask_tensor,
        }
