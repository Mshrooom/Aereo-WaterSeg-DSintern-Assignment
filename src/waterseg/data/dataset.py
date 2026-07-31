from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from waterseg.data.manifest import read_mask, read_rgb


class WaterDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, transform: Any = None):
        self.manifest = manifest.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.manifest.iloc[index]
        image = read_rgb(row.image_path)
        mask = read_mask(row.mask_path)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        return {
            "image": image,
            "mask": mask.astype(np.uint8),
            "image_id": str(row.image_id),
            "image_path": str(row.image_path),
            "original_size": tuple(mask.shape),
        }


def list_collate(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "images": [item["image"] for item in batch],
        "masks": [item["mask"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "original_sizes": [item["original_size"] for item in batch],
    }
