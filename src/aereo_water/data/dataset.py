from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from aereo_water.data.transforms import (
    SpatialTransform,
    apply_synchronized_augmentation,
    resize_pair,
)


def _binary_mask(mask: Image.Image) -> Image.Image:
    array = (np.asarray(mask.convert("L")) > 0).astype(np.uint8)
    return Image.fromarray(array, mode="L")


class SegmentationTrainingDataset(Dataset):
    """Training dataset with deterministic per-sample synchronized augmentation."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor,
        *,
        image_size: int,
        resize_policy: str,
        augmentation_profile: str,
        base_seed: int,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.processor = processor
        self.image_size = int(image_size)
        self.resize_policy = resize_policy
        self.augmentation_profile = augmentation_profile
        self.base_seed = int(base_seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        with Image.open(row["image_path"]) as raw_image:
            image = raw_image.convert("RGB")
        with Image.open(row["mask_path"]) as raw_mask:
            mask = _binary_mask(raw_mask)

        rng = random.Random(
            self.base_seed + self.epoch * len(self.dataframe) + index
        )
        image, mask = apply_synchronized_augmentation(
            image,
            mask,
            profile=self.augmentation_profile,
            rng=rng,
        )
        image, mask, transform = resize_pair(
            image,
            mask,
            size=self.image_size,
            policy=self.resize_policy,
            # Artificial letterbox pixels must not teach the network that
            # black padding is genuine non-water imagery. 255 is retained as
            # an ignore label by the custom training loss.
            mask_pad_value=(255 if self.resize_policy == "letterbox" else 0),
        )

        encoded = self.processor(
            images=image,
            segmentation_maps=mask,
            return_tensors="pt",
            do_resize=False,
            do_reduce_labels=False,
        )
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "labels": encoded["labels"].squeeze(0).long(),
            "image_id": str(row["image_id"]),
            "transform": transform.to_dict(),
        }


class SegmentationInferenceDataset(Dataset):
    """Inference dataset retaining original-resolution targets and transform metadata."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        processor,
        *,
        image_size: int,
        resize_policy: str,
    ) -> None:
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.processor = processor
        self.image_size = int(image_size)
        self.resize_policy = resize_policy

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.dataframe.iloc[index]
        with Image.open(row["image_path"]) as raw_image:
            original_image = raw_image.convert("RGB")
        with Image.open(row["mask_path"]) as raw_mask:
            original_mask = _binary_mask(raw_mask)

        prepared_image, prepared_mask, transform = resize_pair(
            original_image,
            original_mask,
            size=self.image_size,
            policy=self.resize_policy,
        )
        encoded = self.processor(
            images=prepared_image,
            return_tensors="pt",
            do_resize=False,
        )
        target = np.asarray(original_mask, dtype=np.uint8)
        return {
            "pixel_values": encoded["pixel_values"].squeeze(0),
            "target": target,
            "image_id": str(row["image_id"]),
            "split": str(row["split"]),
            "image_path": str(row["image_path"]),
            "mask_path": str(row["mask_path"]),
            "transform": transform.to_dict(),
        }


def inference_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pixel_values": torch.stack(
            [item["pixel_values"] for item in batch]
        ),
        "targets": [item["target"] for item in batch],
        "image_ids": [item["image_id"] for item in batch],
        "splits": [item["split"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
        "transforms": [item["transform"] for item in batch],
    }
