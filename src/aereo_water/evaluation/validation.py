from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aereo_water.data.dataset import (
    SegmentationInferenceDataset,
    inference_collate,
)
from aereo_water.data.transforms import restore_probability_to_original
from aereo_water.evaluation.metrics import segmentation_metrics


@torch.inference_mode()
def validate_original_resolution(
    model: torch.nn.Module,
    processor,
    validation_df: pd.DataFrame,
    *,
    image_size: int,
    resize_policy: str,
    threshold: float,
    device: torch.device | str,
    batch_size: int,
    num_workers: int,
    empty_policy: str,
) -> dict[str, float]:
    """Model-selection metric matching final original-resolution evaluation."""
    device = torch.device(device)
    dataset = SegmentationInferenceDataset(
        validation_df,
        processor,
        image_size=image_size,
        resize_policy=resize_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=inference_collate,
    )
    model.eval()
    ious: list[float] = []
    dices: list[float] = []
    losses: list[float] = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        logits = model(pixel_values=pixel_values).logits
        logits_canvas = F.interpolate(
            logits,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits_canvas, dim=1)[:, 1]
        for index, target in enumerate(batch["targets"]):
            probability = restore_probability_to_original(
                probabilities[index].detach().cpu().numpy(),
                batch["transforms"][index],
            )
            prediction = probability >= threshold
            metrics = segmentation_metrics(
                prediction,
                target,
                include_boundary_metrics=False,
                empty_policy=empty_policy,
            )
            ious.append(metrics["iou"])
            dices.append(metrics["dice"])
            target_float = target.astype(np.float32)
            probability_clipped = np.clip(probability, 1e-6, 1 - 1e-6)
            bce = -np.mean(
                target_float * np.log(probability_clipped)
                + (1 - target_float)
                * np.log(1 - probability_clipped)
            )
            losses.append(float(bce))

    return {
        "val_original_iou": float(np.mean(ious)),
        "val_original_dice": float(np.mean(dices)),
        "val_original_bce": float(np.mean(losses)),
        "validation_images": int(len(ious)),
    }
