from __future__ import annotations

from pathlib import Path

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
def collect_original_resolution_probabilities(
    model: torch.nn.Module,
    processor,
    dataframe: pd.DataFrame,
    *,
    image_size: int,
    resize_policy: str,
    device: torch.device | str,
    batch_size: int,
    num_workers: int,
) -> list[dict]:
    device = torch.device(device)
    dataset = SegmentationInferenceDataset(
        dataframe,
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
    rows: list[dict] = []
    model.eval()
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
            rows.append(
                {
                    "image_id": batch["image_ids"][index],
                    "target": target,
                    "probability": probability,
                }
            )
    return rows


def calibrate_thresholds_from_probabilities(
    probability_rows: list[dict],
    *,
    thresholds: list[float],
    empty_policy: str,
    output_csv: str | Path,
) -> tuple[float, pd.DataFrame]:
    """Select threshold on validation data only."""
    rows: list[dict] = []
    for threshold in thresholds:
        metrics_by_image = [
            segmentation_metrics(
                row["probability"] >= threshold,
                row["target"],
                include_boundary_metrics=False,
                empty_policy=empty_policy,
            )
            for row in probability_rows
        ]
        rows.append(
            {
                "threshold": float(threshold),
                "iou": float(np.mean([m["iou"] for m in metrics_by_image])),
                "dice": float(np.mean([m["dice"] for m in metrics_by_image])),
                "precision": float(
                    np.mean([m["precision"] for m in metrics_by_image])
                ),
                "recall": float(
                    np.mean([m["recall"] for m in metrics_by_image])
                ),
                "images": int(len(metrics_by_image)),
                "test_split_used": False,
            }
        )
    frame = pd.DataFrame(rows).sort_values("threshold")
    best = frame.sort_values(
        ["iou", "dice"],
        ascending=False,
    ).iloc[0]
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return float(best["threshold"]), frame
