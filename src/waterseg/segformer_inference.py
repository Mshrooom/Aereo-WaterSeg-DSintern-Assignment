from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

from waterseg.models.segformer_water import SegformerWaterModel


class SegformerWaterSegmenter:
    """Production inference wrapper for automatic water-body segmentation."""

    def __init__(
        self,
        checkpoint: str | Path,
        device: str = "auto",
        threshold: Optional[float] = None,
    ) -> None:
        checkpoint_path = Path(checkpoint)
        required = [checkpoint_path / "metadata.pt", checkpoint_path / "config.json"]
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "SegFormer checkpoint is incomplete. Missing: " + ", ".join(missing)
            )

        has_weights = any(
            (checkpoint_path / name).exists()
            for name in ("model.safetensors", "pytorch_model.bin")
        )
        if not has_weights:
            raise FileNotFoundError(
                "SegFormer checkpoint must contain model.safetensors or pytorch_model.bin"
            )

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model, self.metadata = SegformerWaterModel.from_checkpoint(
            checkpoint_path,
            self.device,
        )
        self.model.eval()

        saved_threshold = float(self.metadata.get("threshold", 0.5))
        self.threshold = saved_threshold if threshold is None else float(threshold)

    @torch.inference_mode()
    def segment(
        self,
        image: np.ndarray,
        threshold: Optional[float] = None,
        min_component_area: int = 0,
        fill_holes: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected an RGB image with shape H x W x 3")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        height, width = image.shape[:2]
        pixel_values = self.model.prepare_images([image], self.device)
        logits = self.model(pixel_values)
        logits = self.model.resize_logits(logits, (height, width))
        probability = (
            torch.softmax(logits, dim=1)[0, 1]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        used_threshold = self.threshold if threshold is None else float(threshold)
        if not 0.0 <= used_threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")

        mask = (probability >= used_threshold).astype(np.uint8)
        mask = postprocess_mask(
            mask,
            min_component_area=min_component_area,
            fill_holes=fill_holes,
        )
        return mask, probability


def postprocess_mask(
    mask: np.ndarray,
    min_component_area: int = 0,
    fill_holes: bool = False,
) -> np.ndarray:
    """Apply optional connected-component filtering and hole filling."""
    result = (mask > 0).astype(np.uint8)

    if min_component_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            result,
            connectivity=8,
        )
        filtered = np.zeros_like(result)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= min_component_area:
                filtered[labels == index] = 1
        result = filtered

    if fill_holes:
        padded = np.pad(result, 1)
        flood_mask = np.zeros(
            (padded.shape[0] + 2, padded.shape[1] + 2),
            dtype=np.uint8,
        )
        cv2.floodFill(padded, flood_mask, (0, 0), 1)
        holes = 1 - padded[1:-1, 1:-1]
        result = np.maximum(result, holes).astype(np.uint8)

    return result
