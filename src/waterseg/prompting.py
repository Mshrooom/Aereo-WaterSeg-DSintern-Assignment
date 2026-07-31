from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np


@dataclass
class PromptBatch:
    mode: str
    points: Optional[List[np.ndarray]] = None
    labels: Optional[List[np.ndarray]] = None
    boxes: Optional[List[np.ndarray]] = None


def _sample_coordinates(binary: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    coordinates = np.argwhere(binary > 0)
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(coordinates) == 0:
        return np.zeros((count, 2), dtype=np.float32)
    selected = coordinates[rng.choice(len(coordinates), size=count, replace=len(coordinates) < count)]
    return selected[:, ::-1].astype(np.float32)  # y,x -> x,y


def _positive_region(mask: np.ndarray) -> np.ndarray:
    if not mask.any():
        return mask
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
    return eroded if eroded.any() else mask


def _negative_region(mask: np.ndarray) -> np.ndarray:
    background = (mask == 0).astype(np.uint8)
    kernel = np.ones((7, 7), np.uint8)
    eroded = cv2.erode(background, kernel, iterations=1)
    return eroded if eroded.any() else background


def box_from_mask(mask: np.ndarray, jitter_fraction: float, rng: np.random.Generator) -> np.ndarray:
    y, x = np.where(mask > 0)
    height, width = mask.shape
    if len(x) == 0:
        return np.array([0, 0, width - 1, height - 1], dtype=np.float32)
    x0, x1, y0, y1 = x.min(), x.max(), y.min(), y.max()
    jitter_x = int(round(width * jitter_fraction))
    jitter_y = int(round(height * jitter_fraction))
    x0 += int(rng.integers(-jitter_x, jitter_x + 1)) if jitter_x else 0
    x1 += int(rng.integers(-jitter_x, jitter_x + 1)) if jitter_x else 0
    y0 += int(rng.integers(-jitter_y, jitter_y + 1)) if jitter_y else 0
    y1 += int(rng.integers(-jitter_y, jitter_y + 1)) if jitter_y else 0
    x0, x1 = sorted((np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)))
    y0, y1 = sorted((np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)))
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def build_prompt_batch(
    masks: List[np.ndarray],
    mode: str,
    positive_points: int,
    negative_points: int,
    box_jitter_fraction: float,
    rngs: List[np.random.Generator],
) -> PromptBatch:
    if mode == "none":
        return PromptBatch(mode=mode)

    points, labels, boxes = [], [], []
    for mask, rng in zip(masks, rngs):
        if mode in {"point1", "points", "box_points"}:
            n_pos = 1 if mode == "point1" else positive_points
            n_neg = 0 if mode == "point1" else negative_points
            if mask.any():
                positive = _sample_coordinates(_positive_region(mask), n_pos, rng)
                positive_labels = np.ones(len(positive), dtype=np.int64)
            else:
                # Empty-mask samples must never receive a false foreground prompt.
                positive = _sample_coordinates(_negative_region(mask), n_pos, rng)
                positive_labels = np.zeros(len(positive), dtype=np.int64)
            negative = _sample_coordinates(_negative_region(mask), n_neg, rng)
            point_array = np.concatenate([positive, negative], axis=0)
            label_array = np.concatenate(
                [positive_labels, np.zeros(len(negative), dtype=np.int64)]
            )
            points.append(point_array)
            labels.append(label_array)
        if mode in {"box", "box_points"}:
            boxes.append(box_from_mask(mask, box_jitter_fraction, rng))

    if mode not in {"point1", "points", "box", "box_points"}:
        raise ValueError(f"Unsupported prompt mode: {mode}")
    return PromptBatch(mode=mode, points=points or None, labels=labels or None, boxes=boxes or None)


def processor_prompt_kwargs(prompts: PromptBatch) -> dict:
    kwargs = {}
    # HF SAM uses a point-batch dimension: B x 1 x N x 2.
    if prompts.points is not None:
        kwargs["input_points"] = [[points.tolist()] for points in prompts.points]
        kwargs["input_labels"] = [[labels.tolist()] for labels in prompts.labels or []]
    if prompts.boxes is not None:
        kwargs["input_boxes"] = [[box.tolist()] for box in prompts.boxes]
    return kwargs
