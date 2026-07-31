from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from waterseg.prompting import PromptBatch


@dataclass
class AutoPromptResult:
    prompt: PromptBatch
    coarse_mask: np.ndarray
    status: str
    components: int


def _component_centres(mask: np.ndarray, max_points: int, min_area: int) -> tuple[np.ndarray, int]:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    components = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area >= min_area:
            components.append((area, index, centroids[index]))
    components.sort(reverse=True, key=lambda item: item[0])
    points = []
    for _, label_index, centroid in components[:max_points]:
        x, y = centroid
        x_i = int(np.clip(round(x), 0, mask.shape[1] - 1))
        y_i = int(np.clip(round(y), 0, mask.shape[0] - 1))
        if labels[y_i, x_i] != label_index:
            ys, xs = np.where(labels == label_index)
            middle = len(xs) // 2
            x_i, y_i = int(xs[middle]), int(ys[middle])
        points.append([float(x_i), float(y_i)])
    return np.asarray(points, dtype=np.float32), len(components)


def _negative_points(mask: np.ndarray, count: int) -> np.ndarray:
    if count <= 0:
        return np.empty((0, 2), dtype=np.float32)
    background = (mask == 0).astype(np.uint8)
    if not background.any():
        return np.empty((0, 2), dtype=np.float32)
    distance = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    flat = distance.ravel()
    count = min(count, int((flat > 0).sum()))
    if count == 0:
        return np.empty((0, 2), dtype=np.float32)
    indices = np.argpartition(flat, -count)[-count:]
    ys, xs = np.unravel_index(indices, distance.shape)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def automatic_prompt_from_probability(
    probability: np.ndarray,
    coarse_threshold: float = 0.5,
    max_positive_points: int = 3,
    negative_points: int = 1,
    min_component_area: int = 16,
    box_padding_fraction: float = 0.04,
    morphology_kernel: int = 3,
) -> AutoPromptResult:
    probability = np.asarray(probability, dtype=np.float32)
    coarse = (probability >= coarse_threshold).astype(np.uint8)
    if morphology_kernel > 1:
        kernel = np.ones((morphology_kernel, morphology_kernel), np.uint8)
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_OPEN, kernel)
        coarse = cv2.morphologyEx(coarse, cv2.MORPH_CLOSE, kernel)

    positive, component_count = _component_centres(coarse, max_positive_points, min_component_area)
    if len(positive) == 0:
        return AutoPromptResult(PromptBatch(mode="none"), coarse, "no_coarse_candidate", 0)

    y, x = np.where(coarse > 0)
    height, width = coarse.shape
    pad_x = int(round(width * box_padding_fraction))
    pad_y = int(round(height * box_padding_fraction))
    box = np.array(
        [
            max(0, int(x.min()) - pad_x),
            max(0, int(y.min()) - pad_y),
            min(width - 1, int(x.max()) + pad_x),
            min(height - 1, int(y.max()) + pad_y),
        ],
        dtype=np.float32,
    )
    negative = _negative_points(coarse, negative_points)
    points = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate(
        [np.ones(len(positive), dtype=np.int64), np.zeros(len(negative), dtype=np.int64)]
    )
    prompt = PromptBatch(mode="box_points", points=[points], labels=[labels], boxes=[box])
    return AutoPromptResult(prompt, coarse, "ok", component_count)
