from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict

import cv2
import numpy as np


@dataclass
class JointTransform:
    training: bool = True
    horizontal_flip_p: float = 0.5
    vertical_flip_p: float = 0.5
    rotate90_p: float = 0.5
    color_jitter_p: float = 0.35
    blur_p: float = 0.10
    noise_p: float = 0.10

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> Dict[str, np.ndarray]:
        if not self.training:
            return {"image": np.ascontiguousarray(image), "mask": np.ascontiguousarray(mask)}

        if random.random() < self.horizontal_flip_p:
            image, mask = np.fliplr(image), np.fliplr(mask)
        if random.random() < self.vertical_flip_p:
            image, mask = np.flipud(image), np.flipud(mask)
        if random.random() < self.rotate90_p:
            k = random.randint(1, 3)
            image, mask = np.rot90(image, k), np.rot90(mask, k)
        if random.random() < self.color_jitter_p:
            image_float = image.astype(np.float32)
            contrast = random.uniform(0.85, 1.15)
            brightness = random.uniform(-20.0, 20.0)
            image = np.clip(image_float * contrast + brightness, 0, 255).astype(np.uint8)
        if random.random() < self.blur_p:
            image = cv2.GaussianBlur(image, (3, 3), 0)
        if random.random() < self.noise_p:
            noise = np.random.normal(0, random.uniform(2.0, 8.0), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return {"image": np.ascontiguousarray(image), "mask": np.ascontiguousarray(mask)}
