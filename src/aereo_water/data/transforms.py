from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from PIL import Image, ImageEnhance


ResizePolicy = Literal["letterbox", "stretch"]


@dataclass(frozen=True)
class SpatialTransform:
    policy: ResizePolicy
    original_width: int
    original_height: int
    canvas_width: int
    canvas_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    pad_right: int
    pad_bottom: int

    def to_dict(self) -> dict:
        return asdict(self)


def resize_pair(
    image: Image.Image,
    mask: Image.Image,
    *,
    size: int,
    policy: ResizePolicy,
    mask_pad_value: int = 0,
) -> tuple[Image.Image, Image.Image, SpatialTransform]:
    image = image.convert("RGB")
    mask = mask.convert("L")
    original_width, original_height = image.size
    if mask.size != image.size:
        raise ValueError(
            f"Image and mask dimensions differ: {image.size} vs {mask.size}"
        )

    if policy == "stretch":
        resized_image = image.resize(
            (size, size),
            Image.Resampling.BILINEAR,
        )
        resized_mask = mask.resize(
            (size, size),
            Image.Resampling.NEAREST,
        )
        transform = SpatialTransform(
            policy="stretch",
            original_width=original_width,
            original_height=original_height,
            canvas_width=size,
            canvas_height=size,
            resized_width=size,
            resized_height=size,
            pad_left=0,
            pad_top=0,
            pad_right=0,
            pad_bottom=0,
        )
        return resized_image, resized_mask, transform

    if policy != "letterbox":
        raise ValueError(f"Unsupported resize policy: {policy}")

    scale = min(size / original_width, size / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))
    resized_image = image.resize(
        (resized_width, resized_height),
        Image.Resampling.BILINEAR,
    )
    resized_mask = mask.resize(
        (resized_width, resized_height),
        Image.Resampling.NEAREST,
    )

    pad_left = (size - resized_width) // 2
    pad_top = (size - resized_height) // 2
    pad_right = size - resized_width - pad_left
    pad_bottom = size - resized_height - pad_top

    image_canvas = Image.new("RGB", (size, size), color=(0, 0, 0))
    if not 0 <= int(mask_pad_value) <= 255:
        raise ValueError("mask_pad_value must be between 0 and 255")
    mask_canvas = Image.new(
        "L",
        (size, size),
        color=int(mask_pad_value),
    )
    image_canvas.paste(resized_image, (pad_left, pad_top))
    mask_canvas.paste(resized_mask, (pad_left, pad_top))

    transform = SpatialTransform(
        policy="letterbox",
        original_width=original_width,
        original_height=original_height,
        canvas_width=size,
        canvas_height=size,
        resized_width=resized_width,
        resized_height=resized_height,
        pad_left=pad_left,
        pad_top=pad_top,
        pad_right=pad_right,
        pad_bottom=pad_bottom,
    )
    return image_canvas, mask_canvas, transform


def apply_synchronized_augmentation(
    image: Image.Image,
    mask: Image.Image,
    *,
    profile: str,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    """Apply identical spatial transforms and image-only photometric transforms."""
    if profile == "none":
        return image, mask
    if profile not in {"light", "moderate"}:
        raise ValueError(f"Unknown augmentation profile: {profile}")

    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

    if profile == "moderate":
        rotation = rng.choice([0, 90, 180, 270])
        if rotation:
            image = image.rotate(rotation, resample=Image.Resampling.BILINEAR)
            mask = mask.rotate(rotation, resample=Image.Resampling.NEAREST)

        brightness = rng.uniform(0.90, 1.10)
        contrast = rng.uniform(0.90, 1.10)
        image = ImageEnhance.Brightness(image).enhance(brightness)
        image = ImageEnhance.Contrast(image).enhance(contrast)

    return image, mask


def restore_probability_to_original(
    probability_canvas: np.ndarray,
    transform: SpatialTransform | dict,
) -> np.ndarray:
    """Remove padding and restore a probability map to original dimensions."""
    if isinstance(transform, dict):
        transform = SpatialTransform(**transform)
    probability = np.asarray(probability_canvas, dtype=np.float32)
    if probability.shape != (
        transform.canvas_height,
        transform.canvas_width,
    ):
        raise ValueError(
            f"Probability shape {probability.shape} does not match "
            f"canvas {(transform.canvas_height, transform.canvas_width)}"
        )

    if transform.policy == "letterbox":
        cropped = probability[
            transform.pad_top :
            transform.pad_top + transform.resized_height,
            transform.pad_left :
            transform.pad_left + transform.resized_width,
        ]
    else:
        cropped = probability

    probability_image = Image.fromarray(cropped, mode="F")
    restored = probability_image.resize(
        (transform.original_width, transform.original_height),
        Image.Resampling.BILINEAR,
    )
    return np.asarray(restored, dtype=np.float32)
