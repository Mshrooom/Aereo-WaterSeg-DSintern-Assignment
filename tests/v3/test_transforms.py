import random

import numpy as np
from PIL import Image

from aereo_water.data.transforms import (
    apply_synchronized_augmentation,
    resize_pair,
    restore_probability_to_original,
)


def test_letterbox_preserves_aspect_ratio_and_restores_shape():
    image = Image.fromarray(np.zeros((100, 200, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((100, 200), dtype=np.uint8))
    prepared, prepared_mask, transform = resize_pair(
        image,
        mask,
        size=512,
        policy="letterbox",
    )
    assert prepared.size == (512, 512)
    assert prepared_mask.size == (512, 512)
    assert transform.resized_width == 512
    assert transform.resized_height == 256
    probability = np.zeros((512, 512), dtype=np.float32)
    restored = restore_probability_to_original(probability, transform)
    assert restored.shape == (100, 200)


def test_stretch_restores_shape():
    image = Image.new("RGB", (120, 80))
    mask = Image.new("L", (120, 80))
    _, _, transform = resize_pair(
        image, mask, size=64, policy="stretch"
    )
    restored = restore_probability_to_original(
        np.zeros((64, 64), dtype=np.float32),
        transform,
    )
    assert restored.shape == (80, 120)


def test_spatial_augmentation_keeps_image_mask_alignment():
    array = np.zeros((16, 16), dtype=np.uint8)
    array[2:6, 3:8] = 255
    image = Image.fromarray(np.stack([array] * 3, axis=-1))
    mask = Image.fromarray((array > 0).astype(np.uint8))
    image_aug, mask_aug = apply_synchronized_augmentation(
        image,
        mask,
        profile="moderate",
        rng=random.Random(3),
    )
    image_binary = np.asarray(image_aug.convert("L")) > 20
    mask_binary = np.asarray(mask_aug) > 0
    assert np.array_equal(image_binary, mask_binary)


def test_mask_remains_binary_after_resize():
    image = Image.new("RGB", (31, 17))
    mask_array = np.zeros((17, 31), dtype=np.uint8)
    mask_array[:, 10:] = 1
    mask = Image.fromarray(mask_array)
    _, resized_mask, _ = resize_pair(
        image, mask, size=64, policy="letterbox"
    )
    assert set(np.unique(np.asarray(resized_mask))).issubset({0, 1})


def test_letterbox_can_mark_padding_as_ignore_label():
    image = Image.new("RGB", (200, 100))
    mask = Image.new("L", (200, 100), color=1)
    _, prepared_mask, transform = resize_pair(
        image,
        mask,
        size=64,
        policy="letterbox",
        mask_pad_value=255,
    )
    values = set(np.unique(np.asarray(prepared_mask)).tolist())
    assert values == {1, 255}
    assert transform.pad_top > 0
