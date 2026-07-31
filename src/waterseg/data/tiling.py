from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np


@dataclass(frozen=True)
class Window:
    y0: int
    x0: int
    y1: int
    x1: int


def generate_windows(height: int, width: int, tile_size: int, overlap: int) -> List[Window]:
    if overlap >= tile_size:
        raise ValueError("overlap must be smaller than tile_size")
    stride = tile_size - overlap
    ys = list(range(0, max(height - tile_size, 0) + 1, stride))
    xs = list(range(0, max(width - tile_size, 0) + 1, stride))
    if not ys or ys[-1] != max(height - tile_size, 0):
        ys.append(max(height - tile_size, 0))
    if not xs or xs[-1] != max(width - tile_size, 0):
        xs.append(max(width - tile_size, 0))
    return [Window(y, x, min(y + tile_size, height), min(x + tile_size, width)) for y in ys for x in xs]


def extract_tile(array: np.ndarray, window: Window, tile_size: int, pad_value: int = 0) -> np.ndarray:
    tile = array[window.y0 : window.y1, window.x0 : window.x1]
    pad_h, pad_w = tile_size - tile.shape[0], tile_size - tile.shape[1]
    if pad_h or pad_w:
        pad_spec = [(0, pad_h), (0, pad_w)] + ([(0, 0)] if tile.ndim == 3 else [])
        tile = np.pad(tile, pad_spec, mode="constant", constant_values=pad_value)
    return tile


def blending_weights(tile_size: int) -> np.ndarray:
    one_dim = np.hanning(tile_size)
    weight = np.outer(one_dim, one_dim).astype(np.float32)
    return np.maximum(weight, 1e-3)


def stitch_probability_tiles(
    tiles: Iterable[Tuple[Window, np.ndarray]],
    output_shape: Tuple[int, int],
    tile_size: int,
) -> np.ndarray:
    height, width = output_shape
    accumulator = np.zeros((height, width), dtype=np.float32)
    normalizer = np.zeros((height, width), dtype=np.float32)
    weight = blending_weights(tile_size)
    for window, tile_probability in tiles:
        valid_h, valid_w = window.y1 - window.y0, window.x1 - window.x0
        local_weight = weight[:valid_h, :valid_w]
        accumulator[window.y0 : window.y1, window.x0 : window.x1] += tile_probability[:valid_h, :valid_w] * local_weight
        normalizer[window.y0 : window.y1, window.x0 : window.x1] += local_weight
    return accumulator / np.maximum(normalizer, 1e-6)


def materialize_tiled_manifest(
    manifest,
    output_root,
    tile_size: int,
    overlap: int,
):
    """Materialize only images larger than tile_size; keep smaller pairs unchanged.

    Splitting must happen before calling this function so tiles from one parent image
    can never cross train/validation/test boundaries.
    """
    from pathlib import Path

    import cv2
    import pandas as pd

    from waterseg.data.manifest import read_mask, read_rgb

    output_root = Path(output_root)
    image_root = output_root / "images"
    mask_root = output_root / "masks"
    image_root.mkdir(parents=True, exist_ok=True)
    mask_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in manifest.itertuples(index=False):
        image = read_rgb(row.image_path)
        mask = read_mask(row.mask_path)
        if image.shape[0] <= tile_size and image.shape[1] <= tile_size:
            payload = row._asdict()
            payload["parent_image_id"] = row.image_id
            payload.update({"tile_y0": 0, "tile_x0": 0, "tile_y1": image.shape[0], "tile_x1": image.shape[1]})
            rows.append(payload)
            continue
        for tile_index, window in enumerate(generate_windows(image.shape[0], image.shape[1], tile_size, overlap)):
            image_tile = image[window.y0 : window.y1, window.x0 : window.x1]
            mask_tile = mask[window.y0 : window.y1, window.x0 : window.x1]
            tile_id = f"{row.image_id}__y{window.y0}_x{window.x0}"
            image_path = image_root / f"{tile_id}.png"
            mask_path = mask_root / f"{tile_id}.png"
            cv2.imwrite(str(image_path), cv2.cvtColor(image_tile, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(mask_path), mask_tile.astype(np.uint8) * 255)
            payload = row._asdict()
            payload.update(
                {
                    "image_id": tile_id,
                    "parent_image_id": row.image_id,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "height": image_tile.shape[0],
                    "width": image_tile.shape[1],
                    "water_fraction": float(mask_tile.mean()),
                    "has_water": bool(mask_tile.any()),
                    "tile_y0": window.y0,
                    "tile_x0": window.x0,
                    "tile_y1": window.y1,
                    "tile_x1": window.x1,
                }
            )
            rows.append(payload)
    return pd.DataFrame(rows)
