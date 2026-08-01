from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TileWindow:
    tile_id: str
    parent_image_id: str
    row_offset: int
    column_offset: int
    height: int
    width: int
    valid_height: int
    valid_width: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positions(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if overlap < 0 or overlap >= tile_size:
        raise ValueError("overlap must satisfy 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    final = length - tile_size
    if positions[-1] != final:
        positions.append(final)
    return positions


def tile_array(
    array: np.ndarray,
    *,
    tile_size: int,
    overlap: int,
    parent_image_id: str,
    pad_value: int | float = 0,
) -> tuple[list[np.ndarray], pd.DataFrame]:
    """Tile a 2D or HWC array with deterministic edge coverage."""
    source = np.asarray(array)
    if source.ndim not in {2, 3}:
        raise ValueError("Expected a 2D mask or HWC image")
    height, width = source.shape[:2]
    rows = _positions(height, tile_size, overlap)
    columns = _positions(width, tile_size, overlap)
    tiles: list[np.ndarray] = []
    records: list[dict[str, Any]] = []

    for row_index, row_offset in enumerate(rows):
        for column_index, column_offset in enumerate(columns):
            valid_height = min(tile_size, height - row_offset)
            valid_width = min(tile_size, width - column_offset)
            tile_shape = (
                (tile_size, tile_size)
                if source.ndim == 2
                else (tile_size, tile_size, source.shape[2])
            )
            tile = np.full(tile_shape, pad_value, dtype=source.dtype)
            source_window = source[
                row_offset : row_offset + valid_height,
                column_offset : column_offset + valid_width,
            ]
            tile[:valid_height, :valid_width] = source_window
            tile_id = (
                f"{parent_image_id}__r{row_index:03d}_c{column_index:03d}"
            )
            window = TileWindow(
                tile_id=tile_id,
                parent_image_id=parent_image_id,
                row_offset=int(row_offset),
                column_offset=int(column_offset),
                height=int(tile_size),
                width=int(tile_size),
                valid_height=int(valid_height),
                valid_width=int(valid_width),
            )
            tiles.append(tile)
            records.append(window.to_dict())

    return tiles, pd.DataFrame(records)


def reconstruct_tiles(
    tiles: Iterable[np.ndarray],
    manifest: pd.DataFrame,
    *,
    output_height: int,
    output_width: int,
) -> np.ndarray:
    """Reconstruct by averaging overlapping values."""
    tiles = list(tiles)
    if len(tiles) != len(manifest):
        raise ValueError(
            f"Tile count {len(tiles)} != manifest rows {len(manifest)}"
        )
    if not tiles:
        raise ValueError("No tiles were supplied")

    first = np.asarray(tiles[0])
    output_shape = (
        (output_height, output_width)
        if first.ndim == 2
        else (output_height, output_width, first.shape[2])
    )
    accumulator = np.zeros(output_shape, dtype=np.float64)
    weights = np.zeros((output_height, output_width), dtype=np.float64)

    for tile, row in zip(tiles, manifest.itertuples(index=False)):
        tile = np.asarray(tile)
        valid = tile[: row.valid_height, : row.valid_width]
        row_slice = slice(row.row_offset, row.row_offset + row.valid_height)
        column_slice = slice(
            row.column_offset,
            row.column_offset + row.valid_width,
        )
        accumulator[row_slice, column_slice] += valid
        weights[row_slice, column_slice] += 1.0

    if np.any(weights == 0):
        raise AssertionError("Reconstruction left uncovered pixels")

    if accumulator.ndim == 3:
        reconstructed = accumulator / weights[..., None]
    else:
        reconstructed = accumulator / weights
    return reconstructed.astype(first.dtype)


def build_multitile_demo(
    source: np.ndarray,
    *,
    tile_size: int,
    overlap: int,
) -> np.ndarray:
    """Repeat a source so the demonstration necessarily spans multiple tiles."""
    source = np.asarray(source)
    height, width = source.shape[:2]
    stride = tile_size - overlap
    minimum = tile_size + stride
    repeat_y = max(1, math.ceil(minimum / height))
    repeat_x = max(1, math.ceil(minimum / width))
    repeats = (
        (repeat_y, repeat_x)
        if source.ndim == 2
        else (repeat_y, repeat_x, 1)
    )
    mosaic = np.tile(source, repeats)
    if mosaic.shape[0] <= tile_size or mosaic.shape[1] <= tile_size:
        raise AssertionError(
            "Demonstration mosaic does not exceed tile size in both dimensions"
        )
    return mosaic


def tile_geotiff(
    source_path: str | Path,
    output_dir: str | Path,
    *,
    tile_size: int,
    overlap: int,
    parent_image_id: str | None = None,
) -> pd.DataFrame:
    """Materialize GeoTIFF tiles while preserving CRS, transform, nodata and bands."""
    import rasterio
    from rasterio.windows import Window
    from rasterio.windows import bounds as window_bounds
    from rasterio.windows import transform as window_transform

    source_path = Path(source_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    parent = parent_image_id or source_path.stem
    records: list[dict[str, Any]] = []

    with rasterio.open(source_path) as source:
        row_positions = _positions(source.height, tile_size, overlap)
        column_positions = _positions(source.width, tile_size, overlap)

        for row_index, row_offset in enumerate(row_positions):
            for column_index, column_offset in enumerate(column_positions):
                valid_height = min(tile_size, source.height - row_offset)
                valid_width = min(tile_size, source.width - column_offset)
                # Read a fixed-size window with boundless padding. Using a
                # smaller window together with ``out_shape`` would resample edge
                # pixels and break exact reconstruction.
                tile_window = Window(
                    column_offset,
                    row_offset,
                    tile_size,
                    tile_size,
                )
                valid_window = Window(
                    column_offset,
                    row_offset,
                    valid_width,
                    valid_height,
                )
                fill_value = (
                    source.nodata if source.nodata is not None else 0
                )
                data = source.read(
                    window=tile_window,
                    boundless=True,
                    fill_value=fill_value,
                )
                tile_transform = window_transform(
                    tile_window,
                    source.transform,
                )
                tile_id = (
                    f"{parent}__r{row_index:03d}_c{column_index:03d}"
                )
                tile_path = output / f"{tile_id}.tif"
                profile = source.profile.copy()
                profile.update(
                    width=tile_size,
                    height=tile_size,
                    transform=tile_transform,
                )
                with rasterio.open(tile_path, "w", **profile) as destination:
                    destination.write(data)

                tile_bounds = window_bounds(
                    valid_window,
                    source.transform,
                )
                records.append(
                    {
                        "tile_id": tile_id,
                        "parent_image_id": parent,
                        "tile_path": str(tile_path),
                        "row_offset": int(row_offset),
                        "column_offset": int(column_offset),
                        "height": int(tile_size),
                        "width": int(tile_size),
                        "valid_height": int(valid_height),
                        "valid_width": int(valid_width),
                        "crs": str(source.crs) if source.crs else "",
                        "transform": tuple(tile_transform),
                        "nodata": source.nodata,
                        "band_count": int(source.count),
                        "source_bounds": tuple(source.bounds),
                        "tile_bounds": tuple(tile_bounds),
                    }
                )
    return pd.DataFrame(records)


def reconstruct_geotiff(
    manifest: pd.DataFrame,
    output_path: str | Path,
    *,
    source_reference_path: str | Path,
) -> Path:
    """Reconstruct GeoTIFF tiles to the source grid by overlap averaging."""
    import rasterio

    source_reference_path = Path(source_reference_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(source_reference_path) as source:
        accumulator = np.zeros(
            (source.count, source.height, source.width),
            dtype=np.float64,
        )
        weights = np.zeros(
            (source.height, source.width),
            dtype=np.float64,
        )
        for row in manifest.itertuples(index=False):
            with rasterio.open(row.tile_path) as tile_source:
                tile = tile_source.read()
            valid = tile[:, : row.valid_height, : row.valid_width]
            row_slice = slice(
                row.row_offset,
                row.row_offset + row.valid_height,
            )
            column_slice = slice(
                row.column_offset,
                row.column_offset + row.valid_width,
            )
            accumulator[:, row_slice, column_slice] += valid
            weights[row_slice, column_slice] += 1.0

        if np.any(weights == 0):
            raise AssertionError("GeoTIFF reconstruction left uncovered pixels")
        reconstructed = accumulator / weights[None, ...]
        reconstructed = reconstructed.astype(source.dtypes[0])
        profile = source.profile.copy()
        with rasterio.open(output_path, "w", **profile) as destination:
            destination.write(reconstructed)

    return output_path
