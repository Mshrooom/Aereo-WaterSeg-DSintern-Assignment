from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from aereo_water.data.tiling import (
    build_multitile_demo,
    reconstruct_tiles,
    tile_array,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tile-size", type=int, default=1024)
    parser.add_argument("--overlap", type=int, default=128)
    args = parser.parse_args()

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with Image.open(args.image) as raw:
        source = np.asarray(raw.convert("RGB"))
    mosaic = build_multitile_demo(
        source,
        tile_size=args.tile_size,
        overlap=args.overlap,
    )
    tiles, manifest = tile_array(
        mosaic,
        tile_size=args.tile_size,
        overlap=args.overlap,
        parent_image_id=Path(args.image).stem,
    )
    reconstructed = reconstruct_tiles(
        tiles,
        manifest,
        output_height=mosaic.shape[0],
        output_width=mosaic.shape[1],
    )
    assert len(tiles) > 1
    assert manifest["row_offset"].nunique() > 1
    assert manifest["column_offset"].nunique() > 1
    assert np.array_equal(reconstructed, mosaic)
    manifest.to_csv(output / "tiling_manifest.csv", index=False)
    Image.fromarray(mosaic).save(output / "source_mosaic.png")
    Image.fromarray(reconstructed).save(
        output / "reconstructed_mosaic.png"
    )
    print(f"Created {len(tiles)} overlapping tiles.")
    print("Exact reconstruction passed.")


if __name__ == "__main__":
    main()
