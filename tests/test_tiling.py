import numpy as np

from waterseg.data.tiling import extract_tile, generate_windows, stitch_probability_tiles


def test_tiling_and_stitching_preserve_constant_field():
    shape = (150, 170)
    windows = generate_windows(*shape, tile_size=64, overlap=16)
    tiles = [(window, np.ones((64, 64), dtype=np.float32)) for window in windows]
    stitched = stitch_probability_tiles(tiles, shape, tile_size=64)
    assert stitched.shape == shape
    assert np.allclose(stitched, 1.0)


def test_materialized_tiles_keep_parent_split(tmp_path):
    import cv2
    import pandas as pd

    from waterseg.data.tiling import materialize_tiled_manifest

    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[20:80, 30:90] = 150
    mask = np.zeros((100, 120), dtype=np.uint8)
    mask[25:75, 35:85] = 1
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(mask_path), mask * 255)
    manifest = pd.DataFrame([
        {
            "image_id": "parent",
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "height": 100,
            "width": 120,
            "water_fraction": float(mask.mean()),
            "has_water": True,
            "split": "train",
        }
    ])
    tiled = materialize_tiled_manifest(manifest, tmp_path / "tiles", tile_size=64, overlap=16)
    assert len(tiled) > 1
    assert set(tiled.parent_image_id) == {"parent"}
    assert set(tiled.split) == {"train"}
