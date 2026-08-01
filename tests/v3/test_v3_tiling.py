import numpy as np
import rasterio
from rasterio.transform import from_origin

from aereo_water.data.tiling import (
    build_multitile_demo,
    reconstruct_geotiff,
    reconstruct_tiles,
    tile_array,
    tile_geotiff,
)


def test_multitile_overlap_round_trip():
    source = np.arange(32 * 48 * 3, dtype=np.uint16).reshape(32, 48, 3)
    mosaic = build_multitile_demo(source, tile_size=64, overlap=16)
    tiles, manifest = tile_array(
        mosaic,
        tile_size=64,
        overlap=16,
        parent_image_id="sample",
    )
    assert len(tiles) > 1
    assert manifest["row_offset"].nunique() > 1
    assert manifest["column_offset"].nunique() > 1
    restored = reconstruct_tiles(
        tiles,
        manifest,
        output_height=mosaic.shape[0],
        output_width=mosaic.shape[1],
    )
    assert np.array_equal(restored, mosaic)


def test_mask_tiling_round_trip():
    source = np.zeros((130, 170), dtype=np.uint8)
    source[10:90, 20:110] = 1
    tiles, manifest = tile_array(
        source,
        tile_size=64,
        overlap=16,
        parent_image_id="mask",
    )
    restored = reconstruct_tiles(
        tiles,
        manifest,
        output_height=130,
        output_width=170,
    )
    assert np.array_equal(restored, source)


def test_geotiff_metadata_preserved(tmp_path):
    source_path = tmp_path / "source.tif"
    transform = from_origin(100.0, 200.0, 10.0, 10.0)
    data = np.arange(128 * 160, dtype=np.uint16).reshape(128, 160)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        height=128,
        width=160,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as destination:
        destination.write(data, 1)

    manifest = tile_geotiff(
        source_path,
        tmp_path / "tiles",
        tile_size=64,
        overlap=16,
    )
    assert len(manifest) > 1
    output = reconstruct_geotiff(
        manifest,
        tmp_path / "reconstructed.tif",
        source_reference_path=source_path,
    )
    with rasterio.open(source_path) as source, rasterio.open(output) as result:
        assert source.crs == result.crs
        assert source.transform == result.transform
        assert source.bounds == result.bounds
        assert np.array_equal(source.read(1), result.read(1))


def test_small_geotiff_is_padded_without_resampling(tmp_path):
    source_path = tmp_path / "small.tif"
    transform = from_origin(10.0, 20.0, 1.0, 1.0)
    data = np.arange(20 * 30, dtype=np.uint16).reshape(20, 30)
    with rasterio.open(
        source_path,
        "w",
        driver="GTiff",
        height=20,
        width=30,
        count=1,
        dtype=data.dtype,
        crs="EPSG:3857",
        transform=transform,
        nodata=0,
    ) as destination:
        destination.write(data, 1)

    manifest = tile_geotiff(
        source_path,
        tmp_path / "small_tiles",
        tile_size=64,
        overlap=16,
    )
    assert len(manifest) == 1
    output = reconstruct_geotiff(
        manifest,
        tmp_path / "small_reconstructed.tif",
        source_reference_path=source_path,
    )
    with rasterio.open(output) as result:
        assert result.width == 30
        assert result.height == 20
        assert result.transform == transform
        assert np.array_equal(result.read(1), data)
