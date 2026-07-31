from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from waterseg.config import load_config
from waterseg.data.manifest import build_manifest
from waterseg.data.split import add_stratification_bins, assert_no_leakage, stratified_split
from waterseg.data.tiling import materialize_tiled_manifest
from waterseg.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover and split the water-body dataset")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = ensure_dir(cfg.paths.output_dir)
    manifest = build_manifest(
        cfg.paths.dataset_root,
        cfg.data.images_dir_name,
        cfg.data.masks_dir_name,
        cfg.data.extensions,
        cfg.data.verify_files,
    )
    manifest = add_stratification_bins(manifest, cfg.data.stratification_bins)
    train, val, test = stratified_split(
        manifest, cfg.data.train_fraction, cfg.data.val_fraction, cfg.data.test_fraction, cfg.train.seed
    )
    assert_no_leakage(train, val, test)
    manifest.to_csv(output / "manifest.csv", index=False)
    for name, split in (("train", train), ("val", val), ("test", test)):
        split.to_csv(output / f"{name}.csv", index=False)
        if cfg.data.materialize_tiles:
            tiled = materialize_tiled_manifest(
                split, output / "tiles" / name, cfg.data.tile_size, cfg.data.tile_overlap
            )
            tiled.to_csv(output / f"{name}_tiles.csv", index=False)
    summary = {
        "total_images": len(manifest),
        "train_images": len(train),
        "val_images": len(val),
        "test_images": len(test),
        "mean_water_fraction": float(manifest.water_fraction.mean()),
        "empty_masks": int((~manifest.has_water).sum()),
        "image_shapes": manifest.groupby(["height", "width"]).size().sort_values(ascending=False).head(20).to_dict(),
    }
    write_json(output / "dataset_summary.json", summary)
    print(pd.DataFrame([summary]).T)


if __name__ == "__main__":
    main()
