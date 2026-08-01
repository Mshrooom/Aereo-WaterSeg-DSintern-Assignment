from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from aereo_water.data.manifest import (
    assert_split_integrity,
    assign_exact_split,
    discover_pairs,
    recover_historical_split,
    validate_manifest,
)


def _save_pair(root: Path, stem: str, water: bool = True) -> None:
    images = root / "Images"
    masks = root / "Masks"
    images.mkdir(exist_ok=True)
    masks.mkdir(exist_ok=True)
    Image.fromarray(
        np.full((8, 8, 3), 100, dtype=np.uint8)
    ).save(images / f"{stem}.png")
    mask = np.zeros((8, 8), dtype=np.uint8)
    if water:
        mask[2:6, 2:6] = 255
    Image.fromarray(mask).save(masks / f"{stem}.png")


def test_discover_and_validate_pairs(tmp_path):
    _save_pair(tmp_path, "a")
    pairs = discover_pairs(tmp_path / "Images", tmp_path / "Masks")
    validated, errors = validate_manifest(pairs)
    assert len(validated) == 1
    assert errors.empty
    assert validated.iloc[0]["water_fraction"] == 0.25


def test_missing_mask_fails(tmp_path):
    (tmp_path / "Images").mkdir()
    (tmp_path / "Masks").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "Images" / "a.png")
    with pytest.raises(ValueError, match="Images without masks"):
        discover_pairs(tmp_path / "Images", tmp_path / "Masks")


def test_duplicate_stems_fail(tmp_path):
    (tmp_path / "Images").mkdir()
    (tmp_path / "Masks").mkdir()
    Image.new("RGB", (8, 8)).save(tmp_path / "Images" / "a.png")
    Image.new("RGB", (8, 8)).save(tmp_path / "Images" / "a.jpg")
    Image.new("L", (8, 8)).save(tmp_path / "Masks" / "a.png")
    with pytest.raises(ValueError, match="Duplicate filename stems"):
        discover_pairs(tmp_path / "Images", tmp_path / "Masks")


def test_exact_split_counts():
    frame = pd.DataFrame(
        {
            "image_id": [str(i) for i in range(10)],
            "water_fraction": np.linspace(0, 1, 10),
        }
    )
    split = assign_exact_split(
        frame,
        train_count=6,
        validation_count=2,
        test_count=2,
        seed=42,
    )
    assert split["split"].value_counts().to_dict() == {
        "train": 6,
        "validation": 2,
        "test": 2,
    }


def test_incomplete_historical_mapping_fails(tmp_path):
    frame = pd.DataFrame({"image_id": ["a", "b"]})
    history = tmp_path / "history.csv"
    pd.DataFrame({"image_id": ["a"], "split": ["train"]}).to_csv(
        history, index=False
    )
    with pytest.raises(ValueError, match="incomplete"):
        recover_historical_split(frame, history)


def test_hash_crossing_splits_fails():
    frame = pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "split": ["train", "test"],
            "image_sha256": ["x", "x"],
            "mask_sha256": ["m1", "m2"],
        }
    )
    with pytest.raises(ValueError, match="cross split"):
        assert_split_integrity(frame)


def test_near_duplicate_audit_is_complete_across_splits():
    from aereo_water.data.manifest import near_duplicate_audit

    frame = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "split": ["train", "validation", "test"],
            "image_ahash": [
                "0000000000000000",
                "0000000000000001",
                "ffffffffffffffff",
            ],
        }
    )
    result = near_duplicate_audit(
        frame,
        hamming_threshold=1,
        maximum_pairs=None,
    )
    assert len(result) == 1
    assert result.attrs["comparisons_evaluated"] == 3
    assert result.attrs["expected_cross_split_pairs"] == 3
    assert result.attrs["audit_complete"] is True
