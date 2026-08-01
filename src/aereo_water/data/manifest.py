from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from aereo_water.utils import (
    json_dump,
    sha256_dataframe,
    sha256_file,
    utc_now_iso,
)


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def _index_by_stem(directory: str | Path) -> dict[str, Path]:
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")
    index: dict[str, Path] = {}
    duplicates: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        key = path.stem.strip().lower()
        if key in index:
            duplicates.setdefault(key, [index[key]]).append(path)
        else:
            index[key] = path
    if duplicates:
        examples = {
            key: [str(item) for item in values]
            for key, values in list(duplicates.items())[:10]
        }
        raise ValueError(
            "Duplicate filename stems were found. Pairing by stem is ambiguous: "
            f"{examples}"
        )
    return index


def discover_pairs(
    image_dir: str | Path,
    mask_dir: str | Path,
) -> pd.DataFrame:
    """Pair images and masks by case-insensitive filename stem.

    Missing images or masks are treated as data-contract violations rather than
    silently dropped.
    """
    images = _index_by_stem(image_dir)
    masks = _index_by_stem(mask_dir)

    image_only = sorted(set(images) - set(masks))
    mask_only = sorted(set(masks) - set(images))
    if image_only or mask_only:
        raise ValueError(
            "Image-mask pairing failed. "
            f"Images without masks: {image_only[:20]}; "
            f"masks without images: {mask_only[:20]}."
        )

    rows = [
        {
            "image_id": key,
            "image_path": str(images[key]),
            "mask_path": str(masks[key]),
            "image_filename": images[key].name,
            "mask_filename": masks[key].name,
        }
        for key in sorted(images)
    ]
    return pd.DataFrame(rows)


def _average_hash(image: Image.Image, hash_size: int = 8) -> str:
    grayscale = image.convert("L").resize(
        (hash_size, hash_size),
        Image.Resampling.BILINEAR,
    )
    values = np.asarray(grayscale, dtype=np.float32)
    bits = values >= values.mean()
    packed = np.packbits(bits.reshape(-1).astype(np.uint8))
    return packed.tobytes().hex()


def _hamming_distance(hex_a: str, hex_b: str) -> int:
    bytes_a = bytes.fromhex(hex_a)
    bytes_b = bytes.fromhex(hex_b)
    if len(bytes_a) != len(bytes_b):
        raise ValueError("Perceptual hashes have different lengths")
    return sum((a ^ b).bit_count() for a, b in zip(bytes_a, bytes_b))


def validate_manifest(
    pairs: pd.DataFrame,
    *,
    compute_sha256: bool = True,
    compute_perceptual_hash: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decode every pair and create an auditable validated manifest."""
    rows: list[dict] = []
    errors: list[dict] = []

    for row in pairs.itertuples(index=False):
        record = row._asdict()
        try:
            with Image.open(row.image_path) as image_raw:
                image = image_raw.convert("RGB")
                width, height = image.size
                image_array = np.asarray(image)
                perceptual_hash = (
                    _average_hash(image) if compute_perceptual_hash else ""
                )

            with Image.open(row.mask_path) as mask_raw:
                mask = mask_raw.convert("L")
                mask_width, mask_height = mask.size
                mask_array = np.asarray(mask)

            if (width, height) != (mask_width, mask_height):
                raise ValueError(
                    "Image and mask dimensions differ: "
                    f"{(width, height)} vs {(mask_width, mask_height)}"
                )

            binary_mask = mask_array > 0
            unique_mask_values = np.unique(mask_array)
            record.update(
                {
                    "width": int(width),
                    "height": int(height),
                    "channels": int(image_array.shape[2]),
                    "image_dtype": str(image_array.dtype),
                    "mask_dtype": str(mask_array.dtype),
                    "mask_unique_values": ",".join(
                        map(str, unique_mask_values.tolist())
                    ),
                    "water_fraction": float(binary_mask.mean()),
                    "water_pixels": int(binary_mask.sum()),
                    "total_pixels": int(binary_mask.size),
                    "image_sha256": (
                        sha256_file(row.image_path) if compute_sha256 else ""
                    ),
                    "mask_sha256": (
                        sha256_file(row.mask_path) if compute_sha256 else ""
                    ),
                    "image_ahash": perceptual_hash,
                }
            )
            rows.append(record)
        except (
            FileNotFoundError,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            errors.append(
                {
                    "image_id": row.image_id,
                    "image_path": row.image_path,
                    "mask_path": row.mask_path,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    validated = pd.DataFrame(rows)
    error_frame = pd.DataFrame(errors)
    return validated, error_frame


def assign_exact_split(
    manifest: pd.DataFrame,
    *,
    train_count: int,
    validation_count: int,
    test_count: int,
    seed: int,
) -> pd.DataFrame:
    """Create exact counts while approximately preserving water coverage."""
    total = train_count + validation_count + test_count
    if len(manifest) != total:
        raise ValueError(
            f"Expected {total} validated pairs, found {len(manifest)}."
        )

    frame = manifest.copy().reset_index(drop=True)
    try:
        frame["_water_bin"] = pd.qcut(
            frame["water_fraction"],
            q=10,
            labels=False,
            duplicates="drop",
        )
        remainder, test = train_test_split(
            frame,
            test_size=test_count,
            random_state=seed,
            stratify=frame["_water_bin"],
        )
        remainder["_water_bin"] = pd.qcut(
            remainder["water_fraction"],
            q=10,
            labels=False,
            duplicates="drop",
        )
        train, validation = train_test_split(
            remainder,
            test_size=validation_count,
            random_state=seed,
            stratify=remainder["_water_bin"],
        )
    except ValueError:
        shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        test = shuffled.iloc[:test_count]
        validation = shuffled.iloc[
            test_count : test_count + validation_count
        ]
        train = shuffled.iloc[test_count + validation_count :]

    train = train.drop(columns=["_water_bin"], errors="ignore").copy()
    validation = validation.drop(columns=["_water_bin"], errors="ignore").copy()
    test = test.drop(columns=["_water_bin"], errors="ignore").copy()
    train["split"] = "train"
    validation["split"] = "validation"
    test["split"] = "test"

    output = pd.concat([train, validation, test], ignore_index=True)
    output = output.sort_values(["split", "image_id"]).reset_index(drop=True)

    counts = output["split"].value_counts().to_dict()
    expected = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    if counts != expected:
        raise AssertionError(f"Unexpected split counts: {counts} != {expected}")
    return output


def recover_historical_split(
    manifest: pd.DataFrame,
    historical_results_csv: str | Path,
) -> pd.DataFrame:
    """Recover the historical split without trusting historical absolute paths."""
    history = pd.read_csv(historical_results_csv)
    required = {"image_id", "split"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            f"Historical split file is missing columns: {sorted(missing)}"
        )

    mapping = (
        history[["image_id", "split"]]
        .assign(
            image_id=lambda x: x["image_id"].astype(str).str.strip().str.lower(),
            split=lambda x: x["split"].astype(str).str.strip().str.lower(),
        )
        .drop_duplicates()
    )
    conflicts = mapping.groupby("image_id")["split"].nunique()
    conflicting_ids = conflicts[conflicts > 1].index.tolist()
    if conflicting_ids:
        raise ValueError(
            "Historical results assign multiple splits to the same image ID: "
            f"{conflicting_ids[:20]}"
        )

    split_map = dict(zip(mapping["image_id"], mapping["split"]))
    output = manifest.copy()
    output["image_id"] = (
        output["image_id"].astype(str).str.strip().str.lower()
    )
    output["split"] = output["image_id"].map(split_map)
    missing_ids = output.loc[output["split"].isna(), "image_id"].tolist()
    if missing_ids:
        raise ValueError(
            "Historical split is incomplete for the current manifest. "
            f"Missing IDs include: {missing_ids[:20]}"
        )

    output["split"] = output["split"].replace({"val": "validation"})
    invalid = sorted(
        set(output["split"]) - {"train", "validation", "test"}
    )
    if invalid:
        raise ValueError(f"Invalid historical split labels: {invalid}")
    return output.sort_values(["split", "image_id"]).reset_index(drop=True)


def assert_split_integrity(manifest: pd.DataFrame) -> None:
    """Fail on exact image or mask duplication across splits."""
    required = {
        "image_id",
        "split",
        "image_sha256",
        "mask_sha256",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    if manifest["image_id"].duplicated().any():
        duplicated = manifest.loc[
            manifest["image_id"].duplicated(keep=False),
            "image_id",
        ].tolist()
        raise ValueError(f"Duplicate image IDs: {duplicated[:20]}")

    for column in ("image_sha256", "mask_sha256"):
        crossing = (
            manifest.groupby(column)["split"].nunique()
            if manifest[column].astype(bool).all()
            else pd.Series(dtype=int)
        )
        bad = crossing[crossing > 1]
        if not bad.empty:
            raise ValueError(
                f"{column} values cross split boundaries: "
                f"{bad.index.tolist()[:10]}"
            )


def near_duplicate_audit(
    manifest: pd.DataFrame,
    *,
    hamming_threshold: int = 4,
    maximum_pairs: int | None = None,
) -> pd.DataFrame:
    """Audit perceptual-hash similarity across every split boundary.

    Hashes are converted to integers once, avoiding slow dataframe indexing in
    the inner loop. ``maximum_pairs`` is an explicit emergency cap; ``None``
    performs the complete cross-split audit. This is evidence only and does not
    silently rewrite the historical split.
    """
    required = {"image_id", "split", "image_ahash"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Missing near-duplicate columns: {sorted(missing)}")
    if hamming_threshold < 0:
        raise ValueError("hamming_threshold must be non-negative")
    if maximum_pairs is not None and maximum_pairs <= 0:
        raise ValueError("maximum_pairs must be positive or None")

    frame = manifest.loc[
        manifest["image_ahash"].fillna("").astype(str).str.len().gt(0),
        ["image_id", "split", "image_ahash"],
    ].copy()
    frame["hash_int"] = frame["image_ahash"].map(
        lambda value: int(str(value), 16)
    )

    split_groups: dict[str, list[tuple[str, int]]] = {}
    for split, group in frame.groupby("split", sort=True):
        split_groups[str(split)] = list(
            zip(
                group["image_id"].astype(str),
                group["hash_int"].astype(object),
            )
        )

    split_names = sorted(split_groups)
    expected_pairs = sum(
        len(split_groups[left]) * len(split_groups[right])
        for left_index, left in enumerate(split_names)
        for right in split_names[left_index + 1 :]
    )
    rows: list[dict] = []
    comparisons = 0
    stopped_early = False

    for left_index, left_split in enumerate(split_names):
        for right_split in split_names[left_index + 1 :]:
            for left_id, left_hash in split_groups[left_split]:
                for right_id, right_hash in split_groups[right_split]:
                    if (
                        maximum_pairs is not None
                        and comparisons >= maximum_pairs
                    ):
                        stopped_early = True
                        break
                    comparisons += 1
                    distance = int(left_hash ^ right_hash).bit_count()
                    if distance <= hamming_threshold:
                        rows.append(
                            {
                                "left_image_id": left_id,
                                "left_split": left_split,
                                "right_image_id": right_id,
                                "right_split": right_split,
                                "hamming_distance": int(distance),
                            }
                        )
                if stopped_early:
                    break
            if stopped_early:
                break
        if stopped_early:
            break

    result = pd.DataFrame(
        rows,
        columns=[
            "left_image_id",
            "left_split",
            "right_image_id",
            "right_split",
            "hamming_distance",
        ],
    )
    result.attrs.update(
        {
            "comparisons_evaluated": int(comparisons),
            "expected_cross_split_pairs": int(expected_pairs),
            "audit_complete": bool(not stopped_early),
            "hamming_threshold": int(hamming_threshold),
        }
    )
    return result


def make_portable_registry(
    manifest: pd.DataFrame,
    *,
    dataset_root: str | Path,
) -> pd.DataFrame:
    root = Path(dataset_root).resolve()
    output = manifest.copy()

    def relative(path: str) -> str:
        candidate = Path(path).resolve()
        try:
            return candidate.relative_to(root).as_posix()
        except ValueError:
            return candidate.name

    output["image_relative_path"] = output["image_path"].map(relative)
    output["mask_relative_path"] = output["mask_path"].map(relative)
    portable_columns = [
        "image_id",
        "split",
        "image_filename",
        "mask_filename",
        "image_relative_path",
        "mask_relative_path",
        "width",
        "height",
        "channels",
        "water_fraction",
        "water_pixels",
        "total_pixels",
        "image_sha256",
        "mask_sha256",
        "image_ahash",
    ]
    return output[portable_columns].copy()


def write_data_registry(
    portable_manifest: pd.DataFrame,
    *,
    output_csv: str | Path,
    output_json: str | Path,
    dataset_name: str,
    dataset_source: str,
    split_seed: int,
    git_commit: str,
    duplicate_policy: str,
    split_before_tiling: bool = True,
) -> tuple[Path, Path, str]:
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    portable_manifest.to_csv(csv_path, index=False)
    registry_hash = sha256_dataframe(portable_manifest)
    counts = portable_manifest["split"].value_counts().to_dict()
    payload = {
        "schema_version": 2,
        "dataset_name": dataset_name,
        "dataset_source": dataset_source,
        "total_pairs": int(len(portable_manifest)),
        "train_pairs": int(counts.get("train", 0)),
        "validation_pairs": int(counts.get("validation", 0)),
        "test_pairs": int(counts.get("test", 0)),
        "split_seed": int(split_seed),
        "pairing_key": "case-insensitive filename stem",
        "duplicate_policy": duplicate_policy,
        "split_before_tiling": bool(split_before_tiling),
        "split_registry_sha256": registry_hash,
        "git_commit": git_commit,
        "created_at_utc": utc_now_iso(),
    }
    json_path = json_dump(payload, output_json)
    return csv_path, json_path, registry_hash
