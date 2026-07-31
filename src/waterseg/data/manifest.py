from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import cv2
import numpy as np
import pandas as pd

from waterseg.utils import sha256_file

LOGGER = logging.getLogger(__name__)


def read_rgb(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if image.dtype != np.uint8:
        low, high = np.percentile(image.astype(np.float32), [2, 98])
        image = np.clip((image - low) / max(high - low, 1e-6), 0, 1)
        image = (image * 255).astype(np.uint8)
    return image


def read_mask(path: str | Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return (mask > 0).astype(np.uint8)


def _files_by_stem(root: Path, extensions: Iterable[str]) -> Dict[str, Path]:
    allowed = {suffix.lower() for suffix in extensions}
    files: Dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in allowed:
            key = path.stem.lower()
            if key in files:
                raise ValueError(f"Duplicate stem '{key}' in {root}: {files[key]} and {path}")
            files[key] = path
    return files


def build_manifest(
    dataset_root: str | Path,
    images_dir_name: str = "Images",
    masks_dir_name: str = "Masks",
    extensions: Iterable[str] = (".jpg", ".jpeg", ".png", ".tif", ".tiff"),
    verify_files: bool = True,
) -> pd.DataFrame:
    dataset_root = Path(dataset_root)
    image_root = dataset_root / images_dir_name
    mask_root = dataset_root / masks_dir_name
    if not image_root.exists() or not mask_root.exists():
        raise FileNotFoundError(
            f"Expected image and mask directories at {image_root} and {mask_root}. "
            "Adjust paths.images_dir_name and paths.masks_dir_name in the config."
        )

    images = _files_by_stem(image_root, extensions)
    masks = _files_by_stem(mask_root, extensions)
    common = sorted(images.keys() & masks.keys())
    missing_masks = sorted(images.keys() - masks.keys())
    missing_images = sorted(masks.keys() - images.keys())
    if missing_masks or missing_images:
        LOGGER.warning("Unpaired files: %d images without masks; %d masks without images", len(missing_masks), len(missing_images))
    if not common:
        raise RuntimeError("No image-mask pairs were found")

    rows: List[dict] = []
    for key in common:
        image_path, mask_path = images[key], masks[key]
        image = read_rgb(image_path)
        mask = read_mask(mask_path)
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Shape mismatch for {key}: image={image.shape[:2]} mask={mask.shape[:2]}")
        h, w = mask.shape
        row = {
            "image_id": key,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "height": h,
            "width": w,
            "water_fraction": float(mask.mean()),
            "has_water": bool(mask.any()),
            "image_bytes": image_path.stat().st_size,
            "mask_bytes": mask_path.stat().st_size,
        }
        if verify_files:
            row["image_sha256"] = sha256_file(image_path)
            row["mask_sha256"] = sha256_file(mask_path)
        rows.append(row)

    manifest = pd.DataFrame(rows).sort_values("image_id").reset_index(drop=True)
    if verify_files and manifest["image_sha256"].duplicated().any():
        duplicate_count = int(manifest["image_sha256"].duplicated(keep=False).sum())
        LOGGER.warning("Detected %d rows with duplicate image content", duplicate_count)
    return manifest
