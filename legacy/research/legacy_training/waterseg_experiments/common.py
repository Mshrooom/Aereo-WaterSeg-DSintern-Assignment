from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd

from waterseg.metrics import binary_confusion, boundary_metrics, confusion_metrics, surface_distances


def all_split_manifest(output_dir: str | Path) -> pd.DataFrame:
    output_dir = Path(output_dir)
    frames = []
    for split, filename in (("train", "train.csv"), ("validation", "val.csv"), ("test", "test.csv")):
        frame = pd.read_csv(output_dir / filename)
        frame["split"] = split
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if combined.image_id.duplicated().any():
        raise ValueError("An image appears in more than one split")
    combined.to_csv(output_dir / "all_2841_manifest.csv", index=False)
    return combined


def save_binary_mask(path: str | Path, mask: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mask.astype(np.uint8) * 255):
        raise IOError(f"Could not write prediction mask: {path}")
    return str(path)


def save_probability_u16(path: str | Path, probability: np.ndarray) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = np.round(np.clip(probability, 0, 1) * 65535).astype(np.uint16)
    if not cv2.imwrite(str(path), encoded):
        raise IOError(f"Could not write probability map: {path}")
    return str(path)


def read_probability_u16(path: str | Path) -> np.ndarray:
    encoded = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if encoded is None:
        raise ValueError(f"Could not read probability map: {path}")
    return encoded.astype(np.float32) / 65535.0


def per_image_result(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
    *,
    image_id: str,
    split: str,
    experiment: str,
    prompt_mode: str,
    image_path: str,
    mask_path: str,
    prediction_path: str,
    probability_path: str = "",
    latency_ms: float | None = None,
    status: str = "ok",
    extra: dict | None = None,
) -> dict:
    prediction = probability >= threshold
    tp, fp, fn, tn = binary_confusion(prediction, target)
    row = {
        "image_id": image_id,
        "split": split,
        "experiment": experiment,
        "prompt_mode": prompt_mode,
        "status": status,
        "image_path": image_path,
        "mask_path": mask_path,
        "prediction_path": prediction_path,
        "probability_path": probability_path,
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        **confusion_metrics(tp, fp, fn, tn),
        **boundary_metrics(prediction, target),
        **surface_distances(prediction, target),
        "water_fraction": float(target.mean()),
        "predicted_water_fraction": float(prediction.mean()),
    }
    if latency_ms is not None:
        row["latency_ms"] = float(latency_ms)
    if extra:
        row.update(extra)
    return row


def append_rows(csv_path: str | Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


def completed_keys(csv_path: str | Path) -> set[tuple[str, str]]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return set()
    frame = pd.read_csv(csv_path, usecols=["image_id", "prompt_mode"])
    return set(zip(frame.image_id.astype(str), frame.prompt_mode.astype(str)))


def summarise_results(result_files: Iterable[str | Path], output_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [pd.read_csv(path) for path in result_files if Path(path).exists()]
    if not frames:
        raise RuntimeError("No experiment CSVs were found")
    combined = pd.concat(frames, ignore_index=True)
    output_dir = Path(output_dir)
    combined.to_csv(output_dir / "all_experiments_all_images.csv", index=False)

    metric_columns = [
        "iou", "dice", "precision", "recall", "specificity", "pixel_accuracy",
        "balanced_accuracy", "mcc", "cohen_kappa", "boundary_f1", "boundary_iou",
        "hd95", "assd", "water_fraction", "predicted_water_fraction", "latency_ms",
    ]
    present = [column for column in metric_columns if column in combined.columns]
    macro = (
        combined.groupby(["experiment", "prompt_mode", "split"], dropna=False)[present]
        .mean(numeric_only=True)
        .reset_index()
    )
    macro["images"] = (
        combined.groupby(["experiment", "prompt_mode", "split"], dropna=False)
        .size().to_numpy()
    )
    macro.to_csv(output_dir / "summary_macro_by_split.csv", index=False)

    global_rows = []
    for keys, group in combined.groupby(["experiment", "prompt_mode", "split"], dropna=False):
        tp, fp, fn, tn = (int(group[column].sum()) for column in ("tp", "fp", "fn", "tn"))
        global_rows.append({
            "experiment": keys[0], "prompt_mode": keys[1], "split": keys[2],
            "images": len(group), **{f"global_{k}": v for k, v in confusion_metrics(tp, fp, fn, tn).items()},
        })
    global_table = pd.DataFrame(global_rows)
    global_table.to_csv(output_dir / "summary_global_by_split.csv", index=False)
    macro[macro.split == "test"].to_csv(output_dir / "summary_test_only.csv", index=False)
    return macro, global_table
