from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import cv2
import numpy as np
import pandas as pd

EPS = 1e-8
TRAPEZOID = (
    np.trapezoid
    if hasattr(np, "trapezoid")
    else np.trapz
)


def _safe_div(numerator: float, denominator: float, empty_value: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty_value)


def confusion_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    total = tp + fp + fn + tn
    iou = _safe_div(tp, tp + fp + fn, empty_value=1.0)
    dice = _safe_div(2 * tp, 2 * tp + fp + fn, empty_value=1.0)
    precision = _safe_div(tp, tp + fp, empty_value=1.0 if fn == 0 else 0.0)
    recall = _safe_div(tp, tp + fn, empty_value=1.0)
    specificity = _safe_div(tn, tn + fp, empty_value=1.0)
    accuracy = _safe_div(tp + tn, total, empty_value=1.0)
    balanced_accuracy = 0.5 * (recall + specificity)
    # Convert each term to float before multiplication.
    # Pixel-count products can exceed NumPy's fixed-width integer range.
    mcc_denominator = np.sqrt(
        float(tp + fp)
        * float(tp + fn)
        * float(tn + fp)
        * float(tn + fn)
    )
    mcc = _safe_div(tp * tn - fp * fn, float(mcc_denominator), empty_value=0.0)
    observed = accuracy
    expected = _safe_div((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn), total * total, empty_value=0.0)
    kappa = _safe_div(observed - expected, 1.0 - expected, empty_value=0.0)
    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pixel_accuracy": accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "cohen_kappa": kappa,
    }


def binary_confusion(prediction: np.ndarray, target: np.ndarray) -> tuple[int, int, int, int]:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    return tp, fp, fn, tn


def mask_boundary(mask: np.ndarray, thickness: int = 1) -> np.ndarray:
    mask = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(mask, kernel, iterations=thickness)
    return (mask - eroded).astype(bool)


def boundary_metrics(prediction: np.ndarray, target: np.ndarray, tolerance: int = 2) -> Dict[str, float]:
    pred_boundary = mask_boundary(prediction)
    target_boundary = mask_boundary(target)
    if not pred_boundary.any() and not target_boundary.any():
        return {"boundary_precision": 1.0, "boundary_recall": 1.0, "boundary_f1": 1.0, "boundary_iou": 1.0}
    kernel_size = 2 * tolerance + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_pred = cv2.dilate(pred_boundary.astype(np.uint8), kernel).astype(bool)
    dilated_target = cv2.dilate(target_boundary.astype(np.uint8), kernel).astype(bool)
    matched_pred = np.logical_and(pred_boundary, dilated_target).sum()
    matched_target = np.logical_and(target_boundary, dilated_pred).sum()
    precision = _safe_div(matched_pred, pred_boundary.sum(), empty_value=0.0)
    recall = _safe_div(matched_target, target_boundary.sum(), empty_value=0.0)
    f1 = _safe_div(2 * precision * recall, precision + recall, empty_value=0.0)
    intersection = np.logical_and(pred_boundary, target_boundary).sum()
    union = np.logical_or(pred_boundary, target_boundary).sum()
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_iou": _safe_div(intersection, union, empty_value=1.0),
    }


def surface_distances(prediction: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    pred_boundary = mask_boundary(prediction)
    target_boundary = mask_boundary(target)
    diagonal = float(np.hypot(*prediction.shape))
    if not pred_boundary.any() and not target_boundary.any():
        return {"hd95": 0.0, "assd": 0.0}
    if not pred_boundary.any() or not target_boundary.any():
        return {"hd95": diagonal, "assd": diagonal}
    distance_to_target = cv2.distanceTransform((~target_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_pred = cv2.distanceTransform((~pred_boundary).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([distance_to_target[pred_boundary], distance_to_pred[target_boundary]])
    return {"hd95": float(np.percentile(distances, 95)), "assd": float(distances.mean())}


@dataclass
class CalibrationMeter:
    bins: int = 20
    counts: np.ndarray = field(init=False)
    probability_sums: np.ndarray = field(init=False)
    target_sums: np.ndarray = field(init=False)
    brier_sum: float = 0.0
    pixels: int = 0
    pos_hist: np.ndarray = field(init=False)
    neg_hist: np.ndarray = field(init=False)
    auc_bins: int = 512

    def __post_init__(self) -> None:
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.probability_sums = np.zeros(self.bins, dtype=np.float64)
        self.target_sums = np.zeros(self.bins, dtype=np.float64)
        self.pos_hist = np.zeros(self.auc_bins, dtype=np.int64)
        self.neg_hist = np.zeros(self.auc_bins, dtype=np.int64)

    def update(self, probabilities: np.ndarray, targets: np.ndarray) -> None:
        probabilities = np.clip(probabilities.astype(np.float64).ravel(), 0.0, 1.0)
        targets = targets.astype(np.uint8).ravel()
        indices = np.minimum((probabilities * self.bins).astype(int), self.bins - 1)
        self.counts += np.bincount(indices, minlength=self.bins)
        self.probability_sums += np.bincount(indices, weights=probabilities, minlength=self.bins)
        self.target_sums += np.bincount(indices, weights=targets, minlength=self.bins)
        self.brier_sum += float(np.square(probabilities - targets).sum())
        self.pixels += len(targets)
        auc_indices = np.minimum((probabilities * self.auc_bins).astype(int), self.auc_bins - 1)
        self.pos_hist += np.bincount(auc_indices[targets == 1], minlength=self.auc_bins)
        self.neg_hist += np.bincount(auc_indices[targets == 0], minlength=self.auc_bins)

    def compute(self) -> Dict[str, float]:
        nonempty = self.counts > 0
        mean_p = np.divide(self.probability_sums, self.counts, out=np.zeros_like(self.probability_sums), where=nonempty)
        mean_y = np.divide(self.target_sums, self.counts, out=np.zeros_like(self.target_sums), where=nonempty)
        ece = float((np.abs(mean_p - mean_y) * self.counts).sum() / max(self.pixels, 1))
        brier = float(self.brier_sum / max(self.pixels, 1))

        tp = np.cumsum(self.pos_hist[::-1]).astype(np.float64)
        fp = np.cumsum(self.neg_hist[::-1]).astype(np.float64)
        positives, negatives = max(tp[-1], 1.0), max(fp[-1], 1.0)
        tpr, fpr = tp / positives, fp / negatives
        auroc = float(TRAPEZOID(tpr, fpr))
        precision = tp / np.maximum(tp + fp, 1.0)
        recall = tpr
        auprc = float(TRAPEZOID(precision, recall))
        return {"ece": ece, "brier_score": brier, "auroc_hist": auroc, "auprc_hist": auprc}


@dataclass
class BinarySegmentationMeter:
    threshold: float = 0.5
    boundary_tolerance: int = 2
    compute_surface: bool = False
    rows: List[dict] = field(default_factory=list)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    calibration: CalibrationMeter = field(default_factory=CalibrationMeter)

    def update(self, probabilities: np.ndarray, target: np.ndarray, image_id: str, latency_ms: float | None = None) -> None:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        target = np.asarray(target, dtype=np.uint8)
        prediction = probabilities >= self.threshold
        tp, fp, fn, tn = binary_confusion(prediction, target)
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.tn += tn
        row = {"image_id": image_id, **confusion_metrics(tp, fp, fn, tn)}
        row.update(boundary_metrics(prediction, target, self.boundary_tolerance))
        if self.compute_surface:
            row.update(surface_distances(prediction, target))
        row["water_fraction"] = float(target.mean())
        row["predicted_water_fraction"] = float(prediction.mean())
        if latency_ms is not None:
            row["latency_ms"] = float(latency_ms)
        self.rows.append(row)
        self.calibration.update(probabilities, target)

    def compute(self) -> Dict[str, float]:
        global_metrics = {f"global_{key}": value for key, value in confusion_metrics(self.tp, self.fp, self.fn, self.tn).items()}
        dataframe = pd.DataFrame(self.rows)
        mean_columns = [column for column in dataframe.columns if column != "image_id"]
        mean_metrics = {f"mean_{column}": float(dataframe[column].mean()) for column in mean_columns}
        output = {**global_metrics, **mean_metrics, **self.calibration.compute(), "images": len(self.rows)}
        # Primary aliases are macro/per-image values, useful for model selection.
        for name in ("iou", "dice", "precision", "recall", "specificity", "pixel_accuracy", "balanced_accuracy", "mcc", "cohen_kappa"):
            output[name] = output.get(f"mean_{name}", output.get(f"global_{name}", 0.0))
        return output

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


@dataclass
class ThresholdSweep:
    thresholds: Sequence[float]
    confusion: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.confusion = np.zeros((len(self.thresholds), 4), dtype=np.int64)

    def update(self, probabilities: np.ndarray, target: np.ndarray) -> None:
        target_bool = target.astype(bool)
        for index, threshold in enumerate(self.thresholds):
            prediction = probabilities >= threshold
            self.confusion[index] += np.array(binary_confusion(prediction, target_bool), dtype=np.int64)

    def table(self) -> pd.DataFrame:
        rows = []
        for threshold, (tp, fp, fn, tn) in zip(self.thresholds, self.confusion):
            rows.append({"threshold": float(threshold), **confusion_metrics(int(tp), int(fp), int(fn), int(tn))})
        return pd.DataFrame(rows)

    def best(self, metric: str = "iou") -> Dict[str, float]:
        table = self.table()
        best_row = table.loc[table[metric].idxmax()]
        return {key: float(value) for key, value in best_row.to_dict().items()}


def bootstrap_confidence_interval(
    values: Iterable[float], seed: int = 42, iterations: int = 2000, confidence: float = 0.95
) -> Dict[str, float]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return {"mean": float("nan"), "lower": float("nan"), "upper": float("nan")}
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        means[index] = rng.choice(array, size=len(array), replace=True).mean()
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean": float(array.mean()),
        "lower": float(np.quantile(means, alpha)),
        "upper": float(np.quantile(means, 1.0 - alpha)),
    }
