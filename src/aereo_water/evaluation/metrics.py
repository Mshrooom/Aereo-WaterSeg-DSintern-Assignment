from __future__ import annotations

from typing import Any

import numpy as np


EMPTY_POLICIES = {
    "perfect_if_both_empty",
    "zero_if_undefined",
}


def confusion_counts(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, int]:
    pred = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"Shape mismatch: {pred.shape} vs {truth.shape}")
    return {
        "tp": int(np.logical_and(pred, truth).sum()),
        "fp": int(np.logical_and(pred, ~truth).sum()),
        "fn": int(np.logical_and(~pred, truth).sum()),
        "tn": int(np.logical_and(~pred, ~truth).sum()),
    }


def _ratio(
    numerator: float,
    denominator: float,
    *,
    both_empty: bool,
    empty_policy: str,
) -> float:
    if denominator:
        return float(numerator / denominator)
    if empty_policy not in EMPTY_POLICIES:
        raise ValueError(f"Unknown empty-mask policy: {empty_policy}")
    if empty_policy == "perfect_if_both_empty" and both_empty:
        return 1.0
    return 0.0


def segmentation_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    include_boundary_metrics: bool = True,
    boundary_tolerance: int = 2,
    empty_policy: str = "perfect_if_both_empty",
) -> dict[str, Any]:
    pred = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    counts = confusion_counts(pred, truth)
    tp, fp, fn, tn = (counts[key] for key in ("tp", "fp", "fn", "tn"))
    both_empty = not pred.any() and not truth.any()

    iou = _ratio(
        tp,
        tp + fp + fn,
        both_empty=both_empty,
        empty_policy=empty_policy,
    )
    dice = _ratio(
        2 * tp,
        2 * tp + fp + fn,
        both_empty=both_empty,
        empty_policy=empty_policy,
    )
    precision = _ratio(
        tp,
        tp + fp,
        both_empty=both_empty,
        empty_policy=empty_policy,
    )
    recall = _ratio(
        tp,
        tp + fn,
        both_empty=both_empty,
        empty_policy=empty_policy,
    )
    specificity = float(tn / (tn + fp)) if (tn + fp) else 1.0
    total = tp + fp + fn + tn
    pixel_accuracy = float((tp + tn) / total) if total else 1.0
    balanced_accuracy = float((recall + specificity) / 2.0)

    denominator = np.sqrt(
        (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = float(((tp * tn) - (fp * fn)) / denominator) if denominator else 0.0

    expected = (
        ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn))
        / (total * total)
        if total
        else 1.0
    )
    cohen_kappa = (
        float((pixel_accuracy - expected) / (1.0 - expected))
        if expected != 1.0
        else 1.0
    )

    result: dict[str, Any] = {
        **counts,
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "pixel_accuracy": pixel_accuracy,
        "balanced_accuracy": balanced_accuracy,
        "mcc": mcc,
        "cohen_kappa": cohen_kappa,
        "water_fraction": float(truth.mean()),
        "predicted_water_fraction": float(pred.mean()),
        "water_fraction_error": float(pred.mean() - truth.mean()),
        "target_has_water": bool(truth.any()),
        "prediction_has_water": bool(pred.any()),
        "empty_mask_correct": bool(both_empty),
        "empty_mask_false_positive": bool(not truth.any() and pred.any()),
        "empty_policy": empty_policy,
    }
    if include_boundary_metrics:
        result.update(
            boundary_metrics(
                pred,
                truth,
                tolerance=boundary_tolerance,
            )
        )
    return result


def _boundary(mask: np.ndarray) -> np.ndarray:
    from scipy.ndimage import binary_erosion

    binary = np.asarray(mask).astype(bool)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    return np.logical_xor(binary, binary_erosion(binary))


def boundary_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    tolerance: int = 2,
) -> dict[str, float]:
    from scipy.ndimage import binary_dilation, distance_transform_edt

    pred_boundary = _boundary(prediction)
    target_boundary = _boundary(target)

    if not pred_boundary.any() and not target_boundary.any():
        return {
            "boundary_precision": 1.0,
            "boundary_recall": 1.0,
            "boundary_f1": 1.0,
            "boundary_iou": 1.0,
            "hd95": 0.0,
            "assd": 0.0,
        }
    if not pred_boundary.any() or not target_boundary.any():
        return {
            "boundary_precision": 0.0,
            "boundary_recall": 0.0,
            "boundary_f1": 0.0,
            "boundary_iou": 0.0,
            "hd95": float("nan"),
            "assd": float("nan"),
        }

    structure = np.ones((2 * tolerance + 1, 2 * tolerance + 1), dtype=bool)
    pred_dilated = binary_dilation(pred_boundary, structure=structure)
    target_dilated = binary_dilation(target_boundary, structure=structure)
    matched_pred = np.logical_and(pred_boundary, target_dilated).sum()
    matched_target = np.logical_and(target_boundary, pred_dilated).sum()

    precision = float(matched_pred / pred_boundary.sum())
    recall = float(matched_target / target_boundary.sum())
    f1 = (
        float(2 * precision * recall / (precision + recall))
        if precision + recall
        else 0.0
    )
    intersection = np.logical_and(pred_boundary, target_boundary).sum()
    union = np.logical_or(pred_boundary, target_boundary).sum()
    boundary_iou = float(intersection / union) if union else 1.0

    distance_to_target = distance_transform_edt(~target_boundary)
    distance_to_pred = distance_transform_edt(~pred_boundary)
    distances = np.concatenate(
        [
            distance_to_target[pred_boundary],
            distance_to_pred[target_boundary],
        ]
    )
    return {
        "boundary_precision": precision,
        "boundary_recall": recall,
        "boundary_f1": f1,
        "boundary_iou": boundary_iou,
        "hd95": float(np.percentile(distances, 95)),
        "assd": float(distances.mean()),
    }


def reliability_table(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
):
    """Return auditable reliability-bin statistics for a binary predictor."""
    import pandas as pd

    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.shape != target.shape:
        raise ValueError("Probability and target arrays must have equal size")
    if bins <= 1:
        raise ValueError("bins must be greater than one")
    probability = np.clip(probability, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for index in range(bins):
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if index == bins - 1:
            mask = (probability >= lower) & (probability <= upper)
        else:
            mask = (probability >= lower) & (probability < upper)
        count = int(mask.sum())
        rows.append(
            {
                "bin_index": int(index),
                "lower_bound": lower,
                "upper_bound": upper,
                "pixel_count": count,
                "pixel_fraction": float(mask.mean()),
                "mean_confidence": (
                    float(probability[mask].mean()) if count else float("nan")
                ),
                "empirical_water_frequency": (
                    float(target[mask].mean()) if count else float("nan")
                ),
                "absolute_calibration_gap": (
                    float(
                        abs(
                            probability[mask].mean()
                            - target[mask].mean()
                        )
                    )
                    if count
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def probability_calibration_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float]:
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(targets, dtype=np.float64).reshape(-1)
    if probability.shape != target.shape:
        raise ValueError("Probability and target arrays must have equal size")
    probability = np.clip(probability, 0.0, 1.0)
    brier = float(np.mean((probability - target) ** 2))
    entropy = -(
        probability * np.log(np.clip(probability, 1e-12, 1.0))
        + (1.0 - probability)
        * np.log(np.clip(1.0 - probability, 1e-12, 1.0))
    )
    table = reliability_table(probability, target, bins=bins)
    ece = float(
        (
            table["pixel_fraction"]
            * table["absolute_calibration_gap"].fillna(0.0)
        ).sum()
    )
    return {
        "sampled_pixels": int(len(probability)),
        "brier_score": brier,
        "expected_calibration_error": float(ece),
        "mean_predictive_entropy": float(entropy.mean()),
        "low_confidence_fraction": float(
            ((probability >= 0.4) & (probability <= 0.6)).mean()
        ),
    }
