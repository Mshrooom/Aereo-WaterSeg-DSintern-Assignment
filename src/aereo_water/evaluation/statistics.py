from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap_difference(
    current: np.ndarray,
    reference: np.ndarray,
    *,
    iterations: int = 5000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if current.shape != reference.shape:
        raise ValueError("Paired arrays must have equal shape")
    if current.ndim != 1 or not len(current):
        raise ValueError("Paired arrays must be non-empty one-dimensional arrays")

    differences = current - reference
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=np.float64)
    for index in range(iterations):
        sample = rng.integers(0, len(differences), size=len(differences))
        means[index] = differences[sample].mean()
    alpha = 1.0 - confidence
    lower = float(np.quantile(means, alpha / 2))
    upper = float(np.quantile(means, 1 - alpha / 2))
    return {
        "paired_count": int(len(differences)),
        "mean_difference": float(differences.mean()),
        "median_difference": float(np.median(differences)),
        "confidence": float(confidence),
        "ci_lower": lower,
        "ci_upper": upper,
        "improved_fraction": float((differences > 0).mean()),
        "degraded_fraction": float((differences < 0).mean()),
        "unchanged_fraction": float((differences == 0).mean()),
    }


def wilcoxon_paired(current: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    from scipy.stats import wilcoxon

    current = np.asarray(current, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if current.shape != reference.shape:
        raise ValueError("Paired arrays must have equal shape")
    differences = current - reference
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return {"wilcoxon_statistic": 0.0, "wilcoxon_p_value": 1.0}
    from scipy.stats import rankdata

    result = wilcoxon(current, reference, zero_method="wilcox")
    nonzero_differences = differences[differences != 0]
    ranks = rankdata(np.abs(nonzero_differences), method="average")
    positive_rank_sum = float(ranks[nonzero_differences > 0].sum())
    negative_rank_sum = float(ranks[nonzero_differences < 0].sum())
    total_rank_sum = positive_rank_sum + negative_rank_sum
    rank_biserial = (
        (positive_rank_sum - negative_rank_sum) / total_rank_sum
        if total_rank_sum
        else 0.0
    )
    return {
        "wilcoxon_statistic": float(result.statistic),
        "wilcoxon_p_value": float(result.pvalue),
        "positive_rank_sum": positive_rank_sum,
        "negative_rank_sum": negative_rank_sum,
        "rank_biserial_effect": float(rank_biserial),
    }


def add_performance_slices(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["water_coverage_slice"] = pd.cut(
        output["water_fraction"],
        bins=[-1e-9, 0.0, 0.10, 0.25, 0.50, 0.75, 1.0],
        labels=[
            "empty",
            "0-10%",
            "10-25%",
            "25-50%",
            "50-75%",
            "75-100%",
        ],
        include_lowest=True,
    )
    area = output["width"] * output["height"]
    output["image_size_slice"] = pd.qcut(
        area,
        q=3,
        labels=["small", "medium", "large"],
        duplicates="drop",
    )
    return output


def summarize_slices(
    frame: pd.DataFrame,
    *,
    slice_columns: list[str],
    metric_columns: list[str],
) -> pd.DataFrame:
    rows = []
    for slice_column in slice_columns:
        for value, group in frame.groupby(slice_column, observed=True):
            row = {
                "slice_dimension": slice_column,
                "slice_value": str(value),
                "images": int(len(group)),
            }
            for metric in metric_columns:
                if metric in group:
                    row[metric] = float(group[metric].mean())
            rows.append(row)
    return pd.DataFrame(rows)
