from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def add_stratification_bins(manifest: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    result = manifest.copy()
    ranked = result["water_fraction"].rank(method="first")
    effective_bins = min(bins, max(2, len(result) // 20))
    result["stratum"] = pd.qcut(ranked, q=effective_bins, labels=False, duplicates="drop")
    result["stratum"] = result["stratum"].fillna(0).astype(int)
    return result


def stratified_split(
    manifest: pd.DataFrame,
    train_fraction: float,
    val_fraction: float,
    test_fraction: float,
    seed: int,
    group_column: str = "image_sha256",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_fraction + val_fraction + test_fraction - 1.0) > 1e-6:
        raise ValueError("Split fractions must sum to one")
    dataframe = manifest.copy()
    if "stratum" not in dataframe:
        dataframe = add_stratification_bins(dataframe)
    if group_column not in dataframe:
        group_column = "image_id"

    # Duplicate images are assigned as a group to avoid leakage.
    groups = (
        dataframe.groupby(group_column, dropna=False)
        .agg(stratum=("stratum", "first"), row_indices=("image_id", lambda _: list(_.index)))
        .reset_index()
    )
    rng = np.random.default_rng(seed)
    assignments = {}
    for _, stratum_groups in groups.groupby("stratum"):
        indices = stratum_groups.index.to_numpy().copy()
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(round(n * train_fraction))
        n_val = int(round(n * val_fraction))
        if n >= 3:
            n_train = min(max(n_train, 1), n - 2)
            n_val = min(max(n_val, 1), n - n_train - 1)
        for idx in indices[:n_train]:
            assignments[idx] = "train"
        for idx in indices[n_train : n_train + n_val]:
            assignments[idx] = "val"
        for idx in indices[n_train + n_val :]:
            assignments[idx] = "test"

    group_to_split = {groups.loc[idx, group_column]: split for idx, split in assignments.items()}
    dataframe["split"] = dataframe[group_column].map(group_to_split)
    if dataframe["split"].isna().any():
        raise RuntimeError("Some rows were not assigned to a split")

    train = dataframe[dataframe["split"] == "train"].reset_index(drop=True)
    val = dataframe[dataframe["split"] == "val"].reset_index(drop=True)
    test = dataframe[dataframe["split"] == "test"].reset_index(drop=True)
    return train, val, test


def assert_no_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    for column in ("image_id", "image_sha256"):
        if column not in train:
            continue
        train_values, val_values, test_values = set(train[column]), set(val[column]), set(test[column])
        if train_values & val_values or train_values & test_values or val_values & test_values:
            raise AssertionError(f"Leakage detected using column: {column}")
