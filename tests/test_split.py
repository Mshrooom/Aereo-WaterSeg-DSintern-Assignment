import pandas as pd

from waterseg.data.split import add_stratification_bins, assert_no_leakage, stratified_split


def test_split_is_deterministic_and_disjoint():
    dataframe = pd.DataFrame({
        "image_id": [f"image_{index}" for index in range(100)],
        "image_sha256": [f"hash_{index}" for index in range(100)],
        "water_fraction": [index / 100 for index in range(100)],
    })
    dataframe = add_stratification_bins(dataframe, 5)
    first = stratified_split(dataframe, 0.7, 0.15, 0.15, seed=42)
    second = stratified_split(dataframe, 0.7, 0.15, 0.15, seed=42)
    assert [set(part.image_id) for part in first] == [set(part.image_id) for part in second]
    assert_no_leakage(*first)
    assert sum(len(part) for part in first) == len(dataframe)
