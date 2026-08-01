import numpy as np

from aereo_water.evaluation.statistics import (
    paired_bootstrap_difference,
    wilcoxon_paired,
)


def test_paired_bootstrap_detects_improvement():
    reference = np.array([0.3, 0.4, 0.5, 0.6])
    current = reference + 0.1
    result = paired_bootstrap_difference(
        current, reference, iterations=200, seed=1
    )
    assert result["mean_difference"] > 0
    assert result["improved_fraction"] == 1.0


def test_wilcoxon_equal_arrays():
    values = np.array([0.1, 0.2, 0.3])
    result = wilcoxon_paired(values, values)
    assert result["wilcoxon_p_value"] == 1.0
