import numpy as np
import pytest

from aereo_water.evaluation.metrics import (
    probability_calibration_metrics,
    segmentation_metrics,
)


def test_perfect_prediction():
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    metrics = segmentation_metrics(
        target, target, include_boundary_metrics=False
    )
    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0


def test_empty_mask_convention_is_explicit():
    empty = np.zeros((4, 4), dtype=np.uint8)
    perfect = segmentation_metrics(
        empty,
        empty,
        include_boundary_metrics=False,
        empty_policy="perfect_if_both_empty",
    )
    zero = segmentation_metrics(
        empty,
        empty,
        include_boundary_metrics=False,
        empty_policy="zero_if_undefined",
    )
    assert perfect["iou"] == 1.0
    assert zero["iou"] == 0.0


def test_empty_target_false_positive():
    target = np.zeros((4, 4), dtype=np.uint8)
    prediction = np.zeros((4, 4), dtype=np.uint8)
    prediction[0, 0] = 1
    metrics = segmentation_metrics(
        prediction, target, include_boundary_metrics=False
    )
    assert metrics["iou"] == 0.0
    assert metrics["empty_mask_false_positive"] is True


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="Shape mismatch"):
        segmentation_metrics(
            np.zeros((2, 2)),
            np.zeros((3, 3)),
            include_boundary_metrics=False,
        )


def test_calibration_metrics_are_finite():
    probability = np.array([0.1, 0.9, 0.2, 0.8])
    target = np.array([0, 1, 0, 1])
    metrics = probability_calibration_metrics(
        probability, target, bins=4
    )
    assert 0 <= metrics["brier_score"] <= 1
    assert 0 <= metrics["expected_calibration_error"] <= 1


def test_reliability_table_accounts_for_every_pixel():
    from aereo_water.evaluation.metrics import reliability_table

    probabilities = np.array([0.0, 0.2, 0.6, 1.0])
    targets = np.array([0, 0, 1, 1])
    table = reliability_table(probabilities, targets, bins=4)
    assert table["pixel_count"].sum() == 4
    assert table["pixel_fraction"].sum() == pytest.approx(1.0)
