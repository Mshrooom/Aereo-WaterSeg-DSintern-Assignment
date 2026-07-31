import numpy as np

from waterseg.metrics import BinarySegmentationMeter, confusion_metrics


def test_perfect_confusion_metrics():
    metrics = confusion_metrics(tp=10, fp=0, fn=0, tn=20)
    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0
    assert metrics["mcc"] == 1.0


def test_meter_accepts_probabilities():
    target = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    probability = np.array([[0.1, 0.9], [0.8, 0.2]], dtype=np.float32)
    meter = BinarySegmentationMeter(threshold=0.5)
    meter.update(probability, target, "sample")
    result = meter.compute()
    assert result["iou"] == 1.0
    assert result["brier_score"] < 0.05
