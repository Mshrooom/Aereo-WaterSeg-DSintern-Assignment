import numpy as np

from waterseg.auto_prompt import automatic_prompt_from_probability


def test_automatic_prompt_detects_candidate():
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[10:30, 20:45] = 0.9
    result = automatic_prompt_from_probability(probability, min_component_area=4)
    assert result.status == "ok"
    assert result.components == 1
    assert result.prompt.boxes is not None
    assert result.prompt.points is not None
    assert result.prompt.labels[0][0] == 1


def test_automatic_prompt_handles_no_candidate():
    probability = np.zeros((32, 32), dtype=np.float32)
    result = automatic_prompt_from_probability(probability)
    assert result.status == "no_coarse_candidate"
    assert result.prompt.mode == "none"
