import numpy as np

from waterseg.prompting import box_from_mask, build_prompt_batch


def test_box_contains_mask_with_no_jitter():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[5:11, 7:18] = 1
    box = box_from_mask(mask, 0.0, np.random.default_rng(1))
    assert box.tolist() == [7.0, 5.0, 17.0, 10.0]


def test_point_prompts_have_expected_count():
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    prompt = build_prompt_batch([mask], "points", 3, 1, 0.0, [np.random.default_rng(2)])
    assert prompt.points[0].shape == (4, 2)
    assert prompt.labels[0].tolist() == [1, 1, 1, 0]


def test_empty_mask_never_gets_foreground_label():
    mask = np.zeros((16, 16), dtype=np.uint8)
    prompt = build_prompt_batch([mask], "points", 2, 1, 0.0, [np.random.default_rng(3)])
    assert prompt.labels[0].tolist() == [0, 0, 0]
