import torch
import pytest

from aereo_water.config import TrainingConfig
from aereo_water.training.engine import build_adamw_optimizer
from aereo_water.training.hpo import config_from_parameters


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(4, 2)
        self.norm = torch.nn.LayerNorm(2)


def test_weight_decay_groups_exclude_bias_and_norm():
    model = TinyModel()
    optimizer, summary = build_adamw_optimizer(
        model,
        learning_rate=1e-3,
        weight_decay=1e-2,
    )
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 1e-2
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert summary["no_decay_parameters"] > 0


def test_hpo_parameters_derive_dice_weight():
    config = config_from_parameters(
        TrainingConfig(),
        {
            "learning_rate": 2e-5,
            "weight_decay": 1e-3,
            "ce_weight": 0.7,
            "warmup_ratio": 0.05,
            "augmentation_profile": "light",
        },
        epochs=3,
    )
    assert config.dice_weight == pytest.approx(0.3)
    assert config.epochs == 3


def test_training_loss_ignores_letterbox_padding():
    from aereo_water.training.engine import (
        combined_cross_entropy_dice_loss,
    )

    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    targets = torch.tensor([[[1, 255], [0, 255]]], dtype=torch.long)
    loss, components = combined_cross_entropy_dice_loss(
        logits,
        targets,
        ce_weight=0.5,
        dice_weight=0.5,
    )
    assert torch.isfinite(loss)
    assert torch.isfinite(components["cross_entropy"])
    assert torch.isfinite(components["dice_loss"])
