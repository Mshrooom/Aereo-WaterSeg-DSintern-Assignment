from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    return (1.0 - (2.0 * intersection + eps) / (denominator + eps)).mean()


def focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: float = 0.25) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probabilities = torch.sigmoid(logits)
    p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    return (alpha_t * (1.0 - p_t).pow(gamma) * bce).mean()


def hard_iou_target(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    predictions = torch.sigmoid(logits).detach() >= threshold
    targets_bool = targets.detach() >= 0.5
    intersection = (predictions & targets_bool).sum(dim=(1, 2, 3)).float()
    union = (predictions | targets_bool).sum(dim=(1, 2, 3)).float()
    return torch.where(union > 0, intersection / union, torch.ones_like(union))


@dataclass
class CombinedSegmentationLoss:
    bce_weight: float = 0.35
    dice_weight: float = 0.45
    focal_weight: float = 0.20
    iou_head_weight: float = 0.10

    def __call__(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        predicted_iou: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = soft_dice_loss(logits, targets)
        focal = focal_loss(logits, targets)
        total = self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal
        iou_regression = torch.zeros((), device=logits.device)
        if predicted_iou is not None:
            target_iou = hard_iou_target(logits, targets)
            predicted_iou = predicted_iou.reshape(predicted_iou.shape[0], -1)[:, 0]
            iou_regression = F.mse_loss(predicted_iou, target_iou)
            total = total + self.iou_head_weight * iou_regression
        return total, {
            "loss": float(total.detach()),
            "bce_loss": float(bce.detach()),
            "dice_loss": float(dice.detach()),
            "focal_loss": float(focal.detach()),
            "iou_head_loss": float(iou_regression.detach()),
        }
