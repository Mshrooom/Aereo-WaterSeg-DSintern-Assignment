from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SegFormerSpec:
    model_id: str
    num_labels: int = 2
    id2label: dict[int, str] | None = None
    label2id: dict[str, int] | None = None

    def resolved_id2label(self) -> dict[int, str]:
        return self.id2label or {0: "non_water", 1: "water"}

    def resolved_label2id(self) -> dict[str, int]:
        return self.label2id or {"non_water": 0, "water": 1}


def build_segformer(spec: SegFormerSpec):
    """Build a binary SegFormer model from pretrained weights."""
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    processor = SegformerImageProcessor.from_pretrained(
        spec.model_id,
        do_reduce_labels=False,
    )
    model = SegformerForSemanticSegmentation.from_pretrained(
        spec.model_id,
        num_labels=spec.num_labels,
        id2label=spec.resolved_id2label(),
        label2id=spec.resolved_label2id(),
        ignore_mismatched_sizes=True,
    )
    return model, processor


def load_segformer_checkpoint(
    checkpoint_dir: str,
    *,
    device: Any,
):
    from transformers import (
        SegformerForSemanticSegmentation,
        SegformerImageProcessor,
    )

    processor = SegformerImageProcessor.from_pretrained(checkpoint_dir)
    model = SegformerForSemanticSegmentation.from_pretrained(checkpoint_dir)
    model.to(device)
    model.eval()
    return model, processor


def model_parameter_summary(model) -> dict[str, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_parameters": int(total),
        "trainable_parameters": int(trainable),
        "frozen_parameters": int(total - trainable),
    }
