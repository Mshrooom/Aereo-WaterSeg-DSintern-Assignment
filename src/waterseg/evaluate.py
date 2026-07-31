from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import pandas as pd
import torch

from waterseg.engine import predict_loader
from waterseg.metrics import bootstrap_confidence_interval
from waterseg.utils import write_json


def evaluate_prompt_suite(
    model: Any,
    loader: Iterable[dict],
    device: torch.device,
    cfg: Any,
    threshold: float,
    output_dir: str | Path,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    confidence_intervals: Dict[str, dict] = {}
    for prompt_mode in cfg.prompts.eval_modes:
        metrics, per_image = predict_loader(
            model, loader, device, cfg, prompt_mode, threshold, compute_surface=True
        )
        per_image.to_csv(output_dir / f"test_per_image_{prompt_mode}.csv", index=False)
        row = {"prompt_mode": prompt_mode, **metrics}
        rows.append(row)
        confidence_intervals[prompt_mode] = {
            metric: bootstrap_confidence_interval(per_image[metric], seed=cfg.train.seed)
            for metric in ("iou", "dice", "boundary_f1")
        }
    table = pd.DataFrame(rows).sort_values("iou", ascending=False)
    table.to_csv(output_dir / "prompt_comparison.csv", index=False)
    write_json(output_dir / "prompt_comparison_ci.json", confidence_intervals)
    return table
