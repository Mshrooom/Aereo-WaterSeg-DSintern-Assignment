from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from aereo_water.config import load_config
from aereo_water.evaluation.evaluator import evaluate_manifest
from aereo_water.models.segformer import load_segformer_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--threshold-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    threshold = float(
        json.loads(Path(args.threshold_json).read_text())[
            "validation_threshold"
        ]
    )
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    model, processor = load_segformer_checkpoint(
        args.checkpoint,
        device=device,
    )
    frame, calibration = evaluate_manifest(
        model,
        processor,
        manifest,
        image_size=config.data.image_size,
        resize_policy=config.data.resize_policy,
        threshold=threshold,
        device=device,
        batch_size=config.evaluation.batch_size,
        num_workers=config.data.num_workers,
        output_csv=args.output,
        prediction_dir=args.predictions,
        include_boundary_metrics=(
            config.evaluation.include_boundary_metrics
        ),
        boundary_tolerance=(
            config.evaluation.boundary_tolerance_pixels
        ),
        empty_policy=config.evaluation.empty_mask_policy,
        calibration_bins=config.evaluation.calibration_bins,
        calibration_output_dir=(
            Path(args.output).parent / "calibration"
        ),
    )
    Path(args.output).with_suffix(".calibration.json").write_text(
        json.dumps(calibration, indent=2, sort_keys=True)
    )
    print(frame.groupby("split")["iou"].mean())


if __name__ == "__main__":
    main()
