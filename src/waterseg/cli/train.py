from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from waterseg.config import load_config
from waterseg.data.dataset import WaterDataset, list_collate
from waterseg.data.transforms import JointTransform
from waterseg.engine import fit
from waterseg.logging_utils import configure_logging
from waterseg.models.sam_water import SamWaterModel
from waterseg.tracking import ExperimentTracker
from waterseg.utils import ensure_dir, seed_everything, write_json


def run_training(cfg: Any) -> dict:
    configure_logging()
    seed_everything(cfg.train.seed)
    output = ensure_dir(cfg.paths.output_dir)
    write_json(output / "resolved_config.json", cfg.to_dict())

    suffix = "_tiles" if cfg.data.materialize_tiles else ""
    train_df = pd.read_csv(output / f"train{suffix}.csv")
    val_df = pd.read_csv(output / f"val{suffix}.csv")
    train_loader = DataLoader(
        WaterDataset(train_df, JointTransform(training=True)),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=list_collate,
        persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        WaterDataset(val_df, JointTransform(training=False)),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=list_collate,
        persistent_workers=cfg.data.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamWaterModel(
        cfg.model.model_id,
        cfg.model.trainable_parts,
        cfg.model.unfreeze_last_vision_blocks,
    ).to(device)
    tracker = ExperimentTracker(output, cfg.to_dict(), cfg.tracking)
    tracker.log_artifact(output / "manifest.csv", "water-dataset-manifest", "dataset")
    result = fit(model, train_loader, val_loader, device, cfg, tracker, output)
    tracker.log_artifact(result["checkpoint"], "sam-water-segmentation", "model", result)
    tracker.finish()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune SAM for water segmentation")
    parser.add_argument("--config", required=True)
    parser.add_argument("--learning-rate", "--learning_rate", type=float, default=None)
    parser.add_argument("--weight-decay", "--weight_decay", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", type=int, default=None)
    parser.add_argument("--run-name", "--run_name", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay
    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.run_name is not None:
        cfg.tracking.run_name = args.run_name
    print(run_training(cfg))


if __name__ == "__main__":
    main()
