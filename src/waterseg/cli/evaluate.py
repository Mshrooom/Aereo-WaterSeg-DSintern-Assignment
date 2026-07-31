from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from waterseg.config import load_config
from waterseg.data.dataset import WaterDataset, list_collate
from waterseg.data.transforms import JointTransform
from waterseg.evaluate import evaluate_prompt_suite
from waterseg.models.sam_water import SamWaterModel


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate all prompt modes on the complete test split")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", default="")
    args = parser.parse_args()
    cfg = load_config(args.config)
    output = Path(cfg.paths.output_dir)
    checkpoint = Path(args.checkpoint) if args.checkpoint else output / "checkpoints" / "best.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    threshold = float(payload.get("metadata", {}).get("threshold", cfg.inference.threshold))
    suffix = "_tiles" if cfg.data.materialize_tiles else ""
    test_df = pd.read_csv(output / f"test{suffix}.csv")
    loader = DataLoader(
        WaterDataset(test_df, JointTransform(training=False)), batch_size=cfg.train.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True, collate_fn=list_collate,
        persistent_workers=cfg.data.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SamWaterModel(cfg.model.model_id, cfg.model.trainable_parts, cfg.model.unfreeze_last_vision_blocks).to(device)
    model.load_trainable_checkpoint(checkpoint, strict=False)
    table = evaluate_prompt_suite(model, loader, device, cfg, threshold, output / "evaluation")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
