from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from aereo_water.config import load_config
from aereo_water.models.segformer import SegFormerSpec
from aereo_water.training.hpo import run_hpo
from aereo_water.utils import get_git_commit, sha256_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--wandb-mode")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["split"] == "train"].copy()
    validation_df = manifest[
        manifest["split"].isin(["validation", "val"])
    ].copy()
    if set(manifest["split"]) & {"test"} and (
        train_df["image_id"].isin(
            manifest[manifest["split"] == "test"]["image_id"]
        ).any()
        or validation_df["image_id"].isin(
            manifest[manifest["split"] == "test"]["image_id"]
        ).any()
    ):
        raise RuntimeError("Test IDs leaked into HPO manifests")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    split_hash = sha256_dataframe(
        manifest[["image_id", "split"]]
    )
    baseline = {
        "learning_rate": 6e-5,
        "weight_decay": 1e-4,
        "ce_weight": 0.6,
        "dice_weight": 0.4,
        "warmup_ratio": 0.1,
        "augmentation_profile": "moderate",
        "batch_size": 8,
        "gradient_accumulation_steps": 1,
    }
    run_hpo(
        train_df,
        validation_df,
        model_spec=SegFormerSpec(
            model_id=config.model.model_id,
            num_labels=config.model.num_labels,
            id2label=config.model.id2label,
            label2id=config.model.label2id,
        ),
        base_training_config=config.training,
        hpo_config=config.hpo,
        image_size=config.data.image_size,
        resize_policy=config.data.resize_policy,
        output_dir=output,
        device="cuda" if torch.cuda.is_available() else "cpu",
        mlflow_tracking_uri=f"sqlite:///{output / 'mlflow.db'}",
        mlflow_artifact_root=output / "mlartifacts",
        wandb_project=config.tracking.wandb_project,
        wandb_mode=args.wandb_mode or config.tracking.wandb_mode,
        wandb_root=output / "wandb",
        repo_git_commit=get_git_commit(args.repository),
        split_registry_sha256=split_hash,
        baseline_parameters=baseline,
    )


if __name__ == "__main__":
    main()
