from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from aereo_water.config import load_config
from aereo_water.models.segformer import SegFormerSpec
from aereo_water.training.engine import train_segformer
from aereo_water.training.hpo import config_from_parameters
from aereo_water.utils import get_git_commit, sha256_dataframe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--parameters-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repository", default=".")
    parser.add_argument("--resume-from")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = pd.read_csv(args.manifest)
    train_df = manifest[manifest["split"] == "train"].copy()
    validation_df = manifest[
        manifest["split"].isin(["validation", "val"])
    ].copy()
    parameters = json.loads(Path(args.parameters_json).read_text())
    final_config = config_from_parameters(
        config.training,
        parameters,
        epochs=config.training.epochs,
        save_every_epoch=True,
    )
    output = Path(args.output)
    train_segformer(
        train_df,
        validation_df,
        model_spec=SegFormerSpec(
            model_id=config.model.model_id,
            num_labels=config.model.num_labels,
            id2label=config.model.id2label,
            label2id=config.model.label2id,
        ),
        config=final_config,
        image_size=config.data.image_size,
        resize_policy=config.data.resize_policy,
        output_dir=output,
        device="cuda" if torch.cuda.is_available() else "cpu",
        run_name="segformer_final_training",
        experiment_name=config.tracking.mlflow_experiment_final,
        mlflow_tracking_uri=f"sqlite:///{output.parent / 'mlflow.db'}",
        mlflow_artifact_root=output.parent / "mlartifacts",
        wandb_project=config.tracking.wandb_project,
        wandb_mode=config.tracking.wandb_mode,
        wandb_root=output.parent / "wandb",
        repo_git_commit=get_git_commit(args.repository),
        split_registry_sha256=sha256_dataframe(
            manifest[["image_id", "split"]]
        ),
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
