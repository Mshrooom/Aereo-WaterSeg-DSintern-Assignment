from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from waterseg.cli.train import run_training
from waterseg.config import load_config


def _copy_registry_files(source: Path, destination: Path, materialize_tiles: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    names = ["manifest.csv", "train.csv", "val.csv", "test.csv", "dataset_summary.json"]
    if materialize_tiles:
        names.extend(["train_tiles.csv", "val_tiles.csv", "test_tiles.csv"])
    for name in names:
        path = source / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run `waterseg-prepare --config <config>` before starting the sweep."
            )
        shutil.copy2(path, destination / name)


def main() -> None:
    parser = argparse.ArgumentParser(description="W&B sweep entry point")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    import wandb

    cfg = load_config(args.config)
    base_output = Path(cfg.paths.output_dir)
    with wandb.init(project=cfg.tracking.project, entity=cfg.tracking.entity or None) as run:
        sweep = wandb.config
        cfg.train.learning_rate = float(sweep.get("learning_rate", cfg.train.learning_rate))
        cfg.train.weight_decay = float(sweep.get("weight_decay", cfg.train.weight_decay))
        cfg.train.bce_weight = float(sweep.get("bce_weight", cfg.train.bce_weight))
        cfg.train.dice_weight = float(sweep.get("dice_weight", cfg.train.dice_weight))
        cfg.train.focal_weight = float(sweep.get("focal_weight", cfg.train.focal_weight))
        cfg.train.epochs = int(sweep.get("epochs", cfg.train.epochs))

        none_weight = float(sweep.get("none_prompt_weight", cfg.prompts.train_weights[0]))
        none_weight = min(max(none_weight, 0.05), 0.80)
        remaining = 1.0 - none_weight
        other = cfg.prompts.train_weights[1:]
        other_total = sum(other)
        cfg.prompts.train_weights = [none_weight] + [remaining * value / other_total for value in other]

        run_output = base_output / "sweeps" / run.id
        _copy_registry_files(base_output, run_output, cfg.data.materialize_tiles)
        cfg.paths.output_dir = str(run_output)
        cfg.tracking.provider = "wandb"
        cfg.tracking.wandb_mode = "online"
        cfg.tracking.run_name = run.name
        run_training(cfg)


if __name__ == "__main__":
    main()
