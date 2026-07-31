from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

LOGGER = logging.getLogger(__name__)


class ExperimentTracker:
    def __init__(self, output_dir: str | Path, config: Dict[str, Any], tracking_config: Any):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.output_dir / "metrics.jsonl"
        self.run = None
        self.wandb = None
        self.owns_run = False
        if tracking_config.provider.lower() == "wandb":
            try:
                import wandb

                self.wandb = wandb
                existing_run = wandb.run
                self.run = existing_run or wandb.init(
                    project=tracking_config.project,
                    entity=tracking_config.entity or None,
                    name=tracking_config.run_name,
                    mode=tracking_config.wandb_mode,
                    config=config,
                )
                self.owns_run = existing_run is None
                if wandb.run is not None:
                    wandb.config.update(config, allow_val_change=True)
            except Exception as error:
                LOGGER.exception("W&B initialization failed; falling back to local tracking: %s", error)

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        record = {"step": step, **metrics}
        with open(self.history_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=str) + "\n")
        if self.run is not None:
            self.run.log(metrics, step=step)

    def log_artifact(self, path: str | Path, name: str, artifact_type: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        if self.run is None or self.wandb is None:
            return
        artifact = self.wandb.Artifact(name=name, type=artifact_type, metadata=metadata or {})
        path = Path(path)
        if path.is_dir():
            artifact.add_dir(str(path))
        else:
            artifact.add_file(str(path))
        self.run.log_artifact(artifact)

    def finish(self) -> None:
        if self.run is not None and self.owns_run:
            self.run.finish()
