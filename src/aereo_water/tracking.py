from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Any


class DualTracker:
    """Track runs in MLflow and optionally mirror them to Weights & Biases.

    MLflow is the authoritative local system of record. W&B is a visualization
    mirror and may run offline.
    """

    def __init__(
        self,
        *,
        run_name: str,
        experiment_name: str,
        mlflow_tracking_uri: str,
        mlflow_artifact_root: str | Path,
        wandb_project: str | None = None,
        wandb_mode: str = "disabled",
        wandb_root: str | Path | None = None,
        wandb_group: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> None:
        self.run_name = run_name
        self.experiment_name = experiment_name
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.mlflow_artifact_root = Path(mlflow_artifact_root)
        self.wandb_project = wandb_project
        self.wandb_mode = wandb_mode
        self.wandb_root = Path(wandb_root) if wandb_root else None
        self.wandb_group = wandb_group
        self.tags = tags or {}
        self.mlflow = None
        self.mlflow_run = None
        self.wandb = None
        self.wandb_run = None
        self._explicit_status: str | None = None

    def __enter__(self) -> "DualTracker":
        import mlflow

        self.mlflow = mlflow
        self.mlflow_artifact_root.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(
                self.experiment_name,
                artifact_location=self.mlflow_artifact_root.as_uri(),
            )
        else:
            experiment_id = experiment.experiment_id
        self.mlflow_run = mlflow.start_run(
            experiment_id=experiment_id,
            run_name=self.run_name,
            tags=self.tags,
        )

        if (
            self.wandb_project
            and self.wandb_mode != "disabled"
            and self.wandb_root is not None
        ):
            self.wandb_root.mkdir(parents=True, exist_ok=True)
            os.environ["WANDB_DIR"] = str(self.wandb_root)
            os.environ["WANDB_CACHE_DIR"] = str(
                self.wandb_root.parent / "wandb_cache"
            )
            try:
                import wandb

                self.wandb = wandb
                self.wandb_run = wandb.init(
                    project=self.wandb_project,
                    name=self.run_name,
                    group=self.wandb_group,
                    mode=self.wandb_mode,
                    reinit=True,
                    config={},
                    tags=sorted(set(self.tags.values())),
                    dir=str(self.wandb_root),
                )
            except Exception as exc:
                print(
                    "W&B initialization failed; MLflow remains authoritative. "
                    f"{type(exc).__name__}: {exc}"
                )
                self.wandb = None
                self.wandb_run = None
        return self

    @property
    def mlflow_run_id(self) -> str | None:
        return (
            self.mlflow_run.info.run_id
            if self.mlflow_run is not None
            else None
        )

    @property
    def mlflow_artifact_uri(self) -> str | None:
        return (
            self.mlflow_run.info.artifact_uri
            if self.mlflow_run is not None
            else None
        )

    def mark_pruned(self) -> None:
        self._explicit_status = "KILLED"
        self.set_tags({"trial_status": "PRUNED"})

    def log_params(self, params: dict[str, Any]) -> None:
        clean = {
            key: value
            for key, value in params.items()
            if value is not None
        }
        if self.mlflow is not None:
            self.mlflow.log_params(clean)
        if self.wandb_run is not None:
            self.wandb_run.config.update(clean, allow_val_change=True)

    def log_metrics(
        self,
        metrics: dict[str, float | int],
        *,
        step: int | None = None,
    ) -> None:
        clean = {
            key: float(value)
            for key, value in metrics.items()
            if value is not None
        }
        if self.mlflow is not None:
            self.mlflow.log_metrics(clean, step=step)
        if self.wandb_run is not None:
            payload = dict(clean)
            if step is not None:
                payload["epoch"] = step
            self.wandb_run.log(payload, step=step)

    def log_artifact(
        self,
        path: str | Path,
        *,
        artifact_path: str | None = None,
        wandb_artifact_name: str | None = None,
        wandb_artifact_type: str = "evidence",
    ) -> None:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        if self.mlflow is not None:
            if source.is_dir():
                self.mlflow.log_artifacts(
                    str(source),
                    artifact_path=artifact_path,
                )
            else:
                self.mlflow.log_artifact(
                    str(source),
                    artifact_path=artifact_path,
                )
        if self.wandb_run is not None and wandb_artifact_name:
            artifact = self.wandb.Artifact(
                wandb_artifact_name,
                type=wandb_artifact_type,
            )
            if source.is_dir():
                artifact.add_dir(str(source))
            else:
                artifact.add_file(str(source))
            self.wandb_run.log_artifact(artifact)

    def log_dict(self, data: dict[str, Any], artifact_file: str) -> None:
        if self.mlflow is not None:
            self.mlflow.log_dict(data, artifact_file)
        if self.wandb_run is not None:
            local = Path(self.wandb_run.dir) / artifact_file.replace("/", "_")
            local.parent.mkdir(parents=True, exist_ok=True)
            local.write_text(
                json.dumps(data, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            )
            self.wandb.save(str(local), base_path=self.wandb_run.dir)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self.mlflow is not None:
            self.mlflow.set_tags(tags)
        if self.wandb_run is not None:
            self.wandb_run.tags = tuple(
                sorted(set(self.wandb_run.tags or ()) | set(tags.values()))
            )

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self.wandb_run is not None:
            self.wandb_run.finish(exit_code=1 if exc else 0)
        if self.mlflow is not None:
            status = self._explicit_status or ("FAILED" if exc else "FINISHED")
            with contextlib.suppress(Exception):
                self.mlflow.end_run(status=status)
