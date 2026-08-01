from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from aereo_water.utils import (
    canonical_json_dumps,
    json_dump,
    sha256_file,
    utc_now_iso,
)


MODEL_REGISTRY_COLUMNS = [
    "model_version",
    "model_name",
    "base_model",
    "local_runtime_path",
    "artifact_relative_path",
    "mlflow_artifact_uri",
    "github_release_uri",
    "checkpoint_sha256",
    "split_registry_sha256",
    "git_commit",
    "hpo_study",
    "hpo_study_fingerprint",
    "final_parameters_json",
    "best_epoch",
    "validation_threshold",
    "validation_iou",
    "validation_dice",
    "test_iou",
    "test_dice",
    "test_precision",
    "test_recall",
    "test_boundary_f1",
    "p50_model_forward_ms",
    "p95_model_forward_ms",
    "p50_end_to_end_ms",
    "p95_end_to_end_ms",
    "throughput_images_per_second",
    "deployment_status",
    "docker_validation_status",
    "created_at_utc",
]


def checkpoint_weights_path(checkpoint_dir: str | Path) -> Path:
    checkpoint = Path(checkpoint_dir)
    for filename in ("model.safetensors", "pytorch_model.bin"):
        candidate = checkpoint / filename
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"No model.safetensors or pytorch_model.bin in {checkpoint}"
    )


def build_model_record(
    *,
    model_version: str,
    model_name: str,
    base_model: str,
    checkpoint_dir: str | Path,
    artifact_relative_path: str,
    mlflow_artifact_uri: str,
    github_release_uri: str,
    split_registry_sha256: str,
    git_commit: str,
    hpo_study: str,
    hpo_study_fingerprint: str,
    final_parameters: dict[str, Any],
    best_epoch: int,
    validation_threshold: float,
    validation_metrics: dict[str, float],
    test_metrics: dict[str, float],
    latency_metrics: dict[str, float],
    deployment_status: str,
    docker_validation_status: str,
) -> dict[str, Any]:
    weights = checkpoint_weights_path(checkpoint_dir)
    return {
        "model_version": model_version,
        "model_name": model_name,
        "base_model": base_model,
        "local_runtime_path": str(Path(checkpoint_dir)),
        "artifact_relative_path": artifact_relative_path,
        "mlflow_artifact_uri": mlflow_artifact_uri,
        "github_release_uri": github_release_uri,
        "checkpoint_sha256": sha256_file(weights),
        "split_registry_sha256": split_registry_sha256,
        "git_commit": git_commit,
        "hpo_study": hpo_study,
        "hpo_study_fingerprint": hpo_study_fingerprint,
        "final_parameters_json": canonical_json_dumps(final_parameters),
        "best_epoch": int(best_epoch),
        "validation_threshold": float(validation_threshold),
        "validation_iou": float(validation_metrics.get("iou", float("nan"))),
        "validation_dice": float(
            validation_metrics.get("dice", float("nan"))
        ),
        "test_iou": float(test_metrics.get("iou", float("nan"))),
        "test_dice": float(test_metrics.get("dice", float("nan"))),
        "test_precision": float(
            test_metrics.get("precision", float("nan"))
        ),
        "test_recall": float(test_metrics.get("recall", float("nan"))),
        "test_boundary_f1": float(
            test_metrics.get("boundary_f1", float("nan"))
        ),
        "p50_model_forward_ms": float(
            latency_metrics.get("p50_model_forward_ms", float("nan"))
        ),
        "p95_model_forward_ms": float(
            latency_metrics.get("p95_model_forward_ms", float("nan"))
        ),
        "p50_end_to_end_ms": float(
            latency_metrics.get("p50_end_to_end_ms", float("nan"))
        ),
        "p95_end_to_end_ms": float(
            latency_metrics.get("p95_end_to_end_ms", float("nan"))
        ),
        "throughput_images_per_second": float(
            latency_metrics.get(
                "throughput_images_per_second",
                float("nan"),
            )
        ),
        "deployment_status": deployment_status,
        "docker_validation_status": docker_validation_status,
        "created_at_utc": utc_now_iso(),
    }


def validate_model_record(record: dict[str, Any]) -> None:
    missing = set(MODEL_REGISTRY_COLUMNS) - set(record)
    if missing:
        raise ValueError(
            f"Model record is missing columns: {sorted(missing)}"
        )
    parameters = json.loads(record["final_parameters_json"])
    if not isinstance(parameters, dict):
        raise ValueError("final_parameters_json must decode to an object")
    if len(record["checkpoint_sha256"]) != 64:
        raise ValueError("checkpoint_sha256 must be a SHA-256 digest")
    if len(record["split_registry_sha256"]) != 64:
        raise ValueError("split_registry_sha256 must be a SHA-256 digest")


def write_model_registry(
    record: dict[str, Any],
    csv_path: str | Path,
    selected_json_path: str | Path,
) -> tuple[Path, Path]:
    validate_model_record(record)
    csv_output = Path(csv_path)
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    if csv_output.exists():
        current = pd.read_csv(csv_output)
        current = current[
            current["model_version"].astype(str)
            != str(record["model_version"])
        ]
        frame = pd.concat(
            [current, pd.DataFrame([record])],
            ignore_index=True,
        )
    else:
        frame = pd.DataFrame([record])
    for column in MODEL_REGISTRY_COLUMNS:
        if column not in frame:
            frame[column] = None
    frame[MODEL_REGISTRY_COLUMNS].to_csv(csv_output, index=False)
    selected_output = json_dump(record, selected_json_path)
    return csv_output, selected_output
