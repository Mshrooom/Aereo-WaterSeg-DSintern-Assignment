import json

import pytest

from aereo_water.pipeline.state import StageState
from aereo_water.registry import (
    build_model_record,
    validate_model_record,
    write_model_registry,
)


def test_stage_dependency_validation(tmp_path):
    state = StageState(tmp_path / "state.json")
    with pytest.raises(RuntimeError):
        state.require("hpo")
    state.start("data")
    state.complete("data", evidence=["manifest.csv"])
    state.require("hpo")


def test_model_registry_uses_valid_json(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_bytes(b"weights")
    record = build_model_record(
        model_version="v1",
        model_name="model",
        base_model="base",
        checkpoint_dir=checkpoint,
        artifact_relative_path="artifacts/checkpoint",
        mlflow_artifact_uri="file:///tmp/mlflow",
        github_release_uri="",
        split_registry_sha256="a" * 64,
        git_commit="abc",
        hpo_study="study",
        hpo_study_fingerprint="fingerprint",
        final_parameters={"learning_rate": 1e-4},
        best_epoch=2,
        validation_threshold=0.5,
        validation_metrics={"iou": 0.7, "dice": 0.8},
        test_metrics={
            "iou": 0.69,
            "dice": 0.79,
            "precision": 0.8,
            "recall": 0.78,
            "boundary_f1": 0.5,
        },
        latency_metrics={
            "p50_model_forward_ms": 10,
            "p95_model_forward_ms": 12,
            "p50_end_to_end_ms": 20,
            "p95_end_to_end_ms": 30,
            "throughput_images_per_second": 100,
        },
        deployment_status="candidate",
        docker_validation_status="pending",
    )
    validate_model_record(record)
    assert json.loads(record["final_parameters_json"])[
        "learning_rate"
    ] == 1e-4
    csv_path, json_path = write_model_registry(
        record,
        tmp_path / "registry.csv",
        tmp_path / "selected.json",
    )
    assert csv_path.exists()
    assert json_path.exists()


def test_stage_invalidation_removes_downstream_records(tmp_path):
    from aereo_water.pipeline.state import StageState

    state = StageState(tmp_path / "state.json")
    for stage in ["data", "hpo", "confirmation"]:
        state.start(stage)
        state.complete(stage)
    state.invalidate_from("hpo")
    assert state.is_complete("data")
    assert not state.is_complete("hpo")
    assert not state.is_complete("confirmation")
