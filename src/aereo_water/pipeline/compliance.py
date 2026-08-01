from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def build_compliance_table(
    *,
    output_root: str | Path,
    expected_completed_hpo_trials: int,
    expected_total_inference_rows: int,
    pytest_exit_code: int | None,
    docker_validation_complete: bool = False,
    presentation_complete: bool = False,
) -> pd.DataFrame:
    """Derive completion from evidence rather than hard-coded booleans."""
    root = Path(output_root)
    hpo_trials_path = root / "hpo" / "trials.csv"
    if hpo_trials_path.exists():
        trials = pd.read_csv(hpo_trials_path)
        completed_hpo = int(
            trials["state"].astype(str).str.contains("COMPLETE").sum()
        )
    else:
        completed_hpo = 0

    inference_path = root / "evaluation" / "segformer_v3_all_2841.csv"
    inference_rows = (
        len(pd.read_csv(inference_path))
        if inference_path.exists()
        else 0
    )
    mlflow_db = root / "tracking" / "mlflow.db"
    mlartifacts = root / "tracking" / "mlartifacts"
    wandb_root = root / "tracking" / "wandb"
    registry_json = root / "registry" / "data_registry.json"
    model_json = root / "registry" / "selected_model.json"
    inference_mask = (
        root / "production_inference" / "predicted_water_mask.png"
    )
    inference_overlay = (
        root / "production_inference" / "predicted_water_overlay.png"
    )
    inference_log = root / "production_inference" / "inference.jsonl"

    rows = [
        (
            "Validated ingestion",
            registry_json.exists(),
            str(registry_json),
        ),
        (
            "Normalization and synchronized augmentation",
            (root / "figures" / "preprocessing_and_augmentation.png").exists(),
            str(root / "figures" / "preprocessing_and_augmentation.png"),
        ),
        (
            "Overlapping tiling and reconstruction",
            (root / "tiling" / "tiling_manifest.csv").exists(),
            str(root / "tiling" / "tiling_manifest.csv"),
        ),
        (
            "Geospatial metadata-preserving tiling",
            (root / "tiling" / "geotiff_tiling_manifest.csv").exists(),
            str(root / "tiling" / "geotiff_tiling_manifest.csv"),
        ),
        (
            "Optuna HPO",
            completed_hpo >= expected_completed_hpo_trials,
            str(hpo_trials_path),
        ),
        (
            "MLflow database and artifact store",
            mlflow_db.exists() and mlartifacts.exists(),
            f"{mlflow_db}; {mlartifacts}",
        ),
        (
            "W&B offline mirror",
            wandb_root.exists()
            and any(
                path.is_dir()
                for pattern in ("run-*", "offline-run-*", "online-run-*")
                for path in wandb_root.glob(pattern)
            ),
            str(wandb_root),
        ),
        (
            "Same-code baseline confirmation",
            (
                root
                / "confirmation"
                / "confirmation_results.csv"
            ).exists(),
            str(
                root
                / "confirmation"
                / "confirmation_results.csv"
            ),
        ),
        (
            "Seed stability",
            (root / "stability" / "seed_stability.csv").exists(),
            str(root / "stability" / "seed_stability.csv"),
        ),
        (
            "Final resumable training",
            (
                root
                / "final_training"
                / "last_state.pt"
            ).exists(),
            str(root / "final_training" / "last_state.pt"),
        ),
        (
            "Validation-only threshold calibration",
            (
                root
                / "calibration"
                / "selected_threshold.json"
            ).exists(),
            str(root / "calibration" / "selected_threshold.json"),
        ),
        (
            "Frozen held-out test evaluation",
            (
                root
                / "evaluation"
                / "segformer_v3_test_metrics.json"
            ).exists(),
            str(
                root
                / "evaluation"
                / "segformer_v3_test_metrics.json"
            ),
        ),
        (
            "Full 2,841-image inference",
            inference_rows == expected_total_inference_rows,
            str(inference_path),
        ),
        (
            "Paired statistical comparison",
            (
                root
                / "statistics"
                / "paired_comparison.json"
            ).exists(),
            str(root / "statistics" / "paired_comparison.json"),
        ),
        (
            "Performance slices",
            (
                root
                / "slices"
                / "performance_slices.csv"
            ).exists(),
            str(root / "slices" / "performance_slices.csv"),
        ),
        (
            "Model registry",
            model_json.exists(),
            str(model_json),
        ),
        (
            "Production inference artifacts",
            inference_mask.exists()
            and inference_overlay.exists()
            and inference_log.exists(),
            f"{inference_mask}; {inference_overlay}; {inference_log}",
        ),
        (
            "Repository tests",
            pytest_exit_code == 0,
            f"pytest exit code={pytest_exit_code}",
        ),
        (
            "Docker runtime validation",
            docker_validation_complete,
            "docs/docker_validation/",
        ),
        (
            "Presentation",
            presentation_complete,
            "docs/presentation/",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["requirement", "complete", "evidence"],
    )


def validate_inference_evidence(
    *,
    mask_path: str | Path,
    overlay_path: str | Path,
    log_path: str | Path,
) -> dict[str, Any]:
    from PIL import Image
    import numpy as np

    mask_path = Path(mask_path)
    overlay_path = Path(overlay_path)
    log_path = Path(log_path)
    result = {
        "mask_exists": mask_path.exists(),
        "overlay_exists": overlay_path.exists(),
        "log_exists": log_path.exists(),
        "mask_binary": False,
        "log_has_success": False,
    }
    if mask_path.exists():
        with Image.open(mask_path) as raw:
            values = np.unique(np.asarray(raw.convert("L")))
        result["mask_binary"] = set(values.tolist()).issubset({0, 255})
    if log_path.exists():
        rows = [
            json.loads(line)
            for line in log_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        result["log_has_success"] = any(
            row.get("status") == "ok" for row in rows
        )
    result["complete"] = all(
        result[key]
        for key in (
            "mask_exists",
            "overlay_exists",
            "log_exists",
            "mask_binary",
            "log_has_success",
        )
    )
    return result
