from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from aereo_water.config import HPOConfig, TrainingConfig
from aereo_water.models.segformer import SegFormerSpec
from aereo_water.pipeline.failures import append_failure
from aereo_water.training.engine import train_segformer
from aereo_water.utils import (
    canonical_json_dumps,
    json_dump,
    sha256_dataframe,
    sha256_json,
)


SEARCH_SPACE_SPEC = {
    "learning_rate": {"type": "log_float", "low": 1e-5, "high": 1e-4},
    "weight_decay": {"type": "log_float", "low": 1e-6, "high": 1e-2},
    "ce_weight": {"type": "categorical", "choices": [0.4, 0.5, 0.6, 0.7]},
    "warmup_ratio": {"type": "categorical", "choices": [0.0, 0.05, 0.1]},
    "augmentation_profile": {
        "type": "categorical",
        "choices": ["light", "moderate"],
    },
}


def fixed_stratified_subset(
    frame: pd.DataFrame,
    *,
    size: int | None,
    seed: int,
) -> pd.DataFrame:
    if size is None or size >= len(frame):
        return frame.sort_values("image_id").reset_index(drop=True).copy()
    if size <= 0:
        raise ValueError("Subset size must be positive")

    source = frame.copy()
    try:
        source["_water_bin"] = pd.qcut(
            source["water_fraction"],
            q=min(10, max(2, size // 20)),
            labels=False,
            duplicates="drop",
        )
        sampled_parts = []
        grouped = list(source.groupby("_water_bin", dropna=False))
        for group_index, (_, group) in enumerate(grouped):
            group_size = max(
                1,
                int(round(size * len(group) / len(source))),
            )
            group_size = min(group_size, len(group))
            sampled_parts.append(
                group.sample(
                    n=group_size,
                    random_state=seed + group_index,
                )
            )
        sampled = pd.concat(sampled_parts, ignore_index=True)
        if len(sampled) > size:
            sampled = sampled.sample(n=size, random_state=seed)
        elif len(sampled) < size:
            remaining = source.loc[
                ~source["image_id"].isin(sampled["image_id"])
            ]
            extra = remaining.sample(
                n=size - len(sampled),
                random_state=seed,
            )
            sampled = pd.concat([sampled, extra], ignore_index=True)
    except (ValueError, TypeError):
        sampled = source.sample(n=size, random_state=seed)

    return (
        sampled.drop(columns=["_water_bin"], errors="ignore")
        .sort_values("image_id")
        .reset_index(drop=True)
    )


def trial_parameters(trial) -> dict[str, Any]:
    ce_weight = trial.suggest_categorical(
        "ce_weight",
        SEARCH_SPACE_SPEC["ce_weight"]["choices"],
    )
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            SEARCH_SPACE_SPEC["learning_rate"]["low"],
            SEARCH_SPACE_SPEC["learning_rate"]["high"],
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            SEARCH_SPACE_SPEC["weight_decay"]["low"],
            SEARCH_SPACE_SPEC["weight_decay"]["high"],
            log=True,
        ),
        "ce_weight": float(ce_weight),
        "dice_weight": float(1.0 - ce_weight),
        "warmup_ratio": trial.suggest_categorical(
            "warmup_ratio",
            SEARCH_SPACE_SPEC["warmup_ratio"]["choices"],
        ),
        "augmentation_profile": trial.suggest_categorical(
            "augmentation_profile",
            SEARCH_SPACE_SPEC["augmentation_profile"]["choices"],
        ),
    }


def config_from_parameters(
    base: TrainingConfig,
    parameters: dict[str, Any],
    *,
    epochs: int,
    seed: int | None = None,
    save_every_epoch: bool = False,
) -> TrainingConfig:
    config = copy.deepcopy(base)
    config.epochs = int(epochs)
    config.save_every_epoch = bool(save_every_epoch)
    if seed is not None:
        config.seed = int(seed)
    for name, value in parameters.items():
        if name == "dice_weight":
            continue
        if hasattr(config, name):
            setattr(config, name, value)
    config.dice_weight = float(
        parameters.get("dice_weight", 1.0 - config.ce_weight)
    )
    if abs(config.ce_weight + config.dice_weight - 1.0) > 1e-6:
        raise ValueError("CE and Dice weights must sum to one")
    return config


def build_study_fingerprint(
    *,
    git_commit: str,
    split_registry_sha256: str,
    model_id: str,
    image_size: int,
    resize_policy: str,
    hpo_train: pd.DataFrame,
    hpo_validation: pd.DataFrame,
    training_config: TrainingConfig,
    hpo_config: HPOConfig,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "git_commit": git_commit,
        "split_registry_sha256": split_registry_sha256,
        "model_id": model_id,
        "image_size": int(image_size),
        "resize_policy": resize_policy,
        "search_space": SEARCH_SPACE_SPEC,
        "training_contract": {
            "optimizer": "AdamW",
            "scheduler": "cosine_with_warmup",
            "batch_size": int(training_config.batch_size),
            "gradient_accumulation_steps": int(
                training_config.gradient_accumulation_steps
            ),
            "gradient_clip_norm": float(
                training_config.gradient_clip_norm
            ),
            "mixed_precision": bool(training_config.mixed_precision),
            "fixed_validation_threshold": float(
                training_config.fixed_validation_threshold
            ),
            "deterministic": bool(training_config.deterministic),
            "epochs_per_trial": int(hpo_config.epochs_per_trial),
        },
        "hpo_train_sha256": sha256_dataframe(
            hpo_train[["image_id", "split"]]
        ),
        "hpo_validation_sha256": sha256_dataframe(
            hpo_validation[["image_id", "split"]]
        ),
    }
    return sha256_json(payload), payload


def _completed_count(study) -> int:
    import optuna

    return sum(
        trial.state == optuna.trial.TrialState.COMPLETE
        for trial in study.trials
    )


def run_hpo(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    model_spec: SegFormerSpec,
    base_training_config: TrainingConfig,
    hpo_config: HPOConfig,
    image_size: int,
    resize_policy: str,
    output_dir: str | Path,
    device: torch.device | str,
    mlflow_tracking_uri: str,
    mlflow_artifact_root: str | Path,
    wandb_project: str | None,
    wandb_mode: str,
    wandb_root: str | Path,
    repo_git_commit: str,
    split_registry_sha256: str,
    baseline_parameters: dict[str, Any],
) -> tuple[Any, pd.DataFrame]:
    """Run until the requested number of completed trials or attempt limit."""
    import optuna

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trials_root = output / "trials"
    trials_root.mkdir(parents=True, exist_ok=True)
    failure_ledger = output / "failure_ledger.csv"

    hpo_train = fixed_stratified_subset(
        train_df,
        size=hpo_config.train_subset_size,
        seed=base_training_config.seed,
    )
    hpo_validation = fixed_stratified_subset(
        validation_df,
        size=hpo_config.validation_subset_size,
        seed=base_training_config.seed,
    )
    hpo_train.to_csv(output / "hpo_train_manifest.csv", index=False)
    hpo_validation.to_csv(
        output / "hpo_validation_manifest.csv",
        index=False,
    )

    fingerprint, fingerprint_payload = build_study_fingerprint(
        git_commit=repo_git_commit,
        split_registry_sha256=split_registry_sha256,
        model_id=model_spec.model_id,
        image_size=image_size,
        resize_policy=resize_policy,
        hpo_train=hpo_train,
        hpo_validation=hpo_validation,
        training_config=base_training_config,
        hpo_config=hpo_config,
    )

    study = optuna.create_study(
        study_name=hpo_config.study_name,
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=base_training_config.seed
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=4,
            n_warmup_steps=1,
        ),
        storage=f"sqlite:///{output / 'optuna.db'}",
        load_if_exists=True,
    )
    existing_fingerprint = study.user_attrs.get("study_fingerprint")
    if existing_fingerprint and existing_fingerprint != fingerprint:
        raise RuntimeError(
            "The existing Optuna study was created from a different code, "
            "data split, model, resize policy, or search space. Use a new "
            "study directory or remove the stale database."
        )
    study.set_user_attr("study_fingerprint", fingerprint)
    study.set_user_attr("fingerprint_payload", fingerprint_payload)
    study.set_user_attr(
        "objective",
        "mean per-image original-resolution validation IoU at threshold 0.5",
    )

    if not study.trials:
        baseline_for_optuna = {
            key: value
            for key, value in baseline_parameters.items()
            if key in SEARCH_SPACE_SPEC
        }
        study.enqueue_trial(baseline_for_optuna)

    def objective(trial) -> float:
        parameters = trial_parameters(trial)
        config = config_from_parameters(
            base_training_config,
            parameters,
            epochs=hpo_config.epochs_per_trial,
            save_every_epoch=False,
        )
        trial_dir = trials_root / f"trial_{trial.number:03d}"
        try:
            metadata = train_segformer(
                hpo_train,
                hpo_validation,
                model_spec=model_spec,
                config=config,
                image_size=image_size,
                resize_policy=resize_policy,
                output_dir=trial_dir,
                device=device,
                run_name=f"hpo_trial_{trial.number:03d}",
                experiment_name="water_segformer_hpo",
                mlflow_tracking_uri=mlflow_tracking_uri,
                mlflow_artifact_root=mlflow_artifact_root,
                wandb_project=wandb_project,
                wandb_mode=wandb_mode,
                wandb_root=wandb_root,
                repo_git_commit=repo_git_commit,
                split_registry_sha256=split_registry_sha256,
                optuna_trial=trial,
                reset_output=True,
            )
            for key in (
                "best_epoch",
                "best_validation_dice",
                "runtime_seconds",
                "peak_gpu_memory_mb",
                "mlflow_run_id",
                "mlflow_artifact_uri",
                "checkpoint_manifest_sha256",
            ):
                if key in metadata:
                    trial.set_user_attr(key, metadata[key])
            return float(metadata["best_validation_iou"])
        except optuna.TrialPruned:
            trial.set_user_attr("failure_category", "pruned_normally")
            raise
        except torch.cuda.OutOfMemoryError as exc:
            append_failure(
                failure_ledger,
                stage="hpo",
                error=exc,
                configuration=parameters,
                root_cause="GPU memory exhaustion",
                resolution=(
                    "Reduce batch size or use gradient accumulation; "
                    "do not change the test split."
                ),
            )
            trial.set_user_attr("failure_category", "oom")
            raise
        except FloatingPointError as exc:
            append_failure(
                failure_ledger,
                stage="hpo",
                error=exc,
                configuration=parameters,
                root_cause="Non-finite training loss",
                resolution="Review learning rate and input validity.",
            )
            trial.set_user_attr("failure_category", "nan_loss")
            raise
        except Exception as exc:
            append_failure(
                failure_ledger,
                stage="hpo",
                error=exc,
                configuration=parameters,
                root_cause="Unclassified trial failure",
            )
            trial.set_user_attr(
                "failure_category",
                type(exc).__name__,
            )
            raise
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    while (
        _completed_count(study) < hpo_config.completed_trials
        and len(study.trials) < hpo_config.maximum_attempts
    ):
        study.optimize(
            objective,
            n_trials=1,
            gc_after_trial=True,
            catch=(
                RuntimeError,
                FloatingPointError,
                ValueError,
                torch.cuda.OutOfMemoryError,
            ),
        )

    completed = _completed_count(study)
    if completed < hpo_config.completed_trials:
        raise RuntimeError(
            f"Only {completed} completed HPO trials were obtained after "
            f"{len(study.trials)} attempts. Inspect {failure_ledger}."
        )

    trials = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )
    trials.to_csv(output / "trials.csv", index=False)
    state_counts = trials["state"].astype(str).value_counts().to_dict()
    best_payload = {
        "study_name": study.study_name,
        "study_fingerprint": fingerprint,
        "objective": study.user_attrs["objective"],
        "completed_trials": int(completed),
        "attempted_trials": int(len(study.trials)),
        "trial_state_counts": state_counts,
        "best_trial": int(study.best_trial.number),
        "best_validation_iou": float(study.best_value),
        "best_parameters": {
            **study.best_params,
            "dice_weight": 1.0 - study.best_params["ce_weight"],
        },
        "training_images": int(len(hpo_train)),
        "validation_images": int(len(hpo_validation)),
        "test_split_used": False,
    }
    json_dump(best_payload, output / "best_trial.json")
    return study, trials


def _completed_trials_ranked(study) -> list[Any]:
    import optuna

    return sorted(
        [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ],
        key=lambda trial: float(trial.value),
        reverse=True,
    )


def confirm_baseline_and_top_trials(
    study,
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    model_spec: SegFormerSpec,
    base_training_config: TrainingConfig,
    baseline_parameters: dict[str, Any],
    top_k: int,
    confirmation_epochs: int,
    image_size: int,
    resize_policy: str,
    output_dir: str | Path,
    device: torch.device | str,
    mlflow_tracking_uri: str,
    mlflow_artifact_root: str | Path,
    wandb_project: str | None,
    wandb_mode: str,
    wandb_root: str | Path,
    repo_git_commit: str,
    split_registry_sha256: str,
) -> pd.DataFrame:
    """Always compare the same-code baseline with top HPO candidates."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    configurations: list[tuple[str, dict[str, Any], int | None]] = [
        ("same_code_historical_baseline", baseline_parameters, None)
    ]
    seen = {canonical_json_dumps(baseline_parameters)}
    unique_candidate_rank = 0
    for trial in _completed_trials_ranked(study):
        parameters = {
            **trial.params,
            "dice_weight": 1.0 - trial.params["ce_weight"],
        }
        fingerprint = canonical_json_dumps(parameters)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique_candidate_rank += 1
        configurations.append(
            (
                f"optuna_rank_{unique_candidate_rank}",
                parameters,
                int(trial.number),
            )
        )
        if unique_candidate_rank >= top_k:
            break

    if unique_candidate_rank < top_k:
        raise RuntimeError(
            f"Only {unique_candidate_rank} unique non-baseline HPO "
            f"configurations were available; {top_k} were requested."
        )

    rows: list[dict[str, Any]] = []
    for label, parameters, trial_number in configurations:
        config = config_from_parameters(
            base_training_config,
            parameters,
            epochs=confirmation_epochs,
            save_every_epoch=False,
        )
        run_dir = output / label
        metadata = train_segformer(
            train_df,
            validation_df,
            model_spec=model_spec,
            config=config,
            image_size=image_size,
            resize_policy=resize_policy,
            output_dir=run_dir,
            device=device,
            run_name=f"confirmation_{label}",
            experiment_name="water_segformer_confirmation",
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_root=mlflow_artifact_root,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_root=wandb_root,
            repo_git_commit=repo_git_commit,
            split_registry_sha256=split_registry_sha256,
            reset_output=True,
        )
        rows.append(
            {
                "label": label,
                "source_trial_number": trial_number,
                "parameters_json": canonical_json_dumps(parameters),
                "best_validation_iou": metadata[
                    "best_validation_iou"
                ],
                "best_validation_dice": metadata[
                    "best_validation_dice"
                ],
                "best_epoch": metadata["best_epoch"],
                "runtime_seconds": metadata["runtime_seconds"],
                "peak_gpu_memory_mb": metadata[
                    "peak_gpu_memory_mb"
                ],
                "checkpoint": metadata["best_checkpoint"],
                "mlflow_run_id": metadata["mlflow_run_id"],
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["best_validation_iou", "best_validation_dice"],
        ascending=False,
    )
    frame.to_csv(output / "confirmation_results.csv", index=False)
    return frame


def run_seed_stability(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    selected_parameters: dict[str, Any],
    seeds: list[int],
    epochs: int,
    model_spec: SegFormerSpec,
    base_training_config: TrainingConfig,
    image_size: int,
    resize_policy: str,
    output_dir: str | Path,
    device: torch.device | str,
    mlflow_tracking_uri: str,
    mlflow_artifact_root: str | Path,
    wandb_project: str | None,
    wandb_mode: str,
    wandb_root: str | Path,
    repo_git_commit: str,
    split_registry_sha256: str,
) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in seeds:
        config = config_from_parameters(
            base_training_config,
            selected_parameters,
            epochs=epochs,
            seed=seed,
            save_every_epoch=False,
        )
        metadata = train_segformer(
            train_df,
            validation_df,
            model_spec=model_spec,
            config=config,
            image_size=image_size,
            resize_policy=resize_policy,
            output_dir=output / f"seed_{seed}",
            device=device,
            run_name=f"stability_seed_{seed}",
            experiment_name="water_segformer_stability",
            mlflow_tracking_uri=mlflow_tracking_uri,
            mlflow_artifact_root=mlflow_artifact_root,
            wandb_project=wandb_project,
            wandb_mode=wandb_mode,
            wandb_root=wandb_root,
            repo_git_commit=repo_git_commit,
            split_registry_sha256=split_registry_sha256,
            reset_output=True,
        )
        rows.append(
            {
                "seed": seed,
                "best_validation_iou": metadata[
                    "best_validation_iou"
                ],
                "best_validation_dice": metadata[
                    "best_validation_dice"
                ],
                "best_epoch": metadata["best_epoch"],
                "checkpoint": metadata["best_checkpoint"],
                "mlflow_run_id": metadata["mlflow_run_id"],
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "seed_stability.csv", index=False)
    summary = {
        "seeds": seeds,
        "mean_validation_iou": float(
            frame["best_validation_iou"].mean()
        ),
        "std_validation_iou": float(
            frame["best_validation_iou"].std(ddof=1)
            if len(frame) > 1
            else 0.0
        ),
        "minimum_validation_iou": float(
            frame["best_validation_iou"].min()
        ),
        "maximum_validation_iou": float(
            frame["best_validation_iou"].max()
        ),
    }
    json_dump(summary, output / "seed_stability_summary.json")
    return frame
