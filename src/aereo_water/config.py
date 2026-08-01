from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    image_size: int = 512
    resize_policy: str = "letterbox"
    num_workers: int = 2
    train_count: int = 1991
    validation_count: int = 429
    test_count: int = 421
    split_seed: int = 42
    tile_size: int = 1024
    tile_overlap: int = 128
    near_duplicate_hamming_threshold: int = 4


@dataclass
class ModelConfig:
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    num_labels: int = 2
    id2label: dict[int, str] = field(
        default_factory=lambda: {0: "non_water", 1: "water"}
    )
    label2id: dict[str, int] = field(
        default_factory=lambda: {"non_water": 0, "water": 1}
    )


@dataclass
class TrainingConfig:
    seed: int = 42
    epochs: int = 15
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 6e-5
    weight_decay: float = 1e-4
    ce_weight: float = 0.6
    dice_weight: float = 0.4
    warmup_ratio: float = 0.1
    gradient_clip_norm: float = 1.0
    mixed_precision: bool = True
    early_stopping_patience: int = 4
    fixed_validation_threshold: float = 0.5
    augmentation_profile: str = "moderate"
    save_every_epoch: bool = True
    deterministic: bool = True


@dataclass
class HPOConfig:
    completed_trials: int = 12
    maximum_attempts: int = 20
    epochs_per_trial: int = 4
    train_subset_size: int = 1000
    validation_subset_size: int | None = None
    confirmation_top_k: int = 3
    confirmation_epochs: int = 6
    stability_seeds: list[int] = field(
        default_factory=lambda: [42, 2026, 3407]
    )
    study_name: str = "water_segformer_hpo"


@dataclass
class EvaluationConfig:
    batch_size: int = 4
    include_boundary_metrics: bool = True
    boundary_tolerance_pixels: int = 2
    threshold_minimum: float = 0.10
    threshold_maximum: float = 0.90
    threshold_step: float = 0.05
    empty_mask_policy: str = "perfect_if_both_empty"
    calibration_bins: int = 15
    bootstrap_iterations: int = 5000


@dataclass
class TrackingConfig:
    mlflow_experiment_hpo: str = "water_segformer_hpo"
    mlflow_experiment_confirmation: str = "water_segformer_confirmation"
    mlflow_experiment_final: str = "water_segformer_final"
    mlflow_experiment_evaluation: str = "water_segformer_evaluation"
    wandb_project: str = "aereo-water-segmentation"
    wandb_mode: str = "offline"


@dataclass
class PipelineConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    hpo: HPOConfig = field(default_factory=HPOConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(path: str | Path) -> PipelineConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return PipelineConfig(
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        training=TrainingConfig(**payload.get("training", {})),
        hpo=HPOConfig(**payload.get("hpo", {})),
        evaluation=EvaluationConfig(**payload.get("evaluation", {})),
        tracking=TrackingConfig(**payload.get("tracking", {})),
    )
