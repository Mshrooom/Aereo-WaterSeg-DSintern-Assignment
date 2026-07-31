from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml


@dataclass
class PathsConfig:
    dataset_root: str = "/kaggle/input/satellite-images-of-water-bodies/Water Bodies Dataset"
    output_dir: str = "/kaggle/working/aereo-water-sam-output"
    manifest_path: str = ""


@dataclass
class DataConfig:
    images_dir_name: str = "Images"
    masks_dir_name: str = "Masks"
    extensions: List[str] = field(default_factory=lambda: [".jpg", ".jpeg", ".png", ".tif", ".tiff"])
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    stratification_bins: int = 10
    num_workers: int = 2
    verify_files: bool = True
    tile_size: int = 1024
    tile_overlap: int = 128
    materialize_tiles: bool = False


@dataclass
class ModelConfig:
    model_id: str = "facebook/sam-vit-base"
    trainable_parts: str = "mask_decoder"
    unfreeze_last_vision_blocks: int = 0
    multimask_output: bool = False


@dataclass
class PromptConfig:
    train_modes: List[str] = field(
        default_factory=lambda: ["none", "point1", "points", "box", "box_points"]
    )
    train_weights: List[float] = field(default_factory=lambda: [0.35, 0.15, 0.15, 0.20, 0.15])
    eval_modes: List[str] = field(
        default_factory=lambda: ["none", "point1", "points", "box", "box_points"]
    )
    positive_points: int = 3
    negative_points: int = 1
    box_jitter_fraction: float = 0.08


@dataclass
class TrainConfig:
    seed: int = 42
    epochs: int = 12
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    encoder_learning_rate: float = 1e-6
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.08
    max_grad_norm: float = 1.0
    amp: bool = True
    patience: int = 4
    num_thresholds: int = 17
    threshold_min: float = 0.10
    threshold_max: float = 0.90
    primary_prompt_mode: str = "none"
    monitor_metric: str = "iou"
    bce_weight: float = 0.35
    dice_weight: float = 0.45
    focal_weight: float = 0.20
    iou_head_weight: float = 0.10
    resume_checkpoint: str = ""


@dataclass
class TrackingConfig:
    provider: str = "local"
    project: str = "aereo-water-sam"
    entity: str = ""
    run_name: str = "sam-vit-b-water"
    wandb_mode: str = "offline"
    log_images: int = 8


@dataclass
class InferenceConfig:
    threshold: float = 0.50
    use_tiling: bool = True
    tile_size: int = 1024
    tile_overlap: int = 128
    min_component_area: int = 0
    fill_holes: bool = False


@dataclass
class SemanticConfig:
    model_id: str = "nvidia/segformer-b0-finetuned-ade-512-512"
    image_size: int = 512
    epochs: int = 12
    batch_size: int = 8
    learning_rate: float = 6e-5
    weight_decay: float = 1e-4
    patience: int = 4
    ce_weight: float = 0.60
    dice_weight: float = 0.40


@dataclass
class AutoSamConfig:
    coarse_threshold: float = 0.50
    max_positive_points: int = 3
    negative_points: int = 1
    min_component_area: int = 16
    box_padding_fraction: float = 0.04
    morphology_kernel: int = 3
    sam_weight: float = 0.75


@dataclass
class ExportConfig:
    save_masks: bool = True
    save_probability_maps: bool = True
    resume: bool = True


@dataclass
class ExperimentConfig:
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    prompts: PromptConfig = field(default_factory=PromptConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    auto_sam: AutoSamConfig = field(default_factory=AutoSamConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _merge_dataclass(instance: Any, values: Dict[str, Any]) -> Any:
    for key, value in values.items():
        if not hasattr(instance, key):
            raise KeyError(f"Unknown configuration key: {key}")
        setattr(instance, key, value)
    return instance


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    cfg = ExperimentConfig()
    for section, values in raw.items():
        if not hasattr(cfg, section):
            raise KeyError(f"Unknown configuration section: {section}")
        _merge_dataclass(getattr(cfg, section), values or {})

    total = cfg.data.train_fraction + cfg.data.val_fraction + cfg.data.test_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError("train_fraction + val_fraction + test_fraction must equal 1.0")
    if len(cfg.prompts.train_modes) != len(cfg.prompts.train_weights):
        raise ValueError("train_modes and train_weights must have equal lengths")
    return cfg
