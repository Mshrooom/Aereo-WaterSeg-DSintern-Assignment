from __future__ import annotations

import copy
import gc
import json
import math
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from aereo_water.config import TrainingConfig
from aereo_water.data.dataset import SegmentationTrainingDataset
from aereo_water.evaluation.validation import validate_original_resolution
from aereo_water.models.segformer import (
    SegFormerSpec,
    build_segformer,
)
from aereo_water.tracking import DualTracker
from aereo_water.utils import (
    capture_rng_state,
    json_dump,
    restore_rng_state,
    seed_everything,
    sha256_dataframe,
    sha256_file,
    utc_now_iso,
)


def combined_cross_entropy_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    ce_weight: float,
    dice_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ignore_index = 255
    cross_entropy = F.cross_entropy(
        logits,
        targets,
        ignore_index=ignore_index,
    )
    water_probability = torch.softmax(logits, dim=1)[:, 1]
    valid = (targets != ignore_index).float()
    target_float = (targets == 1).float()
    masked_probability = water_probability * valid
    masked_target = target_float * valid
    intersection = (masked_probability * masked_target).sum(dim=(1, 2))
    denominator = (
        masked_probability.sum(dim=(1, 2))
        + masked_target.sum(dim=(1, 2))
    )
    dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)
    dice_loss = 1.0 - dice.mean()
    total = ce_weight * cross_entropy + dice_weight * dice_loss
    return total, {
        "cross_entropy": cross_entropy.detach(),
        "dice_loss": dice_loss.detach(),
    }


def build_adamw_optimizer(
    model: torch.nn.Module,
    *,
    learning_rate: float,
    weight_decay: float,
) -> tuple[torch.optim.Optimizer, dict[str, int]]:
    """Exclude bias and normalization parameters from weight decay."""
    decay_parameters: list[torch.nn.Parameter] = []
    no_decay_parameters: list[torch.nn.Parameter] = []
    no_decay_terms = (
        "bias",
        "layernorm.weight",
        "layer_norm.weight",
        "batchnorm.weight",
        "batch_norm.weight",
        "norm.weight",
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        lowered = name.lower()
        if parameter.ndim == 1 or any(term in lowered for term in no_decay_terms):
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    optimizer = torch.optim.AdamW(
        [
            {
                "params": decay_parameters,
                "weight_decay": weight_decay,
            },
            {
                "params": no_decay_parameters,
                "weight_decay": 0.0,
            },
        ],
        lr=learning_rate,
    )
    summary = {
        "decay_parameter_tensors": len(decay_parameters),
        "no_decay_parameter_tensors": len(no_decay_parameters),
        "decay_parameters": int(
            sum(parameter.numel() for parameter in decay_parameters)
        ),
        "no_decay_parameters": int(
            sum(parameter.numel() for parameter in no_decay_parameters)
        ),
    }
    return optimizer, summary


def _save_huggingface_checkpoint(
    model,
    processor,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)


def _save_resume_state(
    path: Path,
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    scaler,
    data_loader_generator: torch.Generator,
    best_iou: float,
    best_validation_dice: float,
    best_epoch: int,
    early_stopping_counter: int,
    history: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "data_loader_generator_state": (
                data_loader_generator.get_state()
            ),
            "best_iou": float(best_iou),
            "best_validation_dice": float(best_validation_dice),
            "best_epoch": int(best_epoch),
            "early_stopping_counter": int(early_stopping_counter),
            "history": history,
            "rng_state": capture_rng_state(),
        },
        path,
    )


def _load_resume_state(
    path: Path,
    *,
    model,
    optimizer,
    scheduler,
    scaler,
    data_loader_generator: torch.Generator,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(path, map_location=device)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    scaler.load_state_dict(payload["scaler_state_dict"])
    if "data_loader_generator_state" in payload:
        data_loader_generator.set_state(
            payload["data_loader_generator_state"]
        )
    restore_rng_state(payload["rng_state"])
    return payload


def _checkpoint_weights_path(checkpoint_dir: Path) -> Path | None:
    for filename in ("model.safetensors", "pytorch_model.bin"):
        path = checkpoint_dir / filename
        if path.exists():
            return path
    return None


def train_segformer(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    *,
    model_spec: SegFormerSpec,
    config: TrainingConfig,
    image_size: int,
    resize_policy: str,
    output_dir: str | Path,
    device: torch.device | str,
    run_name: str,
    experiment_name: str,
    mlflow_tracking_uri: str,
    mlflow_artifact_root: str | Path,
    wandb_project: str | None,
    wandb_mode: str,
    wandb_root: str | Path,
    repo_git_commit: str,
    split_registry_sha256: str,
    resume_from: str | Path | None = None,
    optuna_trial=None,
    reset_output: bool = False,
) -> dict[str, Any]:
    """Train SegFormer with original-resolution model selection and resumption."""
    from transformers import get_cosine_schedule_with_warmup

    output = Path(output_dir)
    if reset_output and output.exists() and resume_from is None:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    best_checkpoint = output / "best_checkpoint"
    last_checkpoint = output / "last_checkpoint"
    resume_state_path = output / "last_state.pt"
    history_path = output / "history.csv"
    metadata_path = output / "training_metadata.json"

    seed_everything(config.seed, deterministic=config.deterministic)
    device = torch.device(device)
    model, processor = build_segformer(model_spec)
    model.to(device)

    train_dataset = SegmentationTrainingDataset(
        train_df,
        processor,
        image_size=image_size,
        resize_policy=resize_policy,
        augmentation_profile=config.augmentation_profile,
        base_seed=config.seed,
    )
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    optimizer, optimizer_group_summary = build_adamw_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    optimizer_steps_per_epoch = math.ceil(
        len(train_loader) / config.gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_optimizer_steps,
    )
    amp_enabled = bool(
        config.mixed_precision and device.type == "cuda"
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 1
    best_iou = -math.inf
    best_epoch = 0
    best_validation_dice = float("nan")
    early_stopping_counter = 0
    history: list[dict[str, Any]] = []

    resume_candidate = Path(resume_from) if resume_from else None
    if resume_candidate is not None:
        if not resume_candidate.exists():
            raise FileNotFoundError(resume_candidate)
        resumed = _load_resume_state(
            resume_candidate,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            data_loader_generator=generator,
            device=device,
        )
        start_epoch = int(resumed["epoch"]) + 1
        best_iou = float(resumed["best_iou"])
        best_validation_dice = float(
            resumed.get("best_validation_dice", float("nan"))
        )
        best_epoch = int(resumed["best_epoch"])
        early_stopping_counter = int(
            resumed["early_stopping_counter"]
        )
        history = list(resumed["history"])

    train_manifest_hash = sha256_dataframe(
        train_df[["image_id", "split"]],
    )
    validation_manifest_hash = sha256_dataframe(
        validation_df[["image_id", "split"]],
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    training_started = time.perf_counter()

    tags = {
        "pipeline": "segformer-v3",
        "run_type": experiment_name,
        "git_commit": repo_git_commit,
    }
    with DualTracker(
        run_name=run_name,
        experiment_name=experiment_name,
        mlflow_tracking_uri=mlflow_tracking_uri,
        mlflow_artifact_root=mlflow_artifact_root,
        wandb_project=wandb_project,
        wandb_mode=wandb_mode,
        wandb_root=wandb_root,
        wandb_group=experiment_name,
        tags=tags,
    ) as tracker:
        tracker.log_params(
            {
                "model_id": model_spec.model_id,
                "seed": config.seed,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": (
                    config.gradient_accumulation_steps
                ),
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "ce_weight": config.ce_weight,
                "dice_weight": config.dice_weight,
                "warmup_ratio": config.warmup_ratio,
                "gradient_clip_norm": config.gradient_clip_norm,
                "mixed_precision": amp_enabled,
                "early_stopping_patience": (
                    config.early_stopping_patience
                ),
                "fixed_validation_threshold": (
                    config.fixed_validation_threshold
                ),
                "augmentation_profile": config.augmentation_profile,
                "image_size": image_size,
                "resize_policy": resize_policy,
                "training_images": len(train_df),
                "validation_images": len(validation_df),
                "train_manifest_sha256": train_manifest_hash,
                "validation_manifest_sha256": validation_manifest_hash,
                "split_registry_sha256": split_registry_sha256,
                "git_commit": repo_git_commit,
                **optimizer_group_summary,
            }
        )

        try:
            for epoch in range(start_epoch, config.epochs + 1):
                if (
                    early_stopping_counter
                    >= config.early_stopping_patience
                ):
                    break
                epoch_started = time.perf_counter()
                train_dataset.set_epoch(epoch)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                train_losses: list[float] = []
                ce_losses: list[float] = []
                dice_losses: list[float] = []
                gradient_norms: list[float] = []
                optimizer_steps = 0
                examples_seen = 0

                for batch_index, batch in enumerate(train_loader, start=1):
                    pixel_values = batch["pixel_values"].to(
                        device,
                        non_blocking=True,
                    )
                    labels = batch["labels"].to(
                        device,
                        non_blocking=True,
                    )
                    examples_seen += int(len(labels))

                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        logits = model(pixel_values=pixel_values).logits
                        logits = F.interpolate(
                            logits,
                            size=labels.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                        loss, components = combined_cross_entropy_dice_loss(
                            logits,
                            labels,
                            ce_weight=config.ce_weight,
                            dice_weight=config.dice_weight,
                        )
                        scaled_loss = (
                            loss / config.gradient_accumulation_steps
                        )

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"Non-finite loss at epoch {epoch}, "
                            f"batch {batch_index}: {loss.item()}"
                        )

                    scaler.scale(scaled_loss).backward()
                    should_step = (
                        batch_index % config.gradient_accumulation_steps == 0
                        or batch_index == len(train_loader)
                    )
                    if should_step:
                        scaler.unscale_(optimizer)
                        gradient_norm = torch.nn.utils.clip_grad_norm_(
                            model.parameters(),
                            max_norm=config.gradient_clip_norm,
                        )
                        gradient_norms.append(float(gradient_norm))
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)
                        scheduler.step()
                        optimizer_steps += 1

                    train_losses.append(float(loss.detach().cpu()))
                    ce_losses.append(
                        float(components["cross_entropy"].cpu())
                    )
                    dice_losses.append(
                        float(components["dice_loss"].cpu())
                    )

                validation_started = time.perf_counter()
                validation_metrics = validate_original_resolution(
                    model,
                    processor,
                    validation_df,
                    image_size=image_size,
                    resize_policy=resize_policy,
                    threshold=config.fixed_validation_threshold,
                    device=device,
                    batch_size=config.batch_size,
                    num_workers=0,
                    empty_policy="perfect_if_both_empty",
                )
                validation_seconds = (
                    time.perf_counter() - validation_started
                )
                epoch_seconds = time.perf_counter() - epoch_started
                current_iou = validation_metrics["val_original_iou"]
                current_dice = validation_metrics["val_original_dice"]
                improved = current_iou > best_iou + 1e-8

                if improved:
                    best_iou = current_iou
                    best_validation_dice = current_dice
                    best_epoch = epoch
                    early_stopping_counter = 0
                    _save_huggingface_checkpoint(
                        model,
                        processor,
                        best_checkpoint,
                    )
                else:
                    early_stopping_counter += 1

                if config.save_every_epoch or epoch == config.epochs:
                    _save_huggingface_checkpoint(
                        model,
                        processor,
                        last_checkpoint,
                    )

                row = {
                    "epoch": int(epoch),
                    "train_loss": float(np.mean(train_losses)),
                    "train_cross_entropy": float(np.mean(ce_losses)),
                    "train_dice_loss": float(np.mean(dice_losses)),
                    **validation_metrics,
                    "learning_rate": float(
                        optimizer.param_groups[0]["lr"]
                    ),
                    "gradient_norm_mean": float(
                        np.mean(gradient_norms)
                        if gradient_norms
                        else 0.0
                    ),
                    "optimizer_steps": int(optimizer_steps),
                    "examples_seen": int(examples_seen),
                    "examples_per_second": float(
                        examples_seen / max(epoch_seconds, 1e-9)
                    ),
                    "epoch_seconds": float(epoch_seconds),
                    "validation_seconds": float(validation_seconds),
                    "early_stopping_counter": int(
                        early_stopping_counter
                    ),
                    "gpu_allocated_mb": float(
                        torch.cuda.memory_allocated() / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    ),
                    "gpu_reserved_mb": float(
                        torch.cuda.memory_reserved() / (1024**2)
                        if device.type == "cuda"
                        else 0.0
                    ),
                    "improved": bool(improved),
                }
                history.append(row)
                pd.DataFrame(history).to_csv(history_path, index=False)
                _save_resume_state(
                    resume_state_path,
                    epoch=epoch,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    data_loader_generator=generator,
                    best_iou=best_iou,
                    best_validation_dice=best_validation_dice,
                    best_epoch=best_epoch,
                    early_stopping_counter=early_stopping_counter,
                    history=history,
                )
                tracker.log_metrics(
                    {
                        key: value
                        for key, value in row.items()
                        if isinstance(value, (int, float))
                    },
                    step=epoch,
                )

                if optuna_trial is not None:
                    optuna_trial.report(current_iou, step=epoch)
                    if optuna_trial.should_prune():
                        tracker.mark_pruned()
                        import optuna

                        raise optuna.TrialPruned(
                            f"Pruned at epoch {epoch}; "
                            f"validation IoU={current_iou:.4f}"
                        )

                if (
                    early_stopping_counter
                    >= config.early_stopping_patience
                ):
                    break
        finally:
            runtime_seconds = time.perf_counter() - training_started
            peak_gpu_memory_mb = float(
                torch.cuda.max_memory_allocated() / (1024**2)
                if device.type == "cuda"
                else 0.0
            )
            weights = _checkpoint_weights_path(best_checkpoint)
            checkpoint_hash = sha256_file(weights) if weights else ""
            metadata = {
                "schema_version": 2,
                "run_name": run_name,
                "experiment_name": experiment_name,
                "model_id": model_spec.model_id,
                "best_epoch": int(best_epoch),
                "best_validation_iou": float(best_iou),
                "best_validation_dice": float(best_validation_dice),
                "fixed_validation_threshold": float(
                    config.fixed_validation_threshold
                ),
                "selection_resolution": "original",
                "selection_metric": (
                    "mean per-image original-resolution validation IoU"
                ),
                "training_images": int(len(train_df)),
                "validation_images": int(len(validation_df)),
                "runtime_seconds": float(runtime_seconds),
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
                "checkpoint_manifest_sha256": checkpoint_hash,
                "best_checkpoint": str(best_checkpoint),
                "last_checkpoint": str(last_checkpoint),
                "resume_state": str(resume_state_path),
                "mlflow_run_id": tracker.mlflow_run_id,
                "mlflow_artifact_uri": tracker.mlflow_artifact_uri,
                "git_commit": repo_git_commit,
                "split_registry_sha256": split_registry_sha256,
                "created_at_utc": utc_now_iso(),
            }
            json_dump(metadata, metadata_path)
            if history_path.exists():
                tracker.log_artifact(
                    history_path,
                    artifact_path="tables",
                    wandb_artifact_name=f"{run_name}-history",
                    wandb_artifact_type="training-history",
                )
            if best_checkpoint.exists() and any(best_checkpoint.iterdir()):
                tracker.log_artifact(
                    best_checkpoint,
                    artifact_path="checkpoints/best",
                    wandb_artifact_name=f"{run_name}-best-checkpoint",
                    wandb_artifact_type="model",
                )
            tracker.log_dict(metadata, "metadata/training_metadata.json")
            tracker.log_metrics(
                {
                    "best_validation_iou": (
                        best_iou if math.isfinite(best_iou) else -1.0
                    ),
                    "best_validation_dice": (
                        best_validation_dice
                        if math.isfinite(best_validation_dice)
                        else -1.0
                    ),
                    "best_epoch": best_epoch,
                    "runtime_seconds": runtime_seconds,
                    "peak_gpu_memory_mb": peak_gpu_memory_mb,
                }
            )

    del model, processor, optimizer, scheduler, scaler, train_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metadata
