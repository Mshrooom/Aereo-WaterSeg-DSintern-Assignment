from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm.auto import tqdm

from waterseg.losses import CombinedSegmentationLoss
from waterseg.metrics import BinarySegmentationMeter, ThresholdSweep
from waterseg.models.sam_water import SamWaterModel
from waterseg.prompting import build_prompt_batch
from waterseg.utils import stable_int_hash, write_json

LOGGER = logging.getLogger(__name__)


def _cosine_schedule(step: int, total_steps: int, warmup_steps: int) -> float:
    if step < warmup_steps:
        return float(step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _sample_prompt_mode(modes: List[str], weights: List[float], rng: np.random.Generator) -> str:
    normalized = np.asarray(weights, dtype=np.float64)
    normalized /= normalized.sum()
    return str(rng.choice(modes, p=normalized))


def train_one_epoch(
    model: SamWaterModel,
    loader: Iterable[dict],
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.cuda.amp.GradScaler,
    loss_function: CombinedSegmentationLoss,
    device: torch.device,
    cfg: Any,
    epoch: int,
) -> Dict[str, float]:
    model.train()
    running: Dict[str, float] = {}
    optimizer.zero_grad(set_to_none=True)
    epoch_rng = np.random.default_rng(cfg.train.seed + epoch)
    batches = 0

    for batch_index, batch in enumerate(tqdm(loader, desc=f"train {epoch}", leave=False)):
        mode = _sample_prompt_mode(cfg.prompts.train_modes, cfg.prompts.train_weights, epoch_rng)
        rngs = [np.random.default_rng(stable_int_hash(image_id, cfg.train.seed + epoch)) for image_id in batch["image_ids"]]
        prompts = build_prompt_batch(
            batch["masks"], mode, cfg.prompts.positive_points, cfg.prompts.negative_points,
            cfg.prompts.box_jitter_fraction, rngs,
        )
        inputs = model.prepare_inputs(batch["images"], prompts, device)
        with torch.autocast(device_type=device.type, enabled=cfg.train.amp and device.type == "cuda"):
            logits, predicted_iou = model.forward_prepared(inputs, cfg.model.multimask_output)
            targets = model.resize_targets(batch["masks"], logits.shape[-2:], device)
            loss, components = loss_function(logits, targets, predicted_iou)
            scaled_loss = loss / cfg.train.gradient_accumulation_steps
        scaler.scale(scaled_loss).backward()

        should_step = (batch_index + 1) % cfg.train.gradient_accumulation_steps == 0 or (batch_index + 1) == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        for key, value in components.items():
            running[key] = running.get(key, 0.0) + value
        batches += 1

    return {f"train_{key}": value / max(batches, 1) for key, value in running.items()}


@torch.inference_mode()
def predict_loader(
    model: SamWaterModel,
    loader: Iterable[dict],
    device: torch.device,
    cfg: Any,
    prompt_mode: str,
    threshold: float,
    threshold_sweep: ThresholdSweep | None = None,
    compute_surface: bool = False,
) -> tuple[Dict[str, float], pd.DataFrame]:
    model.eval()
    meter = BinarySegmentationMeter(threshold=threshold, compute_surface=compute_surface)
    for batch in tqdm(loader, desc=f"eval {prompt_mode}", leave=False):
        rngs = [np.random.default_rng(stable_int_hash(f"{image_id}:{prompt_mode}", cfg.train.seed)) for image_id in batch["image_ids"]]
        prompts = build_prompt_batch(
            batch["masks"], prompt_mode, cfg.prompts.positive_points, cfg.prompts.negative_points,
            cfg.prompts.box_jitter_fraction, rngs,
        )
        inputs = model.prepare_inputs(batch["images"], prompts, device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits, _ = model.forward_prepared(inputs, cfg.model.multimask_output)
        probabilities_low = torch.sigmoid(logits)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latency_per_image = elapsed_ms / len(batch["images"])

        processed = model.processor.image_processor.post_process_masks(
            probabilities_low,
            inputs["original_sizes"],
            inputs["reshaped_input_sizes"],
            binarize=False,
        )
        for probability_tensor, target, image_id in zip(processed, batch["masks"], batch["image_ids"]):
            probability = probability_tensor.squeeze().detach().cpu().numpy().astype(np.float32)
            meter.update(probability, target, image_id, latency_per_image)
            if threshold_sweep is not None:
                threshold_sweep.update(probability, target)
    return meter.compute(), meter.dataframe()


def fit(
    model: SamWaterModel,
    train_loader: Iterable[dict],
    val_loader: Iterable[dict],
    device: torch.device,
    cfg: Any,
    tracker: Any,
    output_dir: str | Path,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    optimizer = AdamW(
        model.trainable_parameter_groups(cfg.train.learning_rate, cfg.train.encoder_learning_rate, cfg.train.weight_decay)
    )
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / cfg.train.gradient_accumulation_steps)
    total_steps = max(cfg.train.epochs * optimizer_steps_per_epoch, 1)
    warmup_steps = int(total_steps * cfg.train.warmup_fraction)
    scheduler = LambdaLR(optimizer, lambda step: _cosine_schedule(step, total_steps, warmup_steps))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")
    loss_function = CombinedSegmentationLoss(
        cfg.train.bce_weight, cfg.train.dice_weight, cfg.train.focal_weight, cfg.train.iou_head_weight
    )

    best_score = -float("inf")
    best_epoch = -1
    stale_epochs = 0
    start_epoch = 1
    history_path = output_dir / "training_history.csv"
    history = pd.read_csv(history_path).to_dict("records") if history_path.exists() else []
    thresholds = np.linspace(cfg.train.threshold_min, cfg.train.threshold_max, cfg.train.num_thresholds)

    if cfg.train.resume_checkpoint:
        resume_checkpoint = Path(cfg.train.resume_checkpoint)
        LOGGER.info("Resuming model weights from %s", resume_checkpoint)
        model.load_trainable_checkpoint(resume_checkpoint, strict=False)
        state_path = resume_checkpoint.with_name("training_state_last.pt")
        if state_path.exists():
            state = torch.load(state_path, map_location="cpu", weights_only=False)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state.get("scaler", {}))
            start_epoch = int(state["epoch"]) + 1
            best_score = float(state.get("best_score", best_score))
            best_epoch = int(state.get("best_epoch", best_epoch))
            stale_epochs = int(state.get("stale_epochs", stale_epochs))
            if "python_rng_state" in state:
                random.setstate(state["python_rng_state"])
            if "numpy_rng_state" in state:
                np.random.set_state(state["numpy_rng_state"])
            if "torch_rng_state" in state:
                torch.set_rng_state(state["torch_rng_state"])
            if device.type == "cuda" and state.get("cuda_rng_state") is not None:
                torch.cuda.set_rng_state_all(state["cuda_rng_state"])
            LOGGER.info("Resuming at epoch %d", start_epoch)
        else:
            LOGGER.warning("No optimizer state found at %s; resuming weights only", state_path)

    for epoch in range(start_epoch, cfg.train.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, loss_function, device, cfg, epoch)
        sweep = ThresholdSweep(thresholds)
        val_metrics, _ = predict_loader(
            model, val_loader, device, cfg, cfg.train.primary_prompt_mode, 0.5, threshold_sweep=sweep
        )
        best_threshold = sweep.best(cfg.train.monitor_metric)
        record = {
            "epoch": epoch,
            **train_metrics,
            **{f"val_{key}": value for key, value in val_metrics.items()},
            "val_best_threshold": best_threshold["threshold"],
            f"val_threshold_{cfg.train.monitor_metric}": best_threshold[cfg.train.monitor_metric],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        tracker.log(record, step=epoch)
        pd.DataFrame(history).to_csv(history_path, index=False)
        sweep.table().to_csv(output_dir / f"threshold_sweep_epoch_{epoch:02d}.csv", index=False)

        score = best_threshold[cfg.train.monitor_metric]
        metadata = {
            "epoch": epoch,
            "threshold": best_threshold["threshold"],
            "val_metrics": val_metrics,
            "prompt_mode": cfg.train.primary_prompt_mode,
            "config": cfg.to_dict(),
        }
        model.save_trainable_checkpoint(checkpoints_dir / "last.pt", metadata)
        if score > best_score:
            best_score, best_epoch, stale_epochs = score, epoch, 0
            model.save_trainable_checkpoint(checkpoints_dir / "best.pt", metadata)
            write_json(output_dir / "best_model.json", metadata)
        else:
            stale_epochs += 1

        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_score": best_score,
                "best_epoch": best_epoch,
                "stale_epochs": stale_epochs,
                "python_rng_state": random.getstate(),
                "numpy_rng_state": np.random.get_state(),
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
            },
            checkpoints_dir / "training_state_last.pt",
        )
        if stale_epochs >= cfg.train.patience:
            LOGGER.info("Early stopping at epoch %d; best epoch was %d", epoch, best_epoch)
            break

    result = {"best_epoch": best_epoch, "best_score": best_score, "checkpoint": str(checkpoints_dir / "best.pt")}
    write_json(output_dir / "training_summary.json", result)
    return result
