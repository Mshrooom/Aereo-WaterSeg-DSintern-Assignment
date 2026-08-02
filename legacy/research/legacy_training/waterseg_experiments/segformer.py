from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from waterseg.data.manifest import read_mask, read_rgb
from waterseg.data.transforms import JointTransform
from waterseg.experiments.common import (
    append_rows, completed_keys, per_image_result, save_binary_mask, save_probability_u16,
)
from waterseg.metrics import ThresholdSweep
from waterseg.models.segformer_water import SegformerWaterModel, preprocess_segformer_image
from waterseg.utils import seed_everything, write_json


class SegformerDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, image_size: int, training: bool):
        self.manifest = manifest.reset_index(drop=True)
        self.image_size = image_size
        self.transform = JointTransform(training=training)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict:
        row = self.manifest.iloc[index]
        image = read_rgb(row.image_path)
        target = read_mask(row.mask_path)
        transformed = self.transform(image=image, mask=target)
        image, target = transformed["image"], transformed["mask"]
        pixel_values = preprocess_segformer_image(image, self.image_size)
        resized_target = cv2.resize(target, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return {
            "pixel_values": pixel_values,
            "labels": torch.from_numpy(resized_target.astype(np.int64)),
            "image_id": str(row.image_id),
            "image_path": str(row.image_path),
            "mask_path": str(row.mask_path),
            "original_target": target.astype(np.uint8),
        }


def collate_segformer(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "image_ids": [item["image_id"] for item in batch],
        "image_paths": [item["image_path"] for item in batch],
        "mask_paths": [item["mask_path"] for item in batch],
        "targets": [item["original_target"] for item in batch],
    }


def _loss(logits: torch.Tensor, labels: torch.Tensor, ce_weight: float, dice_weight: float) -> tuple[torch.Tensor, dict]:
    logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
    ce = F.cross_entropy(logits, labels)
    probability = torch.softmax(logits, dim=1)[:, 1]
    target = labels.float()
    intersection = (probability * target).sum(dim=(1, 2))
    dice = 1.0 - ((2 * intersection + 1.0) / (probability.sum(dim=(1, 2)) + target.sum(dim=(1, 2)) + 1.0)).mean()
    total = ce_weight * ce + dice_weight * dice
    return total, {"loss": float(total.detach()), "ce": float(ce.detach()), "dice_loss": float(dice.detach())}


@torch.inference_mode()
def _validation_sweep(model: SegformerWaterModel, loader: DataLoader, device: torch.device, cfg: Any) -> dict:
    thresholds = np.linspace(cfg.train.threshold_min, cfg.train.threshold_max, cfg.train.num_thresholds)
    sweep = ThresholdSweep(thresholds)
    model.eval()
    for batch in tqdm(loader, desc="SegFormer validation", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        logits = model(pixel_values)
        probabilities = torch.softmax(F.interpolate(logits, size=batch["labels"].shape[-2:], mode="bilinear", align_corners=False), dim=1)[:, 1]
        for probability, target in zip(probabilities.cpu().numpy(), batch["labels"].numpy()):
            sweep.update(probability, target)
    return sweep.best("iou"), sweep.table()


def train_segformer(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Any, output_dir: str | Path) -> dict:
    seed_everything(cfg.train.seed)
    output_dir = Path(output_dir)
    checkpoint_dir = output_dir / "checkpoints" / "segformer_best"
    history_path = output_dir / "segformer_training_history.csv"
    train_loader = DataLoader(
        SegformerDataset(train_df, cfg.semantic.image_size, True),
        batch_size=cfg.semantic.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
        pin_memory=True, collate_fn=collate_segformer, persistent_workers=cfg.data.num_workers > 0,
    )
    val_loader = DataLoader(
        SegformerDataset(val_df, cfg.semantic.image_size, False),
        batch_size=cfg.semantic.batch_size, shuffle=False, num_workers=cfg.data.num_workers,
        pin_memory=True, collate_fn=collate_segformer, persistent_workers=cfg.data.num_workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SegformerWaterModel(cfg.semantic.model_id, cfg.semantic.image_size).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.semantic.learning_rate, weight_decay=cfg.semantic.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.semantic.epochs, 1))
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.train.amp and device.type == "cuda")
    history, best_iou, stale = [], -1.0, 0

    for epoch in range(1, cfg.semantic.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "ce": 0.0, "dice_loss": 0.0}
        for batch in tqdm(train_loader, desc=f"SegFormer train {epoch}", leave=False):
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=cfg.train.amp and device.type == "cuda"):
                logits = model(pixel_values)
                loss, parts = _loss(logits, labels, cfg.semantic.ce_weight, cfg.semantic.dice_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            for key in totals:
                totals[key] += parts[key]
        scheduler.step()
        best_threshold, threshold_table = _validation_sweep(model, val_loader, device, cfg)
        record = {
            "epoch": epoch,
            **{f"train_{key}": value / max(len(train_loader), 1) for key, value in totals.items()},
            "val_iou": best_threshold["iou"], "val_dice": best_threshold["dice"],
            "val_threshold": best_threshold["threshold"], "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(history_path, index=False)
        threshold_table.to_csv(output_dir / f"segformer_threshold_sweep_epoch_{epoch:02d}.csv", index=False)
        if best_threshold["iou"] > best_iou:
            best_iou, stale = best_threshold["iou"], 0
            metadata = {
                "model_id": cfg.semantic.model_id, "image_size": cfg.semantic.image_size,
                "threshold": best_threshold["threshold"], "epoch": epoch,
                "val_iou": best_threshold["iou"], "config": cfg.to_dict(),
            }
            model.save_checkpoint(checkpoint_dir, metadata)
            write_json(output_dir / "segformer_best.json", metadata)
        else:
            stale += 1
        if stale >= cfg.semantic.patience:
            break
    return json.loads((output_dir / "segformer_best.json").read_text())


@torch.inference_mode()
def evaluate_segformer_all(
    all_manifest: pd.DataFrame,
    cfg: Any,
    output_dir: str | Path,
    save_masks: bool = True,
    save_probabilities: bool = True,
    resume: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    csv_path = output_dir / "experiment_C_segformer_all_2841.csv"
    if csv_path.exists() and not resume:
        csv_path.unlink()
    done = completed_keys(csv_path) if resume else set()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, metadata = SegformerWaterModel.from_checkpoint(output_dir / "checkpoints" / "segformer_best", device)
    threshold = float(metadata["threshold"])
    loader = DataLoader(
        SegformerDataset(all_manifest, model.image_size, False), batch_size=cfg.semantic.batch_size,
        shuffle=False, num_workers=cfg.data.num_workers, pin_memory=True,
        collate_fn=collate_segformer, persistent_workers=cfg.data.num_workers > 0,
    )
    lookup = all_manifest.set_index("image_id")
    model.eval()
    for batch in tqdm(loader, desc="Experiment C: SegFormer all images"):
        if all((str(image_id), "automatic") in done for image_id in batch["image_ids"]):
            continue
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        logits = model(pixel_values)
        logits = F.interpolate(logits, size=(model.image_size, model.image_size), mode="bilinear", align_corners=False)
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        if device.type == "cuda":
            torch.cuda.synchronize()
        latency_ms = (time.perf_counter() - started) * 1000.0 / len(batch["image_ids"])
        rows = []
        for probability_resized, target, image_id in zip(probabilities, batch["targets"], batch["image_ids"]):
            key = (str(image_id), "automatic")
            if key in done:
                continue
            probability = cv2.resize(probability_resized, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
            prediction = (probability >= threshold).astype(np.uint8)
            prediction_path = ""
            probability_path = ""
            if save_masks:
                prediction_path = save_binary_mask(output_dir / "predictions" / "experiment_C_segformer" / f"{image_id}.png", prediction)
            if save_probabilities:
                probability_path = save_probability_u16(output_dir / "probabilities" / "experiment_C_segformer" / f"{image_id}.png", probability)
            info = lookup.loc[str(image_id)]
            rows.append(per_image_result(
                probability, target, threshold, image_id=str(image_id), split=str(info.split),
                experiment="experiment_C_segformer", prompt_mode="automatic",
                image_path=str(info.image_path), mask_path=str(info.mask_path),
                prediction_path=prediction_path, probability_path=probability_path, latency_ms=latency_ms,
            ))
            done.add(key)
        append_rows(csv_path, rows)
    return csv_path
