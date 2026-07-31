from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from waterseg.auto_prompt import automatic_prompt_from_probability
from waterseg.data.manifest import read_mask, read_rgb
from waterseg.experiments.common import (
    append_rows, completed_keys, per_image_result, read_probability_u16, save_binary_mask,
)
from waterseg.metrics import ThresholdSweep
from waterseg.models.sam_water import SamWaterModel


@torch.inference_mode()
def _auto_sam_probability(
    model: SamWaterModel,
    image: np.ndarray,
    coarse_probability: np.ndarray,
    cfg: Any,
    device: torch.device,
) -> tuple[np.ndarray, float, str, int]:
    auto = automatic_prompt_from_probability(
        coarse_probability,
        coarse_threshold=cfg.auto_sam.coarse_threshold,
        max_positive_points=cfg.auto_sam.max_positive_points,
        negative_points=cfg.auto_sam.negative_points,
        min_component_area=cfg.auto_sam.min_component_area,
        box_padding_fraction=cfg.auto_sam.box_padding_fraction,
        morphology_kernel=cfg.auto_sam.morphology_kernel,
    )
    if auto.status != "ok":
        return coarse_probability, 0.0, auto.status, auto.components
    inputs = model.prepare_inputs([image], auto.prompt, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()
    logits, _ = model.forward_prepared(inputs, multimask_output=False)
    processed = model.processor.image_processor.post_process_masks(
        torch.sigmoid(logits), inputs["original_sizes"], inputs["reshaped_input_sizes"], binarize=False
    )[0].squeeze().cpu().numpy().astype(np.float32)
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (time.perf_counter() - started) * 1000.0
    fused = cfg.auto_sam.sam_weight * processed + (1.0 - cfg.auto_sam.sam_weight) * coarse_probability
    return np.clip(fused, 0, 1), latency_ms, auto.status, auto.components


def calibrate_auto_sam_threshold(
    model: SamWaterModel,
    validation_manifest: pd.DataFrame,
    cfg: Any,
    device: torch.device,
    output_dir: str | Path,
) -> float:
    thresholds = np.linspace(cfg.train.threshold_min, cfg.train.threshold_max, cfg.train.num_thresholds)
    sweep = ThresholdSweep(thresholds)
    probability_root = Path(output_dir) / "probabilities" / "experiment_C_segformer"
    model.eval()
    for _, row in tqdm(validation_manifest.iterrows(), total=len(validation_manifest), desc="Experiment D calibration"):
        image = read_rgb(row.image_path)
        target = read_mask(row.mask_path)
        coarse = read_probability_u16(probability_root / f"{row.image_id}.png")
        probability, _, _, _ = _auto_sam_probability(model, image, coarse, cfg, device)
        sweep.update(probability, target)
    table = sweep.table()
    table.to_csv(Path(output_dir) / "experiment_D_threshold_sweep.csv", index=False)
    return float(sweep.best("iou")["threshold"])


def evaluate_auto_sam_all(
    model: SamWaterModel,
    all_manifest: pd.DataFrame,
    threshold: float,
    cfg: Any,
    device: torch.device,
    output_dir: str | Path,
    save_masks: bool = True,
    resume: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    csv_path = output_dir / "experiment_D_auto_sam_all_2841.csv"
    if csv_path.exists() and not resume:
        csv_path.unlink()
    done = completed_keys(csv_path) if resume else set()
    probability_root = output_dir / "probabilities" / "experiment_C_segformer"
    model.eval()
    for _, row in tqdm(all_manifest.iterrows(), total=len(all_manifest), desc="Experiment D: automatic SAM all images"):
        key = (str(row.image_id), "segformer_auto_prompt")
        if key in done:
            continue
        image = read_rgb(row.image_path)
        target = read_mask(row.mask_path)
        coarse = read_probability_u16(probability_root / f"{row.image_id}.png")
        probability, latency_ms, status, components = _auto_sam_probability(model, image, coarse, cfg, device)
        prediction = (probability >= threshold).astype(np.uint8)
        prediction_path = ""
        if save_masks:
            prediction_path = save_binary_mask(
                output_dir / "predictions" / "experiment_D_auto_sam" / f"{row.image_id}.png", prediction
            )
        result = per_image_result(
            probability, target, threshold, image_id=str(row.image_id), split=str(row.split),
            experiment="experiment_D_auto_sam", prompt_mode="segformer_auto_prompt",
            image_path=str(row.image_path), mask_path=str(row.mask_path), prediction_path=prediction_path,
            latency_ms=latency_ms, status=status, extra={"coarse_components": components},
        )
        append_rows(csv_path, [result])
        done.add(key)
    return csv_path
