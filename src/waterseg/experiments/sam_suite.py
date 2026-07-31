from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from waterseg.data.dataset import WaterDataset, list_collate
from waterseg.data.transforms import JointTransform
from waterseg.experiments.common import append_rows, completed_keys, per_image_result, save_binary_mask
from waterseg.metrics import ThresholdSweep
from waterseg.models.sam_water import SamWaterModel, normalize_pred_masks
from waterseg.prompting import build_prompt_batch, processor_prompt_kwargs
from waterseg.utils import stable_int_hash, write_json


@torch.inference_mode()
def _batch_prompt_probabilities(
    model: SamWaterModel,
    batch: dict,
    prompt_modes: Sequence[str],
    cfg: Any,
    device: torch.device,
) -> tuple[Dict[str, list[np.ndarray]], Dict[str, float]]:
    images = batch["images"]
    base = model.processor(images=images, return_tensors="pt")
    base = {key: value.to(device) if torch.is_tensor(value) else value for key, value in base.items()}
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_encoder = time.perf_counter()
    embeddings = model.sam.get_image_embeddings(base["pixel_values"])
    if device.type == "cuda":
        torch.cuda.synchronize()
    encoder_ms = (time.perf_counter() - start_encoder) * 1000.0 / len(images)

    probabilities: Dict[str, list[np.ndarray]] = {}
    latencies: Dict[str, float] = {}
    for mode in prompt_modes:
        rngs = [
            np.random.default_rng(stable_int_hash(f"{image_id}:{mode}", cfg.train.seed))
            for image_id in batch["image_ids"]
        ]
        prompt = build_prompt_batch(
            batch["masks"], mode, cfg.prompts.positive_points, cfg.prompts.negative_points,
            cfg.prompts.box_jitter_fraction, rngs,
        )
        prepared = model.processor(images=images, return_tensors="pt", **processor_prompt_kwargs(prompt))
        prepared = {key: value.to(device) if torch.is_tensor(value) else value for key, value in prepared.items()}
        prepared.pop("pixel_values", None)
        prepared["image_embeddings"] = embeddings
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        outputs = model.sam(**prepared, multimask_output=False)
        logits = normalize_pred_masks(outputs.pred_masks)
        low_probability = torch.sigmoid(logits)
        processed = model.processor.image_processor.post_process_masks(
            low_probability, prepared["original_sizes"], prepared["reshaped_input_sizes"], binarize=False
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        decoder_ms = (time.perf_counter() - started) * 1000.0 / len(images)
        probabilities[mode] = [
            tensor.squeeze().detach().cpu().numpy().astype(np.float32) for tensor in processed
        ]
        latencies[mode] = encoder_ms + decoder_ms
    return probabilities, latencies


def calibrate_sam_thresholds(
    model: SamWaterModel,
    validation_manifest: pd.DataFrame,
    prompt_modes: Sequence[str],
    cfg: Any,
    device: torch.device,
) -> Dict[str, float]:
    loader = DataLoader(
        WaterDataset(validation_manifest, JointTransform(training=False)),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=list_collate,
        persistent_workers=cfg.data.num_workers > 0,
    )
    thresholds = np.linspace(cfg.train.threshold_min, cfg.train.threshold_max, cfg.train.num_thresholds)
    sweeps = {mode: ThresholdSweep(thresholds) for mode in prompt_modes}
    model.eval()
    for batch in tqdm(loader, desc="SAM threshold calibration", leave=False):
        predictions, _ = _batch_prompt_probabilities(model, batch, prompt_modes, cfg, device)
        for mode in prompt_modes:
            for probability, target in zip(predictions[mode], batch["masks"]):
                sweeps[mode].update(probability, target)
    return {mode: sweeps[mode].best("iou")["threshold"] for mode in prompt_modes}


def evaluate_sam_all_images(
    model: SamWaterModel,
    all_manifest: pd.DataFrame,
    prompt_modes: Sequence[str],
    thresholds: Dict[str, float],
    cfg: Any,
    device: torch.device,
    output_dir: str | Path,
    experiment_name: str,
    save_masks: bool = True,
    resume: bool = True,
) -> Path:
    output_dir = Path(output_dir)
    csv_path = output_dir / f"{experiment_name}_all_2841.csv"
    done = completed_keys(csv_path) if resume else set()
    if csv_path.exists() and not resume:
        csv_path.unlink()
    manifest_lookup = all_manifest.set_index("image_id")
    loader = DataLoader(
        WaterDataset(all_manifest, JointTransform(training=False)),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=list_collate,
        persistent_workers=cfg.data.num_workers > 0,
    )
    model.eval()
    for batch in tqdm(loader, desc=experiment_name):
        needed_modes = [
            mode for mode in prompt_modes
            if any((str(image_id), mode) not in done for image_id in batch["image_ids"])
        ]
        if not needed_modes:
            continue
        probabilities, latency = _batch_prompt_probabilities(model, batch, needed_modes, cfg, device)
        rows = []
        for mode in needed_modes:
            threshold = float(thresholds[mode])
            for probability, target, image_id in zip(probabilities[mode], batch["masks"], batch["image_ids"]):
                key = (str(image_id), mode)
                if key in done:
                    continue
                info = manifest_lookup.loc[str(image_id)]
                prediction = (probability >= threshold).astype(np.uint8)
                prediction_path = ""
                if save_masks:
                    prediction_path = save_binary_mask(
                        output_dir / "predictions" / experiment_name / mode / f"{image_id}.png", prediction
                    )
                rows.append(per_image_result(
                    probability, target, threshold,
                    image_id=str(image_id), split=str(info.split), experiment=experiment_name,
                    prompt_mode=mode, image_path=str(info.image_path), mask_path=str(info.mask_path),
                    prediction_path=prediction_path, latency_ms=latency[mode],
                ))
                done.add(key)
        append_rows(csv_path, rows)
    write_json(output_dir / f"{experiment_name}_thresholds.json", thresholds)
    return csv_path
