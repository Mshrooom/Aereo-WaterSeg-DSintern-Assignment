from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from aereo_water.data.dataset import (
    SegmentationInferenceDataset,
    inference_collate,
)
from aereo_water.data.transforms import restore_probability_to_original
from aereo_water.evaluation.metrics import (
    probability_calibration_metrics,
    reliability_table,
    segmentation_metrics,
)


def _measure_forward_ms(model, pixel_values, device: torch.device):
    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(pixel_values=pixel_values).logits
        end.record()
        torch.cuda.synchronize()
        return logits, float(start.elapsed_time(end))
    started = time.perf_counter()
    logits = model(pixel_values=pixel_values).logits
    return logits, float((time.perf_counter() - started) * 1000.0)


@torch.inference_mode()
def evaluate_manifest(
    model: torch.nn.Module,
    processor,
    dataframe: pd.DataFrame,
    *,
    image_size: int,
    resize_policy: str,
    threshold: float,
    device: torch.device | str,
    batch_size: int,
    num_workers: int,
    output_csv: str | Path,
    prediction_dir: str | Path | None = None,
    include_boundary_metrics: bool = True,
    boundary_tolerance: int = 2,
    empty_policy: str = "perfect_if_both_empty",
    probability_sample_pixels: int = 2_000,
    calibration_bins: int = 15,
    calibration_output_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """Run frozen original-resolution inference and export per-image evidence."""
    device = torch.device(device)
    dataset = SegmentationInferenceDataset(
        dataframe,
        processor,
        image_size=image_size,
        resize_policy=resize_policy,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=inference_collate,
    )
    predictions_root = Path(prediction_dir) if prediction_dir else None
    if predictions_root is not None:
        predictions_root.mkdir(parents=True, exist_ok=True)

    model.eval()
    rows: list[dict[str, Any]] = []
    calibration_probabilities: dict[str, list[np.ndarray]] = {}
    calibration_targets: dict[str, list[np.ndarray]] = {}
    rng = np.random.default_rng(42)

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        logits, forward_ms = _measure_forward_ms(model, pixel_values, device)
        logits_canvas = F.interpolate(
            logits,
            size=(image_size, image_size),
            mode="bilinear",
            align_corners=False,
        )
        probabilities = torch.softmax(logits_canvas, dim=1)[:, 1]
        latency_per_image = forward_ms / len(batch["image_ids"])

        for index, target in enumerate(batch["targets"]):
            probability = restore_probability_to_original(
                probabilities[index].detach().cpu().numpy(),
                batch["transforms"][index],
            )
            prediction = (probability >= threshold).astype(np.uint8)
            metrics = segmentation_metrics(
                prediction,
                target,
                include_boundary_metrics=include_boundary_metrics,
                boundary_tolerance=boundary_tolerance,
                empty_policy=empty_policy,
            )
            image_id = batch["image_ids"][index]
            prediction_path = ""
            if predictions_root is not None:
                output_path = predictions_root / f"{image_id}.png"
                Image.fromarray(prediction * 255, mode="L").save(output_path)
                prediction_path = str(output_path)

            flattened_probability = probability.reshape(-1)
            flattened_target = target.reshape(-1)
            take = min(probability_sample_pixels, len(flattened_target))
            sample_indices = rng.choice(
                len(flattened_target),
                size=take,
                replace=False,
            )
            split_name = str(batch["splits"][index])
            calibration_probabilities.setdefault(split_name, []).append(
                flattened_probability[sample_indices]
            )
            calibration_targets.setdefault(split_name, []).append(
                flattened_target[sample_indices]
            )

            rows.append(
                {
                    "image_id": image_id,
                    "split": batch["splits"][index],
                    "image_path": batch["image_paths"][index],
                    "mask_path": batch["mask_paths"][index],
                    "prediction_path": prediction_path,
                    "width": int(target.shape[1]),
                    "height": int(target.shape[0]),
                    "threshold": float(threshold),
                    "model_forward_latency_ms": float(latency_per_image),
                    **metrics,
                }
            )

    frame = pd.DataFrame(rows)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    calibration: dict[str, dict[str, float]] = {}
    calibration_root = (
        Path(calibration_output_dir)
        if calibration_output_dir is not None
        else None
    )
    if calibration_root is not None:
        calibration_root.mkdir(parents=True, exist_ok=True)
    all_probabilities: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    for split_name in sorted(calibration_probabilities):
        split_probabilities = np.concatenate(
            calibration_probabilities[split_name]
        )
        split_targets = np.concatenate(
            calibration_targets[split_name]
        )
        calibration[split_name] = probability_calibration_metrics(
            split_probabilities,
            split_targets,
            bins=calibration_bins,
        )
        if calibration_root is not None:
            reliability_table(
                split_probabilities,
                split_targets,
                bins=calibration_bins,
            ).to_csv(
                calibration_root / f"{split_name}_reliability.csv",
                index=False,
            )
        all_probabilities.append(split_probabilities)
        all_targets.append(split_targets)
    overall_probabilities = np.concatenate(all_probabilities)
    overall_targets = np.concatenate(all_targets)
    calibration["overall"] = probability_calibration_metrics(
        overall_probabilities,
        overall_targets,
        bins=calibration_bins,
    )
    if calibration_root is not None:
        reliability_table(
            overall_probabilities,
            overall_targets,
            bins=calibration_bins,
        ).to_csv(
            calibration_root / "overall_reliability.csv",
            index=False,
        )
    return frame, calibration


def summarize_results(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "iou",
        "dice",
        "precision",
        "recall",
        "specificity",
        "pixel_accuracy",
        "balanced_accuracy",
        "mcc",
        "cohen_kappa",
        "boundary_f1",
        "boundary_iou",
        "hd95",
        "assd",
        "water_fraction",
        "predicted_water_fraction",
        "water_fraction_error",
        "model_forward_latency_ms",
    ]
    available = [column for column in metrics if column in frame]
    summary = (
        frame.groupby("split", dropna=False)[available]
        .mean(numeric_only=True)
        .reset_index()
    )
    counts = frame.groupby("split").size()
    summary.insert(
        1,
        "images",
        summary["split"].map(counts).astype(int),
    )
    return summary


@torch.inference_mode()
def benchmark_model_forward(
    model,
    processor,
    image,
    *,
    image_size: int,
    resize_policy: str,
    device: torch.device | str,
    warmup_runs: int = 5,
    timed_runs: int = 50,
) -> dict[str, float]:
    from aereo_water.data.transforms import resize_pair
    from PIL import Image as PILImage

    device = torch.device(device)
    if not isinstance(image, PILImage.Image):
        with PILImage.open(image) as raw:
            image = raw.convert("RGB")
    dummy_mask = PILImage.new("L", image.size, color=0)
    prepared, _, _ = resize_pair(
        image,
        dummy_mask,
        size=image_size,
        policy=resize_policy,
    )
    encoded = processor(
        images=prepared,
        return_tensors="pt",
        do_resize=False,
    )
    pixel_values = encoded["pixel_values"].to(device)
    model.eval()

    for _ in range(warmup_runs):
        _ = model(pixel_values=pixel_values).logits
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    latencies = []
    for _ in range(timed_runs):
        _, latency = _measure_forward_ms(model, pixel_values, device)
        latencies.append(latency)

    return {
        "warmup_runs": int(warmup_runs),
        "timed_runs": int(timed_runs),
        "p50_model_forward_ms": float(np.percentile(latencies, 50)),
        "p95_model_forward_ms": float(np.percentile(latencies, 95)),
        "mean_model_forward_ms": float(np.mean(latencies)),
        "throughput_images_per_second": float(
            1000.0 / np.mean(latencies)
        ),
        "peak_inference_memory_mb": float(
            torch.cuda.max_memory_allocated() / (1024**2)
            if device.type == "cuda"
            else 0.0
        ),
    }
