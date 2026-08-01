from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from aereo_water.data.transforms import (
    resize_pair,
    restore_probability_to_original,
)
from aereo_water.models.segformer import load_segformer_checkpoint
from aereo_water.registry import checkpoint_weights_path
from aereo_water.utils import sha256_file


class SegFormerPredictor:
    """Production predictor with integrity checks and structured timing."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        selected_model_path: str | Path | None = None,
        threshold: float | None = None,
        image_size: int = 512,
        resize_policy: str = "letterbox",
        device: str | torch.device | None = None,
        log_path: str | Path | None = None,
        model_version: str = "segformer-v3",
        warmup_runs: int = 1,
    ) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)
        self.selected_model_path = (
            Path(selected_model_path)
            if selected_model_path is not None
            else None
        )
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model, self.processor = load_segformer_checkpoint(
            str(self.checkpoint_dir),
            device=self.device,
        )
        self.model_version = model_version
        self.image_size = int(image_size)
        self.resize_policy = resize_policy
        self.log_path = Path(log_path) if log_path else None

        metadata = self._load_metadata()
        registered_threshold = metadata.get("validation_threshold")
        if threshold is None and registered_threshold is None:
            raise ValueError(
                "No threshold was provided and the selected-model metadata "
                "does not contain validation_threshold."
            )
        self.threshold = float(
            threshold if threshold is not None else registered_threshold
        )

        weights = checkpoint_weights_path(self.checkpoint_dir)
        self.checkpoint_sha256 = sha256_file(weights)
        expected_hash = metadata.get("checkpoint_sha256")
        if expected_hash and expected_hash != self.checkpoint_sha256:
            raise RuntimeError(
                "Checkpoint integrity validation failed. Expected "
                f"{expected_hash}, found {self.checkpoint_sha256}."
            )

        if warmup_runs > 0:
            self._warmup(warmup_runs)

    def _load_metadata(self) -> dict[str, Any]:
        candidates = [
            self.selected_model_path,
            self.checkpoint_dir / "selected_model.json",
            self.checkpoint_dir / "selected_threshold.json",
            self.checkpoint_dir.parent / "selected_model.json",
            self.checkpoint_dir.parent / "selected_threshold.json",
        ]
        metadata: dict[str, Any] = {}
        for path in candidates:
            if path is None or not path.exists():
                continue
            try:
                metadata.update(
                    json.loads(path.read_text(encoding="utf-8"))
                )
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON metadata: {path}") from exc
        return metadata

    def _warmup(self, runs: int) -> None:
        dummy = Image.new(
            "RGB",
            (self.image_size, self.image_size),
            color=(0, 0, 0),
        )
        dummy_mask = Image.new("L", dummy.size, color=0)
        prepared, _, _ = resize_pair(
            dummy,
            dummy_mask,
            size=self.image_size,
            policy=self.resize_policy,
        )
        encoded = self.processor(
            images=prepared,
            return_tensors="pt",
            do_resize=False,
        )
        pixel_values = encoded["pixel_values"].to(self.device)
        with torch.inference_mode():
            for _ in range(runs):
                _ = self.model(pixel_values=pixel_values).logits
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    def _write_log(self, record: dict[str, Any]) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    @torch.inference_mode()
    def predict(
        self,
        image_or_path: str | Path | Image.Image,
        *,
        request_id: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        request_id = request_id or uuid.uuid4().hex
        total_started = time.perf_counter()
        record: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "model_version": self.model_version,
            "checkpoint_sha256": self.checkpoint_sha256,
            "device": str(self.device),
            "threshold": self.threshold,
            "resize_policy": self.resize_policy,
            "status": "error",
            "error_type": None,
            "error_message": None,
        }
        try:
            preprocessing_started = time.perf_counter()
            if isinstance(image_or_path, Image.Image):
                image = image_or_path.convert("RGB")
                input_name = "<PIL.Image>"
            else:
                input_path = Path(image_or_path)
                with Image.open(input_path) as raw:
                    image = raw.convert("RGB")
                input_name = input_path.name

            original_width, original_height = image.size
            dummy_mask = Image.new("L", image.size, color=0)
            prepared, _, transform = resize_pair(
                image,
                dummy_mask,
                size=self.image_size,
                policy=self.resize_policy,
            )
            encoded = self.processor(
                images=prepared,
                return_tensors="pt",
                do_resize=False,
            )
            pixel_values = encoded["pixel_values"].to(self.device)
            preprocessing_ms = (
                time.perf_counter() - preprocessing_started
            ) * 1000.0

            if self.device.type == "cuda":
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                logits = self.model(pixel_values=pixel_values).logits
                end.record()
                torch.cuda.synchronize()
                model_ms = float(start.elapsed_time(end))
            else:
                model_started = time.perf_counter()
                logits = self.model(pixel_values=pixel_values).logits
                model_ms = (
                    time.perf_counter() - model_started
                ) * 1000.0

            post_started = time.perf_counter()
            logits_canvas = F.interpolate(
                logits,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            probability_canvas = (
                torch.softmax(logits_canvas, dim=1)[0, 1]
                .detach()
                .cpu()
                .numpy()
            )
            probability = restore_probability_to_original(
                probability_canvas,
                transform,
            )
            mask = (probability >= self.threshold).astype(np.uint8)
            postprocessing_ms = (
                time.perf_counter() - post_started
            ) * 1000.0
            total_ms = (
                time.perf_counter() - total_started
            ) * 1000.0

            record.update(
                {
                    "input_filename": input_name,
                    "input_width": int(original_width),
                    "input_height": int(original_height),
                    "input_pixels": int(
                        original_width * original_height
                    ),
                    "preprocessing_ms": float(preprocessing_ms),
                    "model_ms": float(model_ms),
                    "postprocessing_ms": float(postprocessing_ms),
                    "total_ms": float(total_ms),
                    "predicted_water_fraction": float(mask.mean()),
                    "output_width": int(mask.shape[1]),
                    "output_height": int(mask.shape[0]),
                    "output_values": sorted(
                        np.unique(mask).astype(int).tolist()
                    ),
                    "status": "ok",
                }
            )
            self._write_log(record)
            return mask, probability, record
        except Exception as exc:
            record.update(
                {
                    "total_ms": float(
                        (time.perf_counter() - total_started) * 1000.0
                    ),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            self._write_log(record)
            raise

    @staticmethod
    def save_mask(mask: np.ndarray, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(output)
        return output

    @staticmethod
    def save_overlay(
        image_or_path: str | Path | Image.Image,
        mask: np.ndarray,
        path: str | Path,
        *,
        alpha: int = 110,
    ) -> Path:
        if isinstance(image_or_path, Image.Image):
            image = image_or_path.convert("RGBA")
        else:
            with Image.open(image_or_path) as raw:
                image = raw.convert("RGBA")
        mask_image = Image.fromarray(
            mask.astype(np.uint8) * 255,
            mode="L",
        )
        if mask_image.size != image.size:
            mask_image = mask_image.resize(
                image.size,
                Image.Resampling.NEAREST,
            )
        layer = Image.new("RGBA", image.size, (255, 0, 0, 0))
        layer.putalpha(
            mask_image.point(lambda value: alpha if value else 0)
        )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        Image.alpha_composite(image, layer).convert("RGB").save(output)
        return output
