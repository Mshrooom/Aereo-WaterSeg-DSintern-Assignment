from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from waterseg.auto_prompt import automatic_prompt_from_probability
from waterseg.data.tiling import extract_tile, generate_windows, stitch_probability_tiles
from waterseg.models.sam_water import SamWaterModel
from waterseg.models.segformer_water import SegformerWaterModel


class HybridWaterSegmenter:
    """Automatic SegFormer -> prompt generator -> fine-tuned SAM inference pipeline."""

    def __init__(
        self,
        sam_checkpoint: str | Path,
        segformer_checkpoint: str | Path,
        device: str = "auto",
        threshold: float | None = None,
        coarse_threshold: float = 0.5,
        max_positive_points: int = 3,
        negative_points: int = 1,
        min_component_area: int = 16,
        box_padding_fraction: float = 0.04,
        morphology_kernel: int = 3,
        sam_weight: float = 0.75,
    ):
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        )
        payload = torch.load(sam_checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        model_cfg = metadata.get("config", {}).get("model", {})
        self.sam = SamWaterModel(
            payload.get("model_id", model_cfg.get("model_id", "facebook/sam-vit-base")),
            trainable_parts=model_cfg.get("trainable_parts", "mask_decoder"),
            unfreeze_last_vision_blocks=model_cfg.get("unfreeze_last_vision_blocks", 0),
        )
        self.sam.sam.load_state_dict(payload["trainable_state_dict"], strict=False)
        self.sam.to(self.device).eval()
        self.segformer, seg_metadata = SegformerWaterModel.from_checkpoint(segformer_checkpoint, self.device)
        self.segformer.eval()
        self.threshold = float(threshold if threshold is not None else metadata.get("hybrid_threshold", 0.5))
        self.coarse_threshold = float(coarse_threshold)
        self.max_positive_points = int(max_positive_points)
        self.negative_points = int(negative_points)
        self.min_component_area = int(min_component_area)
        self.box_padding_fraction = float(box_padding_fraction)
        self.morphology_kernel = int(morphology_kernel)
        self.sam_weight = float(sam_weight)

    @torch.inference_mode()
    def coarse_probability(self, image: np.ndarray) -> np.ndarray:
        pixel_values = self.segformer.prepare_images([image], self.device)
        logits = self.segformer(pixel_values)
        probability = torch.softmax(
            F.interpolate(logits, size=image.shape[:2], mode="bilinear", align_corners=False), dim=1
        )[0, 1]
        return probability.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def segment(self, image: np.ndarray, threshold: float | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
        coarse = self.coarse_probability(image)
        auto = automatic_prompt_from_probability(
            coarse,
            coarse_threshold=self.coarse_threshold,
            max_positive_points=self.max_positive_points,
            negative_points=self.negative_points,
            min_component_area=self.min_component_area,
            box_padding_fraction=self.box_padding_fraction,
            morphology_kernel=self.morphology_kernel,
        )
        if auto.status == "ok":
            inputs = self.sam.prepare_inputs([image], auto.prompt, self.device)
            logits, _ = self.sam.forward_prepared(inputs, multimask_output=False)
            sam_probability = self.sam.processor.image_processor.post_process_masks(
                torch.sigmoid(logits), inputs["original_sizes"], inputs["reshaped_input_sizes"], binarize=False
            )[0].squeeze().cpu().numpy().astype(np.float32)
            probability = self.sam_weight * sam_probability + (1.0 - self.sam_weight) * coarse
        else:
            probability = coarse
        used_threshold = self.threshold if threshold is None else float(threshold)
        mask = (probability >= used_threshold).astype(np.uint8)
        return mask, np.clip(probability, 0, 1), {"status": auto.status, "components": auto.components}

    def segment_large_image(
        self, image: np.ndarray, tile_size: int = 1024, overlap: int = 128, threshold: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        if image.shape[0] <= tile_size and image.shape[1] <= tile_size:
            mask, probability, _ = self.segment(image, threshold)
            return mask, probability
        tiles = []
        for window in generate_windows(image.shape[0], image.shape[1], tile_size, overlap):
            tile = extract_tile(image, window, tile_size)
            _, probability, _ = self.segment(tile, threshold)
            tiles.append((window, probability))
        probability = stitch_probability_tiles(tiles, image.shape[:2], tile_size)
        used_threshold = self.threshold if threshold is None else float(threshold)
        return (probability >= used_threshold).astype(np.uint8), probability
