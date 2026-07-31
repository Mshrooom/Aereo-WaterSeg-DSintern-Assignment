from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np
import torch

from waterseg.data.manifest import read_rgb
from waterseg.data.tiling import extract_tile, generate_windows, stitch_probability_tiles
from waterseg.models.sam_water import SamWaterModel
from waterseg.prompting import PromptBatch
from waterseg.utils import sha256_file

LOGGER = logging.getLogger(__name__)


class EmbeddingCache:
    def __init__(self, max_items: int = 8):
        self.max_items = max_items
        self.cache: OrderedDict[str, torch.Tensor] = OrderedDict()

    def get(self, key: str):
        if key not in self.cache:
            return None
        value = self.cache.pop(key)
        self.cache[key] = value
        return value

    def put(self, key: str, value) -> None:
        if key in self.cache:
            self.cache.pop(key)
        self.cache[key] = value
        while len(self.cache) > self.max_items:
            self.cache.popitem(last=False)


class WaterSegmenter:
    def __init__(self, checkpoint: str | Path, device: str = "auto", cache_size: int = 8):
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        metadata = payload.get("metadata", {})
        config = metadata.get("config", {})
        model_cfg = config.get("model", {})
        self.threshold = float(metadata.get("threshold", config.get("inference", {}).get("threshold", 0.5)))
        self.device = torch.device("cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device))
        self.model = SamWaterModel(
            payload.get("model_id", model_cfg.get("model_id", "facebook/sam-vit-base")),
            trainable_parts=model_cfg.get("trainable_parts", "mask_decoder"),
            unfreeze_last_vision_blocks=model_cfg.get("unfreeze_last_vision_blocks", 0),
        )
        self.model.sam.load_state_dict(payload["trainable_state_dict"], strict=False)
        self.model.to(self.device).eval()
        self.cache = EmbeddingCache(cache_size)

    @torch.inference_mode()
    def segment(
        self,
        image: np.ndarray,
        points: Optional[Sequence[Sequence[float]]] = None,
        labels: Optional[Sequence[int]] = None,
        box: Optional[Sequence[float]] = None,
        threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        prompt = PromptBatch(mode="none")
        if points is not None and (labels is None or len(points) != len(labels)):
            raise ValueError("points and labels must have equal lengths")
        if points is not None and box is not None:
            prompt = PromptBatch(
                mode="box_points",
                points=[np.asarray(points, dtype=np.float32)],
                labels=[np.asarray(labels, dtype=np.int64)],
                boxes=[np.asarray(box, dtype=np.float32)],
            )
        elif points is not None:
            prompt = PromptBatch(mode="points", points=[np.asarray(points, dtype=np.float32)], labels=[np.asarray(labels, dtype=np.int64)])
        elif box is not None:
            prompt = PromptBatch(mode="box", boxes=[np.asarray(box, dtype=np.float32)])

        inputs = self.model.prepare_inputs([image], prompt, self.device)
        image_key = hashlib.sha256(
            str(image.shape).encode("utf-8") + str(image.dtype).encode("utf-8") + image.tobytes()
        ).hexdigest()
        image_embeddings = self.cache.get(image_key)
        pixel_values = inputs.pop("pixel_values")
        if image_embeddings is None:
            image_embeddings = self.model.sam.get_image_embeddings(pixel_values)
            self.cache.put(image_key, image_embeddings)
        inputs["image_embeddings"] = image_embeddings
        logits, _ = self.model.forward_prepared(inputs, multimask_output=False)
        probabilities = self.model.processor.image_processor.post_process_masks(
            torch.sigmoid(logits), inputs["original_sizes"], inputs["reshaped_input_sizes"], binarize=False
        )[0].squeeze().cpu().numpy().astype(np.float32)
        used_threshold = self.threshold if threshold is None else threshold
        mask = (probabilities >= used_threshold).astype(np.uint8)
        return mask, probabilities

    def segment_large_image(
        self,
        image: np.ndarray,
        tile_size: int = 1024,
        overlap: int = 128,
        threshold: Optional[float] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if image.shape[0] <= tile_size and image.shape[1] <= tile_size:
            return self.segment(image, threshold=threshold)
        tiles = []
        for window in generate_windows(image.shape[0], image.shape[1], tile_size, overlap):
            tile = extract_tile(image, window, tile_size)
            _, probability = self.segment(tile, threshold=threshold)
            tiles.append((window, probability))
        probability = stitch_probability_tiles(tiles, image.shape[:2], tile_size)
        used_threshold = self.threshold if threshold is None else threshold
        return (probability >= used_threshold).astype(np.uint8), probability


def postprocess_mask(mask: np.ndarray, min_component_area: int = 0, fill_holes: bool = False) -> np.ndarray:
    result = mask.astype(np.uint8)
    if min_component_area > 0:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(result, connectivity=8)
        filtered = np.zeros_like(result)
        for index in range(1, count):
            if stats[index, cv2.CC_STAT_AREA] >= min_component_area:
                filtered[labels == index] = 1
        result = filtered
    if fill_holes:
        flood = result.copy()
        padded = np.pad(flood, 1)
        mask_buffer = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(padded, mask_buffer, (0, 0), 1)
        holes = 1 - padded[1:-1, 1:-1]
        result = np.maximum(result, holes).astype(np.uint8)
    return result


def save_mask_png(path: str | Path, mask: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask.astype(np.uint8) * 255)


def read_image_with_profile(path: str | Path) -> tuple[np.ndarray, dict | None]:
    """Read common RGB files or GeoTIFF while preserving geospatial metadata.

    For multispectral GeoTIFFs, the first three bands are treated as display RGB and
    independently percentile-stretched to uint8. Band selection should be configured
    explicitly for a true multispectral production model.
    """
    path = Path(path)
    if path.suffix.lower() not in {".tif", ".tiff"}:
        return read_rgb(path), None
    try:
        import rasterio
    except ImportError as error:
        raise ImportError("GeoTIFF support requires the optional 'rasterio' dependency") from error
    with rasterio.open(path) as source:
        data = source.read()
        profile = source.profile.copy()
    if data.shape[0] == 1:
        data = np.repeat(data, 3, axis=0)
    elif data.shape[0] >= 3:
        data = data[:3]
    else:
        raise ValueError(f"Unsupported GeoTIFF band count: {data.shape[0]}")
    channels = []
    for band in data:
        band = band.astype(np.float32)
        finite = np.isfinite(band)
        if not finite.any():
            channels.append(np.zeros_like(band, dtype=np.uint8))
            continue
        low, high = np.percentile(band[finite], [2, 98])
        scaled = np.clip((band - low) / max(high - low, 1e-6), 0, 1)
        channels.append((scaled * 255).astype(np.uint8))
    return np.stack(channels, axis=-1), profile


def save_mask(path: str | Path, mask: np.ndarray, profile: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in {".tif", ".tiff"} and profile is not None:
        try:
            import rasterio
        except ImportError as error:
            raise ImportError("GeoTIFF support requires the optional 'rasterio' dependency") from error
        output_profile = profile.copy()
        output_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            height=mask.shape[0],
            width=mask.shape[1],
            nodata=0,
            compress="lzw",
        )
        with rasterio.open(path, "w", **output_profile) as destination:
            destination.write(mask.astype(np.uint8), 1)
    else:
        save_mask_png(path, mask)
