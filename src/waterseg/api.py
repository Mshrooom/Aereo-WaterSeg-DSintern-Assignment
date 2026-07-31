from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from waterseg.inference import WaterSegmenter, postprocess_mask
from waterseg.logging_utils import configure_logging

configure_logging(os.getenv("LOG_LEVEL", "INFO"), json_logs=True)
LOGGER = logging.getLogger(__name__)
SEGMENTER: Optional[WaterSegmenter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEGMENTER
    checkpoint = os.getenv("MODEL_CHECKPOINT", "/models/best.pt")
    SEGMENTER = WaterSegmenter(checkpoint, device=os.getenv("DEVICE", "auto"), cache_size=int(os.getenv("CACHE_SIZE", "8")))
    LOGGER.info("Model loaded", extra={"status": "ready"})
    yield
    SEGMENTER = None


app = FastAPI(title="Aereo Water Segmentation API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    status = 200 if SEGMENTER is not None else 503
    return JSONResponse({"ready": SEGMENTER is not None}, status_code=status)


def _parse_json_field(value: Optional[str], name: str):
    if value is None or value == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as error:
        raise HTTPException(status_code=422, detail=f"Invalid JSON in '{name}': {error}") from error


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    points: Optional[str] = Form(None),
    labels: Optional[str] = Form(None),
    box: Optional[str] = Form(None),
    threshold: Optional[float] = Form(None),
    min_component_area: int = Form(0),
    fill_holes: bool = Form(False),
) -> Response:
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    content = await image.read()
    try:
        rgb = np.asarray(Image.open(BytesIO(content)).convert("RGB"))
    except Exception as error:
        raise HTTPException(status_code=415, detail="Unsupported image") from error
    parsed_points = _parse_json_field(points, "points")
    parsed_labels = _parse_json_field(labels, "labels")
    parsed_box = _parse_json_field(box, "box")
    started = time.perf_counter()
    try:
        mask, _ = SEGMENTER.segment(rgb, parsed_points, parsed_labels, parsed_box, threshold)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    mask = postprocess_mask(mask, min_component_area=min_component_area, fill_holes=fill_holes)
    success, encoded = cv2.imencode(".png", mask * 255)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output mask")
    latency_ms = (time.perf_counter() - started) * 1000.0
    LOGGER.info("Segmentation completed", extra={"latency_ms": latency_ms, "status": "success"})
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={"X-Inference-Latency-Ms": f"{latency_ms:.2f}", "X-Model-Threshold": str(SEGMENTER.threshold if threshold is None else threshold)},
    )
