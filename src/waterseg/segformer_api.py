from __future__ import annotations

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

from waterseg.segformer_inference import SegformerWaterSegmenter

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger(__name__)

SEGMENTER: Optional[SegformerWaterSegmenter] = None
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_MB", "25")) * 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEGMENTER

    threshold_text = os.getenv("MODEL_THRESHOLD", "").strip()
    threshold = None if not threshold_text else float(threshold_text)

    SEGMENTER = SegformerWaterSegmenter(
        checkpoint=os.getenv(
            "SEGFORMER_CHECKPOINT",
            "/models/segformer_best",
        ),
        device=os.getenv("DEVICE", "auto"),
        threshold=threshold,
    )
    LOGGER.info(
        "SegFormer ready: device=%s threshold=%.4f epoch=%s val_iou=%s",
        SEGMENTER.device,
        SEGMENTER.threshold,
        SEGMENTER.metadata.get("epoch"),
        SEGMENTER.metadata.get("val_iou"),
    )

    yield
    SEGMENTER = None


app = FastAPI(
    title="Aereo Water Segmentation API",
    description="Automatic SegFormer-based water-body segmentation for RGB satellite imagery.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    is_ready = SEGMENTER is not None
    return JSONResponse(
        {"ready": is_ready},
        status_code=200 if is_ready else 503,
    )


@app.get("/metadata")
def metadata() -> dict:
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    return {
        "model_id": SEGMENTER.metadata.get("model_id"),
        "task": "water-body segmentation",
        "image_size": SEGMENTER.metadata.get("image_size"),
        "threshold": SEGMENTER.threshold,
        "checkpoint_epoch": SEGMENTER.metadata.get("epoch"),
        "validation_iou": SEGMENTER.metadata.get("val_iou"),
        "device": str(SEGMENTER.device),
    }


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    threshold: Optional[float] = Form(None),
    min_component_area: int = Form(0),
    fill_holes: bool = Form(False),
) -> Response:
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model is not ready")

    content = await image.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image is too large")
    if min_component_area < 0:
        raise HTTPException(status_code=422, detail="min_component_area must be non-negative")

    started = time.perf_counter()
    try:
        rgb = np.asarray(Image.open(BytesIO(content)).convert("RGB"))
    except Exception as error:
        raise HTTPException(status_code=415, detail="Unsupported or corrupt image") from error

    try:
        mask, _ = SEGMENTER.segment(
            rgb,
            threshold=threshold,
            min_component_area=min_component_area,
            fill_holes=fill_holes,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    success, encoded = cv2.imencode(".png", mask * 255)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output mask")

    latency_ms = (time.perf_counter() - started) * 1000.0
    used_threshold = SEGMENTER.threshold if threshold is None else float(threshold)
    LOGGER.info(
        "Segmentation completed: filename=%s size=%dx%d latency_ms=%.2f",
        image.filename,
        rgb.shape[1],
        rgb.shape[0],
        latency_ms,
    )

    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={
            "X-End-To-End-Latency-Ms": f"{latency_ms:.2f}",
            "X-Model-Threshold": f"{used_threshold:.6f}",
        },
    )
