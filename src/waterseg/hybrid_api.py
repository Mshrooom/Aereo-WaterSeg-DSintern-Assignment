from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response
from PIL import Image

from waterseg.hybrid_inference import HybridWaterSegmenter
from waterseg.logging_utils import configure_logging

configure_logging(os.getenv("LOG_LEVEL", "INFO"), json_logs=True)
LOGGER = logging.getLogger(__name__)
SEGMENTER: Optional[HybridWaterSegmenter] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global SEGMENTER
    SEGMENTER = HybridWaterSegmenter(
        sam_checkpoint=os.getenv("SAM_CHECKPOINT", "/models/sam/best.pt"),
        segformer_checkpoint=os.getenv("SEGFORMER_CHECKPOINT", "/models/segformer"),
        device=os.getenv("DEVICE", "auto"),
        threshold=float(os.getenv("HYBRID_THRESHOLD", "0.5")),
        coarse_threshold=float(os.getenv("COARSE_THRESHOLD", "0.5")),
        sam_weight=float(os.getenv("SAM_WEIGHT", "0.75")),
    )
    LOGGER.info("Hybrid model loaded", extra={"status": "ready"})
    yield
    SEGMENTER = None


app = FastAPI(title="Aereo Automatic Water Segmentation API", version="0.2.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> JSONResponse:
    status = 200 if SEGMENTER is not None else 503
    return JSONResponse({"ready": SEGMENTER is not None}, status_code=status)


@app.post("/segment")
async def segment(image: UploadFile = File(...)) -> Response:
    if SEGMENTER is None:
        raise HTTPException(status_code=503, detail="Model is not ready")
    content = await image.read()
    try:
        rgb = np.asarray(Image.open(BytesIO(content)).convert("RGB"))
    except Exception as error:
        raise HTTPException(status_code=415, detail="Unsupported image") from error
    started = time.perf_counter()
    mask, _, metadata = SEGMENTER.segment(rgb)
    success, encoded = cv2.imencode(".png", mask * 255)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to encode output mask")
    latency_ms = (time.perf_counter() - started) * 1000.0
    LOGGER.info(
        "Automatic segmentation completed",
        extra={"latency_ms": latency_ms, "status": metadata.get("status", "ok")},
    )
    return Response(
        content=encoded.tobytes(),
        media_type="image/png",
        headers={
            "X-Inference-Latency-Ms": f"{latency_ms:.2f}",
            "X-Auto-Prompt-Status": str(metadata.get("status", "ok")),
        },
    )
