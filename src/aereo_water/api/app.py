from __future__ import annotations

import asyncio
import io
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError

from aereo_water.inference.predictor import SegFormerPredictor


ALLOWED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/tiff",
    "application/octet-stream",
}


def create_app() -> FastAPI:
    checkpoint = os.getenv(
        "AEREO_CHECKPOINT",
        "artifacts/checkpoints/segformer_best",
    )
    selected_model = os.getenv(
        "AEREO_SELECTED_MODEL",
        "artifacts/checkpoints/selected_model.json",
    )
    threshold_raw = os.getenv("AEREO_THRESHOLD")
    threshold = float(threshold_raw) if threshold_raw else None
    image_size = int(os.getenv("AEREO_IMAGE_SIZE", "512"))
    resize_policy = os.getenv("AEREO_RESIZE_POLICY", "letterbox")
    log_path = os.getenv(
        "AEREO_LOG_PATH",
        "artifacts/logs/inference.jsonl",
    )
    model_version = os.getenv("AEREO_MODEL_VERSION", "segformer-v3")
    device = os.getenv("AEREO_DEVICE")
    max_upload_bytes = int(
        os.getenv("AEREO_MAX_UPLOAD_BYTES", str(20 * 1024 * 1024))
    )
    max_image_pixels = int(
        os.getenv("AEREO_MAX_IMAGE_PIXELS", "100000000")
    )
    max_concurrency = int(os.getenv("AEREO_MAX_CONCURRENCY", "1"))

    state: dict = {
        "predictor": None,
        "error": None,
        "semaphore": asyncio.Semaphore(max_concurrency),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            selected_path = Path(selected_model)
            state["predictor"] = SegFormerPredictor(
                checkpoint,
                selected_model_path=(
                    selected_path if selected_path.exists() else None
                ),
                threshold=threshold,
                image_size=image_size,
                resize_policy=resize_policy,
                device=device,
                log_path=log_path,
                model_version=model_version,
                warmup_runs=1,
            )
        except Exception as exc:
            state["error"] = f"{type(exc).__name__}: {exc}"
        yield
        state["predictor"] = None

    app = FastAPI(
        title="Aereo Water Segmentation API",
        version="3.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(HTTPException)
    async def http_error_handler(
        request: Request,
        exc: HTTPException,
    ):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "request_id": request_id,
                "error": exc.detail,
                "status_code": exc.status_code,
            },
            headers={"X-Request-ID": request_id},
        )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        if state["predictor"] is None:
            raise HTTPException(
                status_code=503,
                detail=state["error"] or "Model is not ready",
            )
        return {"status": "ready"}

    @app.get("/metadata")
    def metadata() -> dict:
        predictor = state["predictor"]
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail=state["error"] or "Model is not ready",
            )
        return {
            "model_version": predictor.model_version,
            "checkpoint_sha256": predictor.checkpoint_sha256,
            "threshold": predictor.threshold,
            "device": str(predictor.device),
            "image_size": predictor.image_size,
            "resize_policy": predictor.resize_policy,
            "classes": {"0": "non_water", "1": "water"},
            "output": "binary PNG; 0=non-water, 255=water",
            "maximum_upload_bytes": max_upload_bytes,
            "maximum_image_pixels": max_image_pixels,
            "maximum_concurrency": max_concurrency,
        }

    @app.post("/segment")
    async def segment(
        request: Request,
        image: UploadFile = File(...),
    ):
        request_id = request.headers.get(
            "X-Request-ID",
            uuid.uuid4().hex,
        )
        if image.content_type not in ALLOWED_CONTENT_TYPES:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Unsupported content type: {image.content_type}. "
                    f"Allowed: {sorted(ALLOWED_CONTENT_TYPES)}"
                ),
            )

        raw = await image.read(max_upload_bytes + 1)
        if not raw:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty",
            )
        if len(raw) > max_upload_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upload exceeds {max_upload_bytes} bytes."
                ),
            )

        try:
            pil_image = Image.open(io.BytesIO(raw))
            pil_image.load()
            pil_image = pil_image.convert("RGB")
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a readable image",
            ) from exc

        width, height = pil_image.size
        if width * height > max_image_pixels:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Image has {width * height} pixels; "
                    f"maximum is {max_image_pixels}."
                ),
            )

        predictor = state["predictor"]
        if predictor is None:
            raise HTTPException(
                status_code=503,
                detail=state["error"] or "Model is not ready",
            )

        async with state["semaphore"]:
            mask, _, metadata = await asyncio.to_thread(
                predictor.predict,
                pil_image,
                request_id=request_id,
            )

        output = io.BytesIO()
        Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
            output,
            format="PNG",
        )
        output.seek(0)
        return StreamingResponse(
            output,
            media_type="image/png",
            headers={
                "X-Request-ID": metadata["request_id"],
                "X-Water-Fraction": str(
                    metadata["predicted_water_fraction"]
                ),
                "X-Total-Latency-MS": str(metadata["total_ms"]),
                "X-Model-Version": predictor.model_version,
                "X-Checkpoint-SHA256": predictor.checkpoint_sha256,
            },
        )

    return app


app = create_app()
