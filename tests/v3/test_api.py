import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

import aereo_water.api.app as api_module


class FakePredictor:
    def __init__(self, *args, **kwargs):
        self.model_version = "fake-v1"
        self.checkpoint_sha256 = "a" * 64
        self.threshold = 0.5
        self.device = "cpu"
        self.image_size = 512
        self.resize_policy = "letterbox"

    def predict(self, image, *, request_id=None):
        width, height = image.size
        mask = np.zeros((height, width), dtype=np.uint8)
        probability = np.zeros((height, width), dtype=np.float32)
        metadata = {
            "request_id": request_id or "request",
            "predicted_water_fraction": 0.0,
            "total_ms": 1.0,
        }
        return mask, probability, metadata


def test_api_health_ready_metadata_and_valid_upload(monkeypatch):
    monkeypatch.setattr(api_module, "SegFormerPredictor", FakePredictor)
    app = api_module.create_app()
    image = Image.new("RGB", (8, 6), color=(10, 20, 30))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/ready").status_code == 200
        assert client.get("/metadata").status_code == 200
        response = client.post(
            "/segment",
            files={"image": ("sample.png", buffer.getvalue(), "image/png")},
        )
        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"


def test_api_rejects_invalid_and_missing_files(monkeypatch):
    monkeypatch.setattr(api_module, "SegFormerPredictor", FakePredictor)
    app = api_module.create_app()
    with TestClient(app) as client:
        invalid = client.post(
            "/segment",
            files={"image": ("sample.txt", b"not image", "text/plain")},
        )
        assert invalid.status_code == 415
        missing = client.post("/segment")
        assert missing.status_code == 422
