FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    AEREO_DEVICE=cpu \
    AEREO_CHECKPOINT=/app/artifacts/checkpoints/segformer_best \
    AEREO_SELECTED_MODEL=/app/artifacts/checkpoints/selected_model.json \
    AEREO_LOG_PATH=/app/artifacts/logs/inference.jsonl

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/production.in /app/requirements/production.in

RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements/production.in

COPY pyproject.toml README.md /app/
COPY src /app/src

RUN python -m pip install --no-deps -e /app

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/artifacts/checkpoints /app/artifacts/logs \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD curl --fail http://localhost:8000/ready || exit 1

CMD ["uvicorn", "aereo_water.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
