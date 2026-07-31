FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    USE_TORCH=1 \
    USE_TF=0 \
    USE_FLAX=0 \
    TOKENIZERS_PARALLELISM=false \
    SEGFORMER_CHECKPOINT=/models/segformer_best \
    DEVICE=cpu \
    LOG_LEVEL=INFO

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/serve-segformer.txt /app/requirements/serve-segformer.txt
RUN python -m pip install --upgrade pip \
    && pip install -r /app/requirements/serve-segformer.txt

COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /models \
    && chown -R appuser:appuser /app /models

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl --fail http://127.0.0.1:8000/ready || exit 1

CMD ["uvicorn", "waterseg.segformer_api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
