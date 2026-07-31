FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_CHECKPOINT=/models/best.pt \
    DEVICE=cpu

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/ /app/requirements/
RUN pip install --no-cache-dir -r /app/requirements/api.txt

COPY pyproject.toml README.md /app/
COPY src/ /app/src/
RUN pip install --no-cache-dir --no-deps .

RUN useradd --create-home --uid 10001 appuser && mkdir -p /models && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "waterseg.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
