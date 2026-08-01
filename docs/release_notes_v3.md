# Aereo Water Segmentation V3

Production-oriented water segmentation using SegFormer-B0.

## Final held-out test results

- Mean per-image IoU: 0.727625
- Dice: 0.825825
- Precision: 0.860920
- Recall: 0.815868
- Boundary F1: 0.655815

## Inference performance

- P50 model-forward latency: 12.05 ms
- P95 model-forward latency: 20.81 ms
- P50 end-to-end latency: 31.91 ms
- P95 end-to-end latency: 39.53 ms
- Throughput: 73.92 images/second

## Pipeline coverage

- Validated ingestion and split registry
- Normalization and synchronized augmentation
- Raster and geospatial tiling
- Optuna hyperparameter optimization
- MLflow and offline W&B tracking
- Same-code baseline confirmation
- Seed-stability analysis
- Resumable final training
- Validation-only threshold calibration
- Frozen held-out test evaluation
- Full 2,841-image inference
- Paired statistics and performance slices
- Production predictor and FastAPI validation
- Model registry and deployment package

## Release assets

- GitHub evidence bundle
- Resume/recovery bundle
- Standalone deployment bundle
- SHA-256 checksum manifest

## Known limitation

The held-out test split contains no empty ground-truth masks. Therefore, empty-mask false-positive rate is reported as not applicable rather than assigned an artificial value.
