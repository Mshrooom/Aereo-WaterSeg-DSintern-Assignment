# Migration from the original notebook

## Original proof of concept

The uploaded notebook loads official SAM ViT-H, selects two individual JPEGs, uses manually hard-coded foreground clicks, and visualizes the three masks returned by SAM. It demonstrates that the checkpoint and GPU inference work, but it does not use the full dataset or quantify water-segmentation quality.

## Production pipeline replacement

| Area | Original notebook | Extended system |
|---|---|---|
| Dataset | Two selected images | Manifest over all paired images and masks |
| Validation | None | Decode, shape, duplicate, coverage, and leakage checks |
| Split | None | Deterministic stratified train/validation/test split |
| Model | Zero-shot ViT-H | Fine-tuned SAM ViT-B, with optional last-block unfreezing |
| Deployment mode | Requires manually chosen points | No-prompt semantic adaptation plus optional interaction |
| Prompts | One or two positive points | None, one point, positive/negative points, box, box plus points |
| Training | None | AMP, gradient accumulation, clipping, warm-up, cosine schedule, early stopping, resume |
| Loss | None | BCE + Dice + focal + IoU-head regression |
| Metrics | SAM predicted score only | Region, boundary, calibration, surface-distance, latency, confidence intervals |
| Threshold | Implicit | Tuned on validation only and frozen for test |
| Tracking | None | Local JSON/CSV plus optional W&B runs, artifacts, registry, and sweeps |
| Large rasters | None | Leakage-safe materialized tiling and overlap-tile inference |
| Inference | Notebook cells | Reusable class and CLI with interactive or automatic mode |
| Geospatial output | None | Optional CRS/transform-preserving GeoTIFF mask output |
| Serving | None | FastAPI, structured logs, health/readiness, embedding cache |
| Packaging | None | Docker, Compose, `pyproject.toml`, Makefile |
| Quality | None | Unit tests and GitHub Actions CI |

## Recommended run order

1. Run `aereo_sam_production_pipeline.ipynb` on a fresh Kaggle GPU session.
2. Complete the default mask-decoder experiment.
3. Review automatic no-prompt test performance and failure cases.
4. Run the W&B sweep only after the default pipeline is stable.
5. Run the optional final-two-vision-block experiment as a separate tracked run.
6. Copy the best compact checkpoint into `artifacts/checkpoints/best.pt` and build the container.
