# Water-Body Segmentation Report

## 1. Executive summary
State the best test IoU/Dice, the deployment prompt mode, latency, and the main limitation.

## 2. Dataset and data quality
Report pair count, dimensions, water-coverage distribution, empty masks, duplicates, corrupt files, and split sizes.

## 3. Method
Describe SAM ViT-B, frozen/trainable components, mixed prompt curriculum, automatic no-prompt mode, augmentations, loss, optimizer, scheduler, and threshold tuning.

## 4. Experiments
Include the training curve, validation threshold sweep, complete prompt comparison, calibration results, boundary metrics, confidence intervals, and failure-case examples.

## 5. Production pipeline
Describe manifest versioning, deterministic split, checkpoint metadata, W&B/local tracking, inference CLI, FastAPI contract, Docker image, structured logging, health checks, tests, and CI.

## 6. Limitations
Oracle point/box prompts are benchmark-only. New-image deployment should use no-prompt mode or real user prompts. Dataset masks may be index-derived rather than manually delineated. RGB inputs omit Sentinel-2 NIR/SWIR information that is highly relevant to water detection.

## 7. Recommendations
Prioritize multispectral bands, geographic holdout evaluation, a semantic baseline, active-learning prompts, quantization/ONNX benchmarking, and drift monitoring.
