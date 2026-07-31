# Aereo Water SAM

A production-oriented machine-learning system for segmenting water bodies in satellite imagery with Segment Anything (SAM). It turns a two-image prompt demo into a reproducible pipeline covering data validation, deterministic splitting, full-dataset fine-tuning, prompt comparison, calibration and boundary metrics, inference, API serving, Docker, tests, CI, and optional W&B tracking.

## Why the design includes a no-prompt mode

Point and box prompts generated from ground-truth masks are useful for a controlled interaction study, but they do not exist for a new image. The training curriculum therefore includes `none`, which fine-tunes SAM's mask decoder to predict the full water mask without an oracle prompt. The test report separates deployable no-prompt performance from interactive point/box performance.

## Repository layout

```text
configs/                 Reproducible experiment configuration
notebooks/               Kaggle orchestration and analysis notebook
src/waterseg/data/       Pairing, validation, split, augmentation, tiling
src/waterseg/models/     Hugging Face SAM wrapper and compact checkpoints
src/waterseg/            Losses, metrics, engine, evaluation, inference, API
requirements/            Kaggle, training, API, and development dependencies
tests/                   Unit tests without model downloads
reports/                 Report structure and generated tables
.github/workflows/       CI
Dockerfile               CPU API container
```

## Kaggle training

1. Create a **fresh GPU notebook** and attach the `Satellite Images of Water Bodies` dataset.
2. Add this repository as a Kaggle dataset or clone it into `/kaggle/working/aereo-water-sam`.
3. Do not upgrade NumPy, SciPy, OpenCV, or Albumentations in a running Kaggle kernel.
4. Run:

```bash
%pip install -q -r /kaggle/working/aereo-water-sam/requirements/kaggle.txt
%pip install -q -e /kaggle/working/aereo-water-sam --no-deps
```

Restart only if Kaggle requests it, then:

```bash
!python -m waterseg.cli.prepare --config /kaggle/working/aereo-water-sam/configs/sam_vit_b.yaml
!python -m waterseg.cli.train --config /kaggle/working/aereo-water-sam/configs/sam_vit_b.yaml
!python -m waterseg.cli.evaluate --config /kaggle/working/aereo-water-sam/configs/sam_vit_b.yaml
```

Training uses every image assigned to the training split; validation and prompt comparisons use every image in their respective splits. There is no example-count limiter.

### Recommended Kaggle settings

- SAM ViT-B rather than ViT-H for reliable T4/P100 training.
- Batch size 2, gradient accumulation 4, mixed precision.
- Train the mask decoder first. Optionally set `trainable_parts: mask_decoder_and_last_blocks` and `unfreeze_last_vision_blocks: 2` for a second low-learning-rate experiment.
- Set W&B to `offline` until the complete run is stable; change tracking provider to `wandb` and add the Kaggle secret afterward.

## Metrics

The final evaluator produces both global pixel metrics and per-image macro metrics:

- IoU/Jaccard and Dice/F1
- Precision, recall/sensitivity, specificity
- Pixel and balanced accuracy
- Matthews correlation coefficient and Cohen's kappa
- Boundary precision/recall/F1 and boundary IoU
- HD95 and average symmetric surface distance
- ECE, Brier score, histogram AUROC and AUPRC
- Per-image latency and water-area error
- Bootstrap 95% confidence intervals for IoU, Dice, and boundary F1

The classification threshold is selected only on validation data and then frozen for the test set.

## Local inference

```bash
waterseg-infer \
  --checkpoint artifacts/checkpoints/best.pt \
  --image sample.png \
  --output predicted_mask.png
```

Interactive box example:

```bash
waterseg-infer --checkpoint artifacts/checkpoints/best.pt --image sample.png \
  --output predicted_mask.png --box '[20,30,900,700]'
```

## API and Docker

```bash
docker build -t aereo-water-sam .
docker run --rm -p 8000:8000 \
  -v "$PWD/artifacts/checkpoints:/models:ro" \
  -e MODEL_CHECKPOINT=/models/best.pt \
  aereo-water-sam
```

```bash
curl -X POST http://localhost:8000/segment \
  -F image=@sample.png \
  --output mask.png
```

Optional prompt fields are JSON strings:

```bash
curl -X POST http://localhost:8000/segment \
  -F image=@sample.png \
  -F 'points=[[120,80],[300,200]]' \
  -F 'labels=[1,0]' \
  --output mask.png
```

Use one Uvicorn worker per GPU model replica. Horizontal scaling is safer than loading multiple large SAM copies into one GPU process.

## Production optimizations already included

- Frozen foundation-model layers by default
- AMP and gradient accumulation
- Compact checkpoints containing trainable weights plus the base model identifier
- Deterministic prompts and leakage-safe split
- Validation-only threshold tuning
- Batched training/evaluation
- Overlap-tile inference with weighted stitching
- Optional connected-component cleanup
- Structured JSON logs, health/readiness endpoints
- Unit tests and CI

## Honest limitations

- Original SAM is promptable object segmentation, not inherently a water classifier. The no-prompt domain adaptation is the deployable semantic experiment and must be reported separately.
- Oracle point/box prompts use ground truth and are benchmark-only.
- RGB images do not expose Sentinel-2 NIR/SWIR bands, so the model cannot directly learn NDWI/MNDWI evidence.
- This repository is syntax- and unit-tested without downloading model weights or running the 2,841-image training job. Final performance numbers must come from the Kaggle run.

## Hyperparameter tuning with W&B Sweeps

Prepare the dataset registry once, then create and run a sweep from the repository root:

```bash
waterseg-prepare --config configs/sam_vit_b.yaml
wandb sweep configs/wandb_sweep.yaml
wandb agent <entity/project/sweep_id>
```

Each run reuses the same manifest and split, writes to `output_dir/sweeps/<run_id>`, and searches learning rate, weight decay, loss weights, and the no-prompt sampling weight. The sweep optimizes validation IoU after validation-only threshold tuning.

## Resume an interrupted Kaggle run

Set this in the YAML configuration:

```yaml
train:
  resume_checkpoint: /kaggle/working/aereo-water-sam-output/checkpoints/last.pt
```

The trainer restores trainable model weights, optimizer, scheduler, AMP scaler, early-stopping counters, and random-number-generator states from `training_state_last.pt`.

## Optional materialized tiling

For genuinely large rasters, set:

```yaml
data:
  materialize_tiles: true
  tile_size: 1024
  tile_overlap: 128
```

The pipeline splits parent images first and only then creates tiles, preventing tiles from the same source raster from leaking across train, validation, and test sets. Small images remain unpadded and use their original files.

## GeoTIFF inference

Install the optional geospatial dependency:

```bash
pip install -e ".[geo]"
```

When the input and output are GeoTIFF files, the CLI preserves the source CRS, transform, dimensions, and other compatible profile metadata in a single-band `uint8` output mask:

```bash
waterseg-infer \
  --checkpoint artifacts/checkpoints/best.pt \
  --image input_scene.tif \
  --output water_mask.tif
```
