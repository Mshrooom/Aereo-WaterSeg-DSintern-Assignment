# Repository Architecture

## Active production path

```text
Satellite image
    ↓
Input validation and preprocessing
    ↓
SegFormer checkpoint loader
    ↓
Probability map
    ↓
Validation-selected threshold
    ↓
Binary water mask
    ↓
FastAPI response / PNG output
```

## Experiment lifecycle

```text
Raw image-mask pairs
    ↓
Pair discovery and validation
    ↓
Deterministic train/validation/test registry
    ↓
Normalization and augmentation
    ↓
SegFormer training
    ↓
Validation checkpoint and threshold selection
    ↓
Held-out test evaluation
    ↓
Model registry and result tables
    ↓
Inference API
    ↓
Docker image
```

## Source boundaries

### Active package: `src/waterseg/`

Reusable code used by the maintained data, evaluation, inference, serving, logging, and utility paths.

### Research archive: `research/legacy_training/`

Original experimental modules whose complete dependency bundle is not part of the cleaned active package.

### Notebooks

Notebooks document orchestration and measured studies. Reusable logic should live in `src/` or `scripts/`, not only in notebook cells.

## Model decision

SegFormer-B0 is the production model because it achieved the best automatic test IoU, Dice, boundary F1, and recorded model-stage latency.

SAM-based experiments remain comparative research evidence.
