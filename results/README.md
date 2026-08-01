# Results Guide

This directory separates high-level comparison tables from detailed audit artifacts.

## Start here

Use:

```text
summary/summary_test_only.csv
```

for the primary held-out test comparison.

## Directory map

### `summary/`

Small tables intended for direct review:

- `summary_test_only.csv`: principal test-set comparison
- `summary_macro_by_split.csv`: macro metrics by experiment, prompt mode, and split
- `summary_global_by_split.csv`: pooled confusion-matrix metrics by split
- `experiment_A_test_prompt_summary.csv`: zero-shot SAM prompt comparison

### `training/`

Epoch-level histories:

- `segformer_training_history.csv`
- `sam_training_history.csv`

Losses should be compared within a model family, not directly across different training objectives.

### `calibration/`

Validation-only output calibration:

- `experiment_D_threshold_sweep.csv`
- `threshold_sweeps/`: epoch-specific threshold tables

### `registry/`

Model and artifact metadata:

- `model_registry.csv`

### `full/`

Complete per-image registries:

- `experiment_A_zero_shot_sam_all_2841.csv`
- `experiment_B_finetuned_sam_all_2841.csv`
- `experiment_C_segformer_all_2841.csv`
- `experiment_D_auto_sam_all_2841.csv`

These files support auditing, error analysis, and reconstruction of the deterministic split. They are not the first files a reviewer needs to open.

### `hpo/`

Measured Optuna and MLflow pilot-study outputs will be placed here:

- `trials.csv`
- `best_trial.json`
- `mlflow_runs.csv`
- optimization figures
- study metadata

### `figures/`

Report-ready plots.

### `sample_predictions/`

A small curated set of input, ground-truth, and prediction examples.

## Interpretation caveats

- Experiments A and B use oracle prompts derived from masks.
- Experiment D is an automatic hybrid ablation, not the production model.
- Hybrid latency excludes part of the separately materialized coarse stage.
- Thresholds were selected on validation data.
- The held-out test split is reserved for final comparison.
