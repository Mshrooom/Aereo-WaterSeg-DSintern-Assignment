# SegFormer V3 Production Pipeline

## Scientific contract

- The historical train/validation/test assignment is recovered when available.
- Training, HPO, confirmation, checkpoint selection, and threshold calibration
  never receive test rows.
- HPO optimizes original-resolution mean per-image validation IoU at a fixed
  probability threshold of 0.50.
- The original SegFormer configuration is always rerun under the new code as a
  same-code control.
- The final test split is opened only after hyperparameters, checkpoint, and
  threshold are frozen.
- Historical SAM prompt modes are selected on validation, never on test.

## Execution stages

1. `data`
2. `hpo`
3. `confirmation`
4. `stability`
5. `final_train`
6. `calibrate`
7. `evaluate`
8. `inference`
9. `api_test`
10. `export`

Every stage writes evidence and resumable state under one output root.

## Tracking

MLflow is authoritative and uses:
- SQLite database: `tracking/mlflow.db`
- artifact root: `tracking/mlartifacts/`

W&B is an optional offline mirror under `tracking/wandb/`.

## Main outputs

- portable split and data registry
- exact and near-duplicate audit
- overlapping pixel and GeoTIFF tiling evidence
- Optuna study and failure ledger
- same-code baseline and top-candidate confirmation
- seed stability
- resumable final training state
- validation-only threshold sweep
- full 2,841-image inference and frozen test metrics
- calibration, paired statistics, slices, and failure cases
- model registry, model card, and deployment bundle
- predictor mask, overlay, log, and export-reload parity
- API and repository test evidence
- checksums and evidence-driven assignment compliance
