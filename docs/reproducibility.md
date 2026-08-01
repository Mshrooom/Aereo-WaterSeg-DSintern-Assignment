# Reproducibility Status

## Fully evidenced

- Fixed 1,991/429/421 split
- Per-image results for all four experiments
- Test-only and split-level summary tables
- SegFormer and SAM training histories
- Validation threshold sweeps
- Model registry metadata
- SegFormer inference and API implementation
- Docker build configuration

## Research record only

The original four-experiment notebook relied on an earlier experimental source bundle. The retained notebook and archived modules document that work, but the cleaned repository should not claim exact clone-and-run reproduction of every SAM and hybrid training stage.

## Active reproducibility target

The maintained reproducible path is:

1. Prepare image-mask manifests.
2. Train SegFormer-B0.
3. Run Optuna search with MLflow tracking.
4. Select checkpoint and threshold using validation data.
5. Evaluate once on the held-out test split.
6. Export a Hugging Face-compatible checkpoint.
7. Serve through FastAPI and Docker.

## Deployment validation

Current honest status:

- Docker image build: completed
- Full fresh-clone container smoke test: pending external confirmation

Do not change this status until `/health`, `/ready`, `/metadata`, `/segment`, invalid-input handling, and restart behaviour have been tested.
