# Dependency sets

- `production.in`: complete Kaggle training, HPO, evaluation, tracking, and API environment.
- `serve.in`: minimal inference/API container environment; excludes Optuna, MLflow, W&B, rasterio, plotting, and test tools.
- `ci.in`: lightweight dependencies for compilation and unit tests without model downloads.

The executed notebook exports `evidence/environment/pip_freeze.txt`, which is the exact environment lock for the successful run. Do not claim a lock file was tested until the full Kaggle execution finishes.
