# Repository layout

The repository separates maintained implementation, reproducibility evidence,
human-facing reports, generated artifacts, deployment files, and historical
material.

- `src/`: maintained Python implementation. Both existing packages are retained
  during this cleanup to avoid breaking imports.
- `scripts/`: command-line training, evaluation, inference, and release helpers.
- `notebooks/`: source and executed notebooks. They are intentionally not moved
  by the cleanup script because relative paths may be embedded in notebook cells.
- `configs/`: pipeline and acceptance-criteria configuration.
- `tests/`: unit and integration tests.
- `evidence/`: small committed JSON/CSV/TXT evidence required to audit results.
- `reports/`: figures, tables, report material, and presentation assets.
- `artifacts/`: large generated checkpoints, predictions, overlays, logs, MLflow,
  and W&B output. This directory is ignored except for its README.
- `deployment/`: canonical Dockerfile and Compose configuration.
- `legacy/`: older deployment files retained for reference rather than deleted.

The V3 Docker configuration is canonical after restructuring. External Docker
runtime validation remains a separate evidence step.