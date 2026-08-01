# Notebook Guide

## `01_four_experiments_research_record.ipynb`

The original orchestration notebook for the four-experiment study.

Important limitation: it relied on an earlier source bundle that is not identical to the cleaned public package. It is retained for methodological transparency and result provenance, not as the primary clone-and-run entry point.

## `02_segformer_optuna_mlflow_hpo.ipynb`

The constrained SegFormer-B0 pilot hyperparameter study.

It should:

- reconstruct the original deterministic split;
- use fixed training and validation pilot subsets;
- use Optuna for search;
- use MLflow for local tracking;
- optimize validation IoU;
- avoid the held-out test split;
- export all measured results to `results/hpo/`.

## Notebook policy

Executed notebooks should preserve useful tables and plots but avoid embedding thousands of generated masks. Large checkpoints and bulk artifacts belong in Kaggle outputs or a GitHub release.
