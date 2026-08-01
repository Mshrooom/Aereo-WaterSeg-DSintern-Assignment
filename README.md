# Aereo Water-Body Segmentation

A research and deployment project for binary water-body segmentation in Sentinel-2-derived RGB satellite imagery.

## Final production decision

**SegFormer-B0 is the selected production model.** It provided the strongest automatic segmentation quality, the best boundary accuracy, and substantially lower recorded model-stage latency than the SAM-based alternatives.

| Experiment | Operating mode | Test IoU | Test Dice | Boundary F1 | Recorded latency |
|---|---|---:|---:|---:|---:|
| Zero-shot SAM | Oracle prompts | 0.5667 | 0.6890 | 0.3544 | 235.1 ms |
| Fine-tuned SAM | Oracle prompts | 0.6043 | 0.7348 | 0.4021 | 237.1 ms |
| **SegFormer-B0** | **Fully automatic** | **0.7190** | **0.8191** | **0.5985** | **8.3 ms** |
| SegFormer–SAM hybrid | Fully automatic ablation | 0.6093 | 0.7352 | 0.4131 | >242.8 ms* |

\*The hybrid timing is a lower bound because the saved coarse SegFormer stage was materialized separately.

## Dataset

The project uses 2,841 image–mask pairs from the Kaggle **Satellite Images of Water Bodies** dataset.

Fixed split:

- Training: 1,991
- Validation: 429
- Test: 421

The dataset itself is not committed to Git.

## Repository guide

| Path | Purpose |
|---|---|
| `src/waterseg/` | Active reusable ingestion, evaluation, inference, API, logging, and utility code |
| `notebooks/` | Research record and hyperparameter-study notebooks |
| `results/summary/` | Main comparison tables |
| `results/training/` | Epoch-level training histories |
| `results/calibration/` | Validation threshold sweeps |
| `results/full/` | Full per-image experiment registries |
| `results/hpo/` | Optuna and MLflow pilot-study outputs |
| `research/` | Archived legacy experimental code and caveats |
| `docs/` | Architecture, reproducibility, report, and archived documentation |
| `Dockerfile` | SegFormer inference image |
| `docker-compose.yml` | Local inference-service configuration |

## Read this first

- [Results guide](results/README.md)
- [Repository architecture](docs/architecture.md)
- [Reproducibility status](docs/reproducibility.md)
- [Notebook guide](notebooks/README.md)
- [Legacy research notes](research/README.md)

## Experimental scope

Four configurations were studied:

1. Zero-shot SAM with mask-derived oracle prompts
2. Fine-tuned SAM with mask-derived oracle prompts
3. Fully automatic SegFormer-B0
4. Automatic SegFormer-to-SAM hybrid ablation

The SAM baselines are research comparisons, not deployable automatic baselines, because their evaluation prompts were derived from ground-truth masks.

## Metrics

The result registry includes:

- IoU and Dice
- Precision, recall, specificity, and pixel accuracy
- Balanced accuracy, MCC, and Cohen's kappa
- Boundary precision, recall, F1, and IoU
- HD95 and ASSD
- Predicted water fraction
- Recorded model-stage latency

Use `results/summary/summary_test_only.csv` for the primary model comparison.

## Hyperparameter study

A constrained SegFormer-B0 pilot study is being added using:

- **Optuna** for search and trial selection
- **MLflow** for local experiment tracking
- Fixed train and validation subsets
- Validation IoU as the objective
- No test-set access during tuning

Measured HPO outputs will be stored under `results/hpo/`.

## Inference and Docker

The production path is the SegFormer inference service.

Expected local checkpoint directory:

```text
artifacts/checkpoints/segformer_best/
```

Build and start:

```powershell
docker compose config
docker compose build
docker compose up -d
docker compose ps
docker compose logs --tail=200
```

The Docker image has been built successfully. End-to-end container and endpoint smoke testing should be recorded separately before claiming runtime validation.

## Research record versus reproducible production code

The original four-experiment notebook is retained as a research record. It depended on an earlier experimental source bundle and should not be presented as clone-and-run production code.

The active repository is being consolidated around a self-contained SegFormer training, tuning, inference, and deployment path. See [reproducibility status](docs/reproducibility.md).

## Author

Maanvi Bansal
