# Aereo Water-Body Segmentation — Data Science Intern Assignment

[![CI](https://github.com/Mshrooom/Aereo-WaterSeg-DSintern-Assignment/actions/workflows/ci.yml/badge.svg)](https://github.com/Mshrooom/Aereo-WaterSeg-DSintern-Assignment/actions/workflows/ci.yml)
[![Release](https://img.shields.io/badge/release-v3.0.0-blue)](https://github.com/Mshrooom/Aereo-WaterSeg-DSintern-Assignment/releases/tag/v3.0.0)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Segmentation-orange)](https://pytorch.org/)

A complete comparative and production-oriented study of **water-body segmentation from RGB satellite imagery**.

The repository evaluates five systems:

1. zero-shot Segment Anything Model (SAM);
2. water-specific fine-tuned SAM;
3. direct SegFormer-B0 semantic segmentation;
4. an automatic SegFormer-to-SAM refinement pipeline;
5. a governed SegFormer-B0 optimization experiment.

The project demonstrates more than a final model score. It includes validated data ingestion, leakage-controlled splitting, synchronized preprocessing, promptable and automatic segmentation experiments, Optuna search, MLflow tracking, seed-stability analysis, validation-only threshold calibration, frozen held-out evaluation, paired statistics, failure slices, model and data registries, FastAPI inference, Docker packaging, continuous integration, and versioned evidence.

The governed SegFormer-B0 checkpoint is the **selected automatic deployment candidate** because it provides the strongest combined automatic overlap, boundary quality, latency, and operational simplicity. Fine-tuned SAM remains an analyst-assisted research option, while the hybrid is preserved as a valuable negative result.

---

## Contents

1. [Project status](#project-status)
2. [Key results](#key-results)
3. [Experiment suite](#experiment-suite)
4. [System architecture](#system-architecture)
5. [Dataset and governance](#dataset-and-governance)
6. [Repository structure](#repository-structure)
7. [Quick start](#quick-start)
8. [Obtain the model artifact](#obtain-the-model-artifact)
9. [Run direct Python inference](#run-direct-python-inference)
10. [Run the FastAPI service](#run-the-fastapi-service)
11. [Recreate the Docker image and run the container](#Docker-Deployment)
12. [Reproduce the experiments](#reproduce-the-experiments)
13. [Evidence, tracking, and registries](#evidence-tracking-and-registries)
14. [Tests and continuous integration](#tests-and-continuous-integration)
15. [Limitations and future work](#limitations-and-future-work)
16. [Documentation](#documentation)

---

# Project status

| Component | Status |
|---|---|
| Dataset validation and deterministic split registry | Completed |
| Zero-shot SAM experiment | Completed |
| Fine-tuned SAM experiment | Completed |
| Historical SegFormer experiment | Completed |
| Automatic SegFormer-to-SAM hybrid | Completed |
| Governed SegFormer optimization | Completed |
| Frozen held-out test evaluation | Completed |
| Paired bootstrap and Wilcoxon analysis | Completed |
| MLflow experiment tracking and registries | Completed |
| FastAPI in-process validation | Completed |
| Repository test suite | **43 passed** |
| GitHub Actions CI | Configured in `.github/workflows/ci.yml` |
| Dockerfile and Compose packaging | Provided |
| External Docker build/start/restart validation | **Pending until executed and preserved as evidence** |

The repository is production-oriented, but it does not claim that external Docker runtime validation is complete before the real container lifecycle has been executed.

---

# Key results

## Dataset

| Item | Value |
|---|---:|
| Image-mask pairs | 2,841 |
| Training images | 1,991 |
| Validation images | 429 |
| Held-out test images | 421 |
| Input modality | RGB satellite imagery |
| Target | Binary water/non-water mask |
| Model input | 512 × 512 letterboxed |
| Service output | Original-resolution binary PNG |

The held-out test set contains **zero empty masks**. Empty-scene false-positive robustness is therefore `NOT_APPLICABLE`, not zero and not passed.

## Selected automatic system

| Item | Value |
|---|---|
| Architecture | SegFormer-B0 |
| Pretrained checkpoint | `nvidia/segformer-b0-finetuned-ade-512-512` |
| Output classes | 2: non-water, water |
| Resize policy | Aspect-ratio-preserving letterbox |
| Validation-selected threshold | 0.45 |
| Loss | Cross-entropy + Dice |
| Selected loss weights | CE 0.4, Dice 0.6 |
| Augmentation profile | Light |
| Experiment tracker | MLflow |
| Optional visualization mirror | W&B offline |
| Model version | `segformer-v3.0.0` |

## Final held-out metrics

| Metric | Result |
|---|---:|
| Mean per-image IoU | **0.727625** |
| Dice | **0.825825** |
| Precision | **0.860920** |
| Recall | **0.815868** |
| Specificity | **0.910250** |
| Pixel accuracy | **0.896098** |
| Balanced accuracy | **0.863059** |
| Matthews correlation coefficient | **0.698699** |
| Cohen’s kappa | **0.728508** |
| Boundary F1 | **0.655815** |
| Boundary IoU | **0.196826** |
| Global IoU | **0.904199** |
| Global Dice | **0.949689** |

Mean per-image IoU is the primary comparison statistic because source image dimensions vary substantially. Global metrics are retained as secondary pixel-weighted evidence.

## Paired improvement over the historical SegFormer

| Statistic | Result |
|---|---:|
| Mean paired IoU difference | **+0.008631** |
| 95% paired-bootstrap confidence interval | **[0.002960, 0.014245]** |
| Wilcoxon signed-rank p-value | **1.14 × 10⁻⁹** |
| Images improved | 61.28% |
| Images degraded | 34.92% |
| Images unchanged | 3.80% |

## Measured inference performance

Measured on a Kaggle Tesla T4 with batch size 1:

| Metric | Result |
|---|---:|
| Cold start | 143.89 ms |
| P50 model-forward | 12.05 ms |
| P95 model-forward | 20.81 ms |
| Mean model-forward | 13.53 ms |
| P50 end-to-end | 31.91 ms |
| P95 end-to-end | 39.53 ms |
| Mean end-to-end | 33.34 ms |
| Model-forward throughput | 73.92 images/s |
| Peak inference GPU memory | 283.62 MB |

Model-forward and end-to-end timings measure different scopes and should not be combined. Performance will vary with hardware, drivers, storage, input dimensions, and service concurrency.

---

# Experiment suite

| System | Inference input | Adaptation | Scientific purpose | Deployment interpretation |
|---|---|---|---|---|
| Zero-shot SAM | RGB + oracle prompt | None | Measure prompt sensitivity and foundation-model transfer | Controlled benchmark only |
| Fine-tuned SAM | RGB + oracle/user prompt | Water-specific mask-decoder adaptation | Measure domain adaptation under matched prompts | Analyst-assisted option |
| Historical SegFormer | RGB only | End-to-end semantic segmentation | Establish the original automatic baseline | Strong original candidate |
| SegFormer-to-SAM hybrid | RGB only; prompts generated internally | Reuses SegFormer and fine-tuned SAM | Test whether promptable refinement improves automatic masks | Complex negative ablation |
| Governed SegFormer | RGB only | Optuna, same-code confirmation, three seeds, resumable training, calibration | Test controlled improvement of the strongest automatic family | Selected deployment candidate |

## Representative held-out comparison

| System | Automatic? | IoU | Dice | Precision | Recall | Boundary F1 |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot SAM, multiple points | No | 0.566666 | 0.688989 | 0.8113 | 0.6701 | 0.354419 |
| Fine-tuned SAM, representative prompted mode | No | 0.603761 | 0.735317 | 0.7670 | 0.7395 | 0.392837 |
| Historical SegFormer | Yes | 0.718995 | 0.819071 | 0.8596* | 0.8059* | 0.598512 |
| SegFormer-to-SAM hybrid | Yes | 0.609329 | 0.735206 | 0.6914 | 0.8495 | 0.413059 |
| Governed SegFormer | Yes | 0.727625 | 0.825825 | 0.860920 | 0.815868 | 0.655815 |

`*` Historical precision and recall are retained at the precision available in the archived result summary.

The hybrid recovers more water pixels but introduces substantially more false positives, lowers overlap and boundary quality, increases latency, and adds another model, threshold, prompt-generation stage, and fallback path.

---

# System architecture

## Inference path

```text
RGB satellite image
        │
        ▼
Request and image validation
        │
        ▼
Aspect-ratio-preserving letterbox preprocessing
        │
        ▼
Registered SegFormer-B0 checkpoint
        │
        ▼
Two-class logits and water probability
        │
        ▼
Restore probability map to original dimensions
        │
        ▼
Validation-selected threshold = 0.45
        │
        ▼
Binary water mask
  0   = non-water
  255 = water
        │
        ├── PNG response
        ├── optional overlay
        └── structured JSONL request record
```

## Governed experiment path

```text
Validated image-mask manifest
        │
        ▼
Deterministic train/validation/test registry
        │
        ▼
Exact and perceptual duplicate audit
        │
        ▼
Synchronized augmentation and letterbox preprocessing
        │
        ▼
Optuna search on train/validation only
        │
        ▼
Historical same-code baseline + top-candidate confirmation
        │
        ▼
Three-seed stability: 42, 2026, 3407
        │
        ▼
Resumable final training
        │
        ▼
Validation-only threshold calibration
        │
        ▼
Model-selection lock
        │
        ▼
Frozen 421-image held-out evaluation
        │
        ▼
Statistics, slices, registries, inference, API, and release export
```

---

# Dataset and governance

The project uses the **Satellite Images of Water Bodies** dataset:

- 2,841 RGB satellite images;
- 2,841 corresponding binary masks;
- variable source dimensions;
- binary water/non-water targets;
- PNG, TIFF, and GeoTIFF-compatible ingestion paths.

## Deterministic split

| Split | Count | Permitted use |
|---|---:|---|
| Training | 1,991 | Weight optimization and training-only transformations |
| Validation | 429 | HPO, early stopping, confirmation, checkpoint selection, and threshold calibration |
| Held-out test | 421 | Final reporting only, after decisions are locked |
| **Total** | **2,841** | Registered corpus |

Parent images are assigned to a split before optional tiling. Tiles from one parent image must not cross split boundaries.

## Validation and leakage controls

The ingestion pipeline:

1. discovers image and mask candidates recursively;
2. pairs files using stable identifiers;
3. rejects unreadable, duplicate, missing, or incompatible pairs;
4. preserves original dimensions;
5. normalizes mask interpretation;
6. measures water coverage;
7. computes SHA-256 evidence;
8. audits exact and perceptual duplicates;
9. writes portable manifests;
10. writes a deterministic split registry consumed by every experiment.

Key evidence:

```text
evidence/registry/validated_manifest.csv
evidence/registry/runtime_manifest.csv
evidence/registry/split_registry.csv
evidence/registry/data_registry.json
evidence/registry/near_duplicate_audit.csv
evidence/registry/model_selection_lock.json
```

---

# Repository structure

```text
.
├── .github/
│   └── workflows/              GitHub Actions CI
├── configs/                    Pipeline and acceptance configuration
├── deployment/                 Canonical Dockerfile and Compose configuration
├── docs/                       Dataset/model cards, architecture, runbook, release notes
├── evidence/                   Small committed experimental evidence
│   ├── acceptance/
│   ├── api/
│   ├── calibration/
│   ├── environment/
│   ├── evaluation/
│   ├── inference/
│   ├── registry/
│   ├── results/
│   ├── run_state/
│   ├── slices/
│   └── statistics/
├── legacy/                     Historical code and deployment assets
├── notebooks/                  Source, full-run, executed, and historical notebooks
├── reports/                    Human-facing figures and report assets
├── requirements/               Environment profiles
├── scripts/                    Training, evaluation, inference, and export utilities
├── src/
│   ├── aereo_water/            Maintained V3 package
│   │   ├── api/
│   │   ├── data/
│   │   ├── evaluation/
│   │   ├── inference/
│   │   ├── models/
│   │   ├── pipeline/
│   │   └── training/
│   └── waterseg/               Historical installed package
├── tests/                      Unit and integration tests
├── .dockerignore
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

Large generated artifacts are intentionally excluded from Git:

```text
artifacts/checkpoints/
artifacts/predictions/
artifacts/overlays/
artifacts/logs/
artifacts/mlruns/
artifacts/wandb/
```

Obtain checkpoints through the GitHub release or recreate them through the full experiment.

## Where to look first

| Goal | Path |
|---|---|
| Review the original comparative assignment | `notebooks/aereo-task-original.ipynb` |
| Review the complete A–D experiment run | `notebooks/aereo-task-maanvi-bansal (1).ipynb` |
| Inspect SAM zero-shot and fine-tuning | `notebooks/aereo_sam_production_pipeline.ipynb` |
| Run a governed V3 smoke validation | `notebooks/01_complete_segformer_production_pipeline_source.ipynb` |
| Recreate governed Experiment E | `notebooks/01_complete_segformer_production_pipeline_full_run.ipynb` |
| Review executed governed V3 evidence | `notebooks/Aereo_Production_SegFormer_V3_EXECUTED.ipynb` |
| Inspect final metrics | `evidence/evaluation/segformer_v3_test_metrics.json` |
| Inspect all V3 per-image rows | `evidence/evaluation/segformer_v3_all_2841.csv` |
| Inspect historical experiments | `evidence/results/full/` |
| Inspect threshold calibration | `evidence/calibration/` |
| Inspect paired statistics | `evidence/statistics/` |
| Inspect failure slices | `evidence/slices/` |
| Inspect selected model metadata | `evidence/registry/selected_model.json` |
| Inspect API evidence | `evidence/api/` |
| Inspect latency evidence | `evidence/inference/` |
| Build the service | `deployment/Dockerfile`, `deployment/compose.yaml` |

---

# Quick start

> **Use a fresh virtual environment. Do not install the project into a shared or global Python environment.**

## Windows PowerShell

```powershell
git clone `
  https://github.com/Mshrooom/Aereo-WaterSeg-DSintern-Assignment.git

Set-Location Aereo-WaterSeg-DSintern-Assignment

py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip setuptools wheel
python -m pip install --editable .
python -m pip check
```

Verify that Python comes from the repository virtual environment:

```powershell
python -c "import sys; print(sys.executable)"
```

The printed path should contain:

```text
Aereo-WaterSeg-DSintern-Assignment\.venv\
```

## Local tests

```powershell
python -m pip install -r .\requirements\ci.in
python -m compileall -q src scripts
python -m pytest -q
python -m pip check
```

Expected repository evidence:

```text
43 passed
```

The latest GitHub Actions run is the source of truth for the current commit.

---

# Obtain the model artifact

The selected checkpoint is not assumed to exist in a fresh clone.

Download the deployment bundle from:

**[GitHub Release v3.0.0](https://github.com/Mshrooom/Aereo-WaterSeg-DSintern-Assignment/releases/tag/v3.0.0)**

Expected release assets include:

```text
aereo-water-v3-github-evidence.zip
aereo-water-segformer-v3-deployment.zip
AEREO_V3_SHA256SUMS.txt
```

## Verify the downloaded bundle

```powershell
Get-FileHash `
  .\aereo-water-segformer-v3-deployment.zip `
  -Algorithm SHA256

Get-Content .\AEREO_V3_SHA256SUMS.txt
```

Compare the calculated hash with the checksum manifest before extraction.

## Extract and place the checkpoint

The following PowerShell block searches the extracted bundle rather than assuming one fixed ZIP directory layout:

```powershell
$Bundle = ".\aereo-water-segformer-v3-deployment.zip"
$ExtractRoot = ".\artifacts\deployment_bundle"
$CheckpointRoot = ".\artifacts\checkpoints"

New-Item -ItemType Directory -Force $ExtractRoot | Out-Null
New-Item -ItemType Directory -Force $CheckpointRoot | Out-Null

Expand-Archive `
  -Path $Bundle `
  -DestinationPath $ExtractRoot `
  -Force

$Checkpoint = Get-ChildItem `
  -Path $ExtractRoot `
  -Directory `
  -Recurse `
  -Filter "segformer_best" |
  Select-Object -First 1

if (-not $Checkpoint) {
    throw "segformer_best was not found in the deployment bundle."
}

$Target = Join-Path $CheckpointRoot "segformer_best"

if (Test-Path $Target) {
    Remove-Item -Recurse -Force $Target
}

Copy-Item `
  -Path $Checkpoint.FullName `
  -Destination $Target `
  -Recurse `
  -Force

Write-Host "Checkpoint installed at: $Target"
```

Confirm:

```powershell
Test-Path ".\artifacts\checkpoints\segformer_best"
Test-Path ".\evidence\registry\selected_model.json"
```

Both should return `True` before starting inference or Docker.

---

# Run direct Python inference

Create `run_v3_inference.py` in the repository root:

```python
from pathlib import Path

import torch

from aereo_water.inference.predictor import SegFormerPredictor


CHECKPOINT_DIR = Path("artifacts/checkpoints/segformer_best")
SELECTED_MODEL = Path("evidence/registry/selected_model.json")
INPUT_IMAGE = Path("sample_satellite_image.png")
OUTPUT_DIR = Path("artifacts/local_inference")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

predictor = SegFormerPredictor(
    CHECKPOINT_DIR,
    selected_model_path=SELECTED_MODEL,
    image_size=512,
    resize_policy="letterbox",
    device=device,
    log_path=OUTPUT_DIR / "inference.jsonl",
    model_version="segformer-v3.0.0",
    warmup_runs=1,
)

mask, probability, metadata = predictor.predict(
    INPUT_IMAGE,
    request_id="local-example-001",
)

mask_path = predictor.save_mask(
    mask,
    OUTPUT_DIR / "predicted_water_mask.png",
)

overlay_path = predictor.save_overlay(
    INPUT_IMAGE,
    mask,
    OUTPUT_DIR / "predicted_water_overlay.png",
)

print("Device:", device)
print("Mask:", mask_path)
print("Overlay:", overlay_path)
print("Metadata:", metadata)
```

Run:

```powershell
python .\run_v3_inference.py
```

Expected outputs:

```text
artifacts/local_inference/
├── predicted_water_mask.png
├── predicted_water_overlay.png
└── inference.jsonl
```

The mask preserves the original image dimensions and contains:

```text
0   = non-water
255 = water
```

---

# Run the FastAPI service

## Configure the service

```powershell
New-Item -ItemType Directory -Force `
  ".\artifacts\logs" | Out-Null

$env:AEREO_CHECKPOINT = (
    Resolve-Path ".\artifacts\checkpoints\segformer_best"
).Path

$env:AEREO_SELECTED_MODEL = (
    Resolve-Path ".\evidence\registry\selected_model.json"
).Path

$env:AEREO_DEVICE = "cpu"
$env:AEREO_IMAGE_SIZE = "512"
$env:AEREO_RESIZE_POLICY = "letterbox"
$env:AEREO_LOG_PATH = (
    Join-Path $PWD "artifacts\logs\api_inference.jsonl"
)

$env:AEREO_MAX_UPLOAD_BYTES = "10485760"
$env:AEREO_MAX_IMAGE_PIXELS = "40000000"
$env:AEREO_MAX_CONCURRENCY = "1"
```

Use CUDA only when a compatible CUDA-enabled PyTorch environment is installed:

```powershell
$env:AEREO_DEVICE = "cuda"
```

## Start the API

```powershell
python -m uvicorn `
  aereo_water.api.app:app `
  --host 0.0.0.0 `
  --port 8000
```

Interactive documentation:

```text
http://localhost:8000/docs
```

## Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/health` | GET | Process liveness |
| `/ready` | GET | Checkpoint and predictor readiness |
| `/metadata` | GET | Model version, device, threshold, and metadata |
| `/segment` | POST | Image upload to original-resolution binary PNG |

## Test the API

Open a second PowerShell window:

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod http://localhost:8000/metadata
```

Send an image:

```powershell
curl.exe `
  -X POST `
  "http://localhost:8000/segment" `
  -F "image=@sample_satellite_image.png;type=image/png" `
  --output predicted_water_mask.png
```

The API tests cover valid segmentation, invalid content type, missing file, and oversized upload behavior.

---

## Docker Deployment

The Aereo Water Segmentation service is deployed as a FastAPI application containing the validated SegFormer V3 water-segmentation model.

The service supports two deployment methods:

1. Pull the complete, self-contained image from Docker Hub.
2. Build and run the image locally using Docker Compose.

### Validated production configuration

| Item | Value |
|---|---|
| Local Docker image | `aereo-water-segmentation:v3` |
| Docker Hub image | `YOUR_DOCKERHUB_USERNAME/aereo-water-segmentation:v3.0.0` |
| API port | `8000` |
| Production checkpoint | `artifacts/checkpoints/segformer_best` |
| Selected-model registry | `evidence/registry/selected_model.json` |
| Validation image | `evidence/inference/water_body_2496.jpg` |
| Selected threshold | `0.45` |
| Checkpoint SHA-256 | `dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4` |

The production Docker image contains:

- the FastAPI application;
- production Python dependencies;
- the validated SegFormer V3 checkpoint;
- the SegFormer processor configuration;
- the calibrated segmentation threshold;
- the selected-model registry;
- a writable inference-log directory.

---

## Deployment files

The canonical Docker deployment files are:

```text
deployment/Dockerfile
deployment/compose.yaml
.dockerignore
```

The repository checkpoint bundle is:

```text
artifacts/
└── checkpoints/
    └── segformer_best/
        ├── config.json
        ├── model.safetensors
        ├── preprocessor_config.json
        └── selected_threshold.json
```

The selected-model registry is:

```text
evidence/
└── registry/
    └── selected_model.json
```

---

# Option A — Run the published Docker Hub image

This is the recommended approach for reviewers because it does not require rebuilding the image or installing Python dependencies locally.

Replace `YOUR_DOCKERHUB_USERNAME` with the actual Docker Hub username.

## 1. Pull the versioned image

```powershell
docker pull YOUR_DOCKERHUB_USERNAME/aereo-water-segmentation:v3.0.0
```

Use the versioned `v3.0.0` tag for reproducibility.

The `latest` tag may also be available:

```powershell
docker pull YOUR_DOCKERHUB_USERNAME/aereo-water-segmentation:latest
```

However, `latest` may change in future releases.

## 2. Start the container

```powershell
docker run `
  -d `
  --name aereo-water-segmentation `
  -p 8000:8000 `
  -e AEREO_CHECKPOINT=/app/artifacts/checkpoints/segformer_best `
  -e AEREO_SELECTED_MODEL=/app/evidence/registry/selected_model.json `
  -e AEREO_LOG_PATH=/app/artifacts/logs/inference.jsonl `
  -e AEREO_DEVICE=cpu `
  -e AEREO_IMAGE_SIZE=512 `
  -e AEREO_RESIZE_POLICY=letterbox `
  -e AEREO_MAX_UPLOAD_BYTES=20971520 `
  -e AEREO_MAX_IMAGE_PIXELS=100000000 `
  -e AEREO_MAX_CONCURRENCY=1 `
  YOUR_DOCKERHUB_USERNAME/aereo-water-segmentation:v3.0.0
```

Docker returns a hexadecimal container ID when the container starts.

## 3. Inspect the container

```powershell
docker ps `
  --filter "name=aereo-water-segmentation"
```

Inspect startup logs:

```powershell
docker logs `
  aereo-water-segmentation `
  --tail 200
```

Do not continue until the container is running and the logs show that the FastAPI application has started.

## 4. Test the API

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod http://localhost:8000/metadata
```

Expected behavior:

- `/health` confirms API-process liveness;
- `/ready` confirms that the model, processor, threshold, and registry loaded successfully;
- `/metadata` returns the selected production-model information.

A successful `/health` response does not by itself prove that the model loaded. `/ready` must also succeed.

## 5. Stop the Docker Hub container

```powershell
docker rm -f aereo-water-segmentation
```

---

# Option B — Build and run from the repository

Run the following commands from the repository root.

```powershell
cd <path-to-aereo-water-segmentation>
```

For example:

```powershell
cd D:\aereotasksubmission_maanvibansal\Source\aereo-water-segmentation
```

---

## 1. Prerequisites

Confirm that:

- Docker Desktop is installed;
- Docker Desktop is running;
- Docker is using Linux containers;
- Docker Compose is available;
- port `8000` is free;
- the production checkpoint exists;
- the selected-model registry exists;
- the checkpoint bundle is complete.

Check Docker:

```powershell
docker --version
docker compose version
docker info
```

`docker info` must display both client and server information.

### Check required project files

```powershell
$requiredFiles = @(
  ".\deployment\Dockerfile",
  ".\deployment\compose.yaml",
  ".\artifacts\checkpoints\segformer_best\config.json",
  ".\artifacts\checkpoints\segformer_best\model.safetensors",
  ".\artifacts\checkpoints\segformer_best\preprocessor_config.json",
  ".\artifacts\checkpoints\segformer_best\selected_threshold.json",
  ".\evidence\registry\selected_model.json"
)

$missingFiles = $requiredFiles |
  Where-Object { -not (Test-Path $_) }

if ($missingFiles) {
  Write-Host "Missing required Docker deployment files:"
  $missingFiles
  throw "The production deployment bundle is incomplete."
}

Write-Host "All required Docker deployment files are present."
```

Create the writable inference-log directory:

```powershell
New-Item `
  -ItemType Directory `
  -Force `
  -Path ".\artifacts\logs" |
Out-Null
```

---

## 2. Docker Desktop troubleshooting

When Docker cannot connect to its daemon:

```powershell
docker desktop start
docker context ls
docker context use desktop-linux
docker info
```

When Docker Desktop is open but Docker remains unavailable, restart Docker Desktop from the application.

WSL may also be restarted with:

```powershell
wsl --shutdown
```

Then reopen Docker Desktop and run:

```powershell
docker info
```

Opening a fresh PowerShell window can also clear stale Docker-context or environment state.

Do not continue until `docker info` displays server information.

---

## 3. Check whether port 8000 is already occupied

```powershell
Get-NetTCPConnection `
  -LocalPort 8000 `
  -ErrorAction SilentlyContinue
```

Inspect Docker port mappings:

```powershell
docker ps `
  --format "table {{.Names}}\t{{.Ports}}"
```

When another Docker container is using port `8000`, stop it:

```powershell
docker rm -f <container-name>
```

Alternatively, use a different host port:

```powershell
docker run `
  -d `
  --name aereo-water-segmentation-test `
  -p 8001:8000 `
  aereo-water-segmentation:v3
```

The API would then be available at:

```text
http://localhost:8001
```

---

## 4. Verify the host checkpoint

The checkpoint file must exist:

```powershell
Test-Path `
  ".\artifacts\checkpoints\segformer_best\model.safetensors"
```

Expected:

```text
True
```

Read the selected-model registry:

```powershell
$registry = Get-Content `
  ".\evidence\registry\selected_model.json" `
  -Raw |
ConvertFrom-Json
```

Compare the registered checkpoint hash with the actual checkpoint hash:

```powershell
$expectedHash = $registry.checkpoint_sha256.ToLower()

$actualHash = (
  Get-FileHash `
    ".\artifacts\checkpoints\segformer_best\model.safetensors" `
    -Algorithm SHA256
).Hash.ToLower()

Write-Host "Expected checkpoint hash:" $expectedHash
Write-Host "Actual checkpoint hash:  " $actualHash

if ($expectedHash -ne $actualHash) {
  throw "Checkpoint integrity validation failed."
}

Write-Host "Checkpoint integrity validation passed."
```

Expected SHA-256:

```text
dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
```

Verify the selected threshold:

```powershell
if ($null -eq $registry.validation_threshold) {
  throw "selected_model.json does not contain validation_threshold."
}

Write-Host "Validation threshold:" $registry.validation_threshold
```

Expected:

```text
0.45
```

Do not disable checkpoint-integrity validation.

Do not change the registry hash merely to accept a different checkpoint.

---

## 5. Validate the Docker Compose configuration

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  config
```

This command:

- validates the YAML structure;
- resolves environment variables;
- resolves volume paths;
- displays the final Compose configuration.

It does not build or start the service.

### Required Compose paths

Because `compose.yaml` is inside the `deployment` directory, repository-root mounts must use `../`.

The required volume configuration is:

```yaml
volumes:
  - ../artifacts/checkpoints:/app/artifacts/checkpoints:ro
  - ../evidence/registry:/app/evidence/registry:ro
  - ../artifacts/logs:/app/artifacts/logs
```

Do not use:

```yaml
- ./artifacts/checkpoints:/app/artifacts/checkpoints:ro
```

That incorrect path resolves to:

```text
deployment/artifacts/checkpoints
```

instead of:

```text
artifacts/checkpoints
```

---

## 6. Remove old containers

Remove a previous Compose deployment:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  down `
  --remove-orphans
```

Remove any standalone test containers:

```powershell
docker rm -f aereo-water-segmentation 2>$null
docker rm -f aereo-standalone-test 2>$null
```

---

## 7. Determine whether the image must be rebuilt

Rebuild the Docker image after changing:

- `deployment/Dockerfile`;
- `.dockerignore`;
- `requirements/production.in`;
- `pyproject.toml`;
- application code under `src/`;
- the checkpoint embedded in the standalone image;
- the selected-model registry embedded in the image;
- system packages;
- Python dependencies.

A full image rebuild is generally unnecessary after changing only:

- mounted checkpoint files;
- mounted registry files;
- Compose environment variables;
- Compose volume paths;
- log files.

For mounted-file or Compose-only changes, recreate the container:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  down `
  --remove-orphans

docker compose `
  -f ".\deployment\compose.yaml" `
  up `
  -d `
  --force-recreate
```

---

## 8. Build the image from scratch

Build through Docker Compose:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  build `
  --pull `
  --no-cache
```

Equivalent direct Docker build:

```powershell
docker build `
  --pull `
  --no-cache `
  -f ".\deployment\Dockerfile" `
  -t aereo-water-segmentation:v3 `
  .
```

A clean build can take several minutes because production dependencies are downloaded and installed.

Inspect the resulting image:

```powershell
docker image inspect aereo-water-segmentation:v3
docker image ls aereo-water-segmentation:v3
```

---

## 9. Verify that the image is self-contained

The image published to Docker Hub must contain the production checkpoint and registry without relying on host volume mounts.

Verify the embedded checkpoint:

```powershell
docker run `
  --rm `
  --entrypoint sha256sum `
  aereo-water-segmentation:v3 `
  /app/artifacts/checkpoints/segformer_best/model.safetensors
```

Expected:

```text
dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
```

Verify the embedded registry:

```powershell
docker run `
  --rm `
  --entrypoint python `
  aereo-water-segmentation:v3 `
  -c "import json; d=json.load(open('/app/evidence/registry/selected_model.json')); print('hash:', d['checkpoint_sha256']); print('threshold:', d['validation_threshold'])"
```

Expected:

```text
hash: dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
threshold: 0.45
```

List the files embedded in the image:

```powershell
docker run `
  --rm `
  --entrypoint sh `
  aereo-water-segmentation:v3 `
  -c "ls -lah /app/artifacts/checkpoints/segformer_best && echo '--- registry ---' && ls -lah /app/evidence/registry/selected_model.json"
```

Expected checkpoint files:

```text
config.json
model.safetensors
preprocessor_config.json
selected_threshold.json
```

---

## 10. Start the Compose service

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  up `
  -d `
  --force-recreate
```

Inspect the service:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  ps
```

Inspect startup logs:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  logs `
  --tail=300
```

Do not continue until the service is running and the logs show successful application initialization.

---

## 11. Verify the mounted checkpoint inside the container

List the mounted checkpoint files:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  exec water-segmentation-v3 `
  ls -lah /app/artifacts/checkpoints/segformer_best
```

Verify the mounted model hash:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  exec water-segmentation-v3 `
  sha256sum /app/artifacts/checkpoints/segformer_best/model.safetensors
```

Expected:

```text
dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
```

Verify the mounted registry:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  exec water-segmentation-v3 `
  python -c "import json; d=json.load(open('/app/evidence/registry/selected_model.json')); print('hash:', d.get('checkpoint_sha256')); print('threshold:', d.get('validation_threshold'))"
```

Expected:

```text
hash: dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
threshold: 0.45
```

---

# API validation

## 1. Test health, readiness, and metadata

```powershell
Invoke-RestMethod http://localhost:8000/health
Invoke-RestMethod http://localhost:8000/ready
Invoke-RestMethod http://localhost:8000/metadata
```

Expected behavior:

### `/health`

Confirms that the API process is alive.

### `/ready`

Confirms that:

- the model checkpoint loaded;
- the processor loaded;
- the selected threshold loaded;
- the checkpoint passed integrity validation;
- the predictor is ready for inference.

### `/metadata`

Returns information about:

- the selected model;
- checkpoint identity;
- validation threshold;
- model version;
- registered performance values.

Do not run inference until `/ready` succeeds.

---

## 2. Test segmentation

Confirm that the repository validation image exists:

```powershell
Test-Path ".\evidence\inference\water_body_2496.jpg"
```

Expected:

```text
True
```

Submit the image:

```powershell
curl.exe `
  -X POST `
  "http://localhost:8000/segment" `
  -F "image=@evidence/inference/water_body_2496.jpg;type=image/jpeg" `
  --output docker_predicted_water_mask.png
```

Confirm that the response exists:

```powershell
Test-Path ".\docker_predicted_water_mask.png"
Get-Item ".\docker_predicted_water_mask.png"
```

Do not use:

```text
sample_satellite_image.png
```

That file is not included in the repository.

---

## 3. Verify that the response is a PNG

Use `Resolve-Path` so the absolute path is passed to .NET:

```powershell
$outputPath = (
  Resolve-Path ".\docker_predicted_water_mask.png"
).Path

$bytes = [System.IO.File]::ReadAllBytes($outputPath)

$signature = (
  $bytes[0..7] |
  ForEach-Object { $_.ToString("X2") }
) -join " "

Write-Host "Output file:" $outputPath
Write-Host "PNG signature:" $signature

if ($signature -ne "89 50 4E 47 0D 0A 1A 0A") {
  throw "The API response is not a valid PNG file."
}

Write-Host "Valid PNG prediction received."
```

Expected PNG signature:

```text
89 50 4E 47 0D 0A 1A 0A
```

Open the prediction:

```powershell
Start-Process ".\docker_predicted_water_mask.png"
```

Confirm that the request returned HTTP `200`:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  logs `
  --tail=100
```

Look for a successful request similar to:

```text
POST /segment HTTP/1.1" 200 OK
```

---

## 4. Test invalid input

Create an invalid text file:

```powershell
"not an image" |
  Set-Content `
  ".\invalid.txt" `
  -Encoding ASCII
```

Submit it:

```powershell
curl.exe `
  -i `
  -X POST `
  "http://localhost:8000/segment" `
  -F "image=@invalid.txt;type=text/plain"
```

Expected response:

```text
HTTP 415 Unsupported Media Type
```

The service must reject invalid input instead of returning a segmentation mask.

---

## 5. Restart and verify persistence

Restart the service:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  restart
```

Wait briefly:

```powershell
Start-Sleep -Seconds 5
```

Test readiness again:

```powershell
Invoke-RestMethod http://localhost:8000/ready
```

Run segmentation after restart:

```powershell
curl.exe `
  -X POST `
  "http://localhost:8000/segment" `
  -F "image=@evidence/inference/water_body_2496.jpg;type=image/jpeg" `
  --output docker_predicted_water_mask_after_restart.png
```

Verify the response:

```powershell
Test-Path ".\docker_predicted_water_mask_after_restart.png"

$outputPath = (
  Resolve-Path ".\docker_predicted_water_mask_after_restart.png"
).Path

$bytes = [System.IO.File]::ReadAllBytes($outputPath)

($bytes[0..7] |
  ForEach-Object { $_.ToString("X2") }) -join " "
```

Expected:

```text
89 50 4E 47 0D 0A 1A 0A
```

---

# Save Docker validation evidence

Create the validation-evidence directory:

```powershell
New-Item `
  -ItemType Directory `
  -Force `
  ".\docs\docker_validation" |
Out-Null
```

## Save container status

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  ps |
Set-Content `
  ".\docs\docker_validation\container_status.txt" `
  -Encoding UTF8
```

## Save container logs

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  logs |
Set-Content `
  ".\docs\docker_validation\container_logs.txt" `
  -Encoding UTF8
```

## Save API responses

```powershell
Invoke-RestMethod http://localhost:8000/health |
  ConvertTo-Json -Depth 10 |
  Set-Content `
    ".\docs\docker_validation\health_response.json" `
    -Encoding UTF8

Invoke-RestMethod http://localhost:8000/ready |
  ConvertTo-Json -Depth 10 |
  Set-Content `
    ".\docs\docker_validation\readiness_response.json" `
    -Encoding UTF8

Invoke-RestMethod http://localhost:8000/metadata |
  ConvertTo-Json -Depth 10 |
  Set-Content `
    ".\docs\docker_validation\metadata_response.json" `
    -Encoding UTF8
```

## Save checkpoint-integrity evidence

```powershell
$registry = Get-Content `
  ".\evidence\registry\selected_model.json" `
  -Raw |
ConvertFrom-Json

$integrityEvidence = [ordered]@{
  expected_sha256 = $registry.checkpoint_sha256
  actual_sha256 = (
    Get-FileHash `
      ".\artifacts\checkpoints\segformer_best\model.safetensors" `
      -Algorithm SHA256
  ).Hash.ToLower()
  validation_threshold = $registry.validation_threshold
  validated_at_utc = [DateTime]::UtcNow.ToString("o")
}

$integrityEvidence |
  ConvertTo-Json |
  Set-Content `
    ".\docs\docker_validation\checkpoint_integrity.json" `
    -Encoding UTF8
```

## Save prediction outputs

```powershell
Copy-Item `
  ".\docker_predicted_water_mask.png" `
  ".\docs\docker_validation\predicted_mask.png" `
  -Force

Copy-Item `
  ".\docker_predicted_water_mask_after_restart.png" `
  ".\docs\docker_validation\predicted_mask_after_restart.png" `
  -Force
```

Docker validation should only be marked complete after:

- the image builds successfully;
- the service starts successfully;
- `/health` succeeds;
- `/ready` succeeds;
- `/metadata` succeeds;
- valid-image inference succeeds;
- the returned result is verified as a PNG;
- invalid content is rejected;
- the service restarts successfully;
- inference succeeds after restart;
- the checkpoint hash matches the registry;
- validation evidence is preserved;
- the service is shut down cleanly.

---

# Publish the image to Docker Hub

Only publish the image after confirming that it contains the production checkpoint and registry.

## 1. Log in

```powershell
docker login
```

Complete authentication and wait for:

```text
Login Succeeded
```

Do not place Docker Hub passwords or access tokens inside scripts, the repository, or the README.

## 2. Define the Docker Hub repository

```powershell
$dockerHubUser = Read-Host "Enter Docker Hub username"
$repository = "$dockerHubUser/aereo-water-segmentation"

Write-Host "Docker Hub repository:" $repository
```

The Docker Hub repository should be named:

```text
aereo-water-segmentation
```

A public repository allows reviewers to pull the image without authenticating.

## 3. Tag the validated image

```powershell
docker tag `
  aereo-water-segmentation:v3 `
  "${repository}:v3.0.0"

docker tag `
  aereo-water-segmentation:v3 `
  "${repository}:latest"
```

Verify the tags:

```powershell
docker image ls $repository
```

## 4. Push the images

```powershell
docker push "${repository}:v3.0.0"
docker push "${repository}:latest"
```

No API container needs to be running during a Docker Hub push.

Host ports `8000` and `8001` are not involved in the push.

## 5. Verify the published image without using ports

Pull the versioned image:

```powershell
docker pull "${repository}:v3.0.0"
```

Verify its checkpoint:

```powershell
docker run `
  --rm `
  --entrypoint sha256sum `
  "${repository}:v3.0.0" `
  /app/artifacts/checkpoints/segformer_best/model.safetensors
```

Expected:

```text
dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
```

Verify its registry:

```powershell
docker run `
  --rm `
  --entrypoint python `
  "${repository}:v3.0.0" `
  -c "import json; d=json.load(open('/app/evidence/registry/selected_model.json')); print('hash:', d['checkpoint_sha256']); print('threshold:', d['validation_threshold'])"
```

Expected:

```text
hash: dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
threshold: 0.45
```

## 6. Record the Docker Hub digest

```powershell
$digest = docker image inspect `
  "${repository}:v3.0.0" `
  --format '{{index .RepoDigests 0}}'

$digest
```

Save the Docker Hub repository and digest:

```powershell
New-Item `
  -ItemType Directory `
  -Force `
  ".\docs\docker_validation" |
Out-Null

$repository |
  Set-Content `
    ".\docs\docker_validation\dockerhub_repository.txt" `
    -Encoding UTF8

$digest |
  Set-Content `
    ".\docs\docker_validation\dockerhub_digest.txt" `
    -Encoding UTF8
```

---

# Troubleshooting

## Docker cannot connect to the daemon

Run:

```powershell
docker desktop start
docker context ls
docker context use desktop-linux
docker info
```

When necessary:

```powershell
wsl --shutdown
```

Restart Docker Desktop and open a new PowerShell window.

---

## Port 8000 is already allocated

Inspect active containers:

```powershell
docker ps `
  --format "table {{.Names}}\t{{.Ports}}"
```

Stop the conflicting container:

```powershell
docker rm -f <container-name>
```

Or use port `8001`:

```powershell
docker run `
  -d `
  --name aereo-water-segmentation-test `
  -p 8001:8000 `
  aereo-water-segmentation:v3
```

Then use:

```text
http://localhost:8001
```

---

## `/health` works but `/ready` returns 503

This means that the web server is alive but model initialization failed.

Inspect logs:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  logs `
  --tail=300
```

Check:

- checkpoint existence;
- checkpoint SHA-256;
- `preprocessor_config.json`;
- `selected_threshold.json`;
- selected-model registry path;
- `validation_threshold`;
- Compose volume paths.

---

## Missing `preprocessor_config.json`

Verify:

```powershell
Test-Path `
  ".\artifacts\checkpoints\segformer_best\preprocessor_config.json"
```

The complete deployment bundle must contain:

```text
config.json
model.safetensors
preprocessor_config.json
selected_threshold.json
```

Copy the complete `segformer_best` directory.

Do not copy only `model.safetensors`.

The validated Kaggle deployment archive was exported as:

```text
aereo-water-segformer-v3-deployment.zip
```

---

## Missing validation threshold

```powershell
$registry = Get-Content `
  ".\evidence\registry\selected_model.json" `
  -Raw |
ConvertFrom-Json

$registry.validation_threshold
```

Expected:

```text
0.45
```

Confirm the runtime registry environment path:

```yaml
AEREO_SELECTED_MODEL: /app/evidence/registry/selected_model.json
```

Confirm the registry mount:

```yaml
- ../evidence/registry:/app/evidence/registry:ro
```

---

## Checkpoint-integrity validation failed

Compare the registered and actual hashes:

```powershell
$registry = Get-Content `
  ".\evidence\registry\selected_model.json" `
  -Raw |
ConvertFrom-Json

$registry.checkpoint_sha256

(
  Get-FileHash `
    ".\artifacts\checkpoints\segformer_best\model.safetensors" `
    -Algorithm SHA256
).Hash.ToLower()
```

Expected:

```text
dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4
```

Do not disable integrity validation.

Do not modify the registered hash to accept an unrelated checkpoint.

Restore the validated checkpoint bundle and recreate the service:

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  down `
  --remove-orphans

docker compose `
  -f ".\deployment\compose.yaml" `
  up `
  -d `
  --force-recreate
```

---

## Docker cannot copy the checkpoint during build

Example:

```text
COPY artifacts/checkpoints/segformer_best ... not found
CopyIgnoredFile
```

This means `.dockerignore` excluded the checkpoint from the Docker build context.

The Dockerfile must contain:

```dockerfile
COPY artifacts/checkpoints/segformer_best \
     /app/artifacts/checkpoints/segformer_best

COPY evidence/registry/selected_model.json \
     /app/evidence/registry/selected_model.json
```

A working `.dockerignore` is:

```dockerignore
.git
**/__pycache__/
**/*.pyc
**/.ipynb_checkpoints/
outputs/
**/.pytest_cache/
```

After correcting `.dockerignore`, rebuild:

```powershell
docker build `
  --pull `
  --no-cache `
  -f ".\deployment\Dockerfile" `
  -t aereo-water-segmentation:v3 `
  .
```

The build context should include the approximately 15 MB checkpoint instead of containing only a few kilobytes.

---

## The image contains no checkpoint

Verify the image without Compose mounts:

```powershell
docker run `
  --rm `
  --entrypoint sha256sum `
  aereo-water-segmentation:v3 `
  /app/artifacts/checkpoints/segformer_best/model.safetensors
```

When the file is missing, the image is not self-contained.

Correct the Dockerfile and `.dockerignore`, rebuild, and verify the checkpoint before pushing to Docker Hub.

---

## `curl: (26) Failed to open/read local data`

This is a local input-file error, not an API or Docker failure.

Do not use:

```text
sample_satellite_image.png
```

Use:

```text
evidence/inference/water_body_2496.jpg
```

Correct command:

```powershell
curl.exe `
  -X POST `
  "http://localhost:8000/segment" `
  -F "image=@evidence/inference/water_body_2496.jpg;type=image/jpeg" `
  --output docker_predicted_water_mask.png
```

---

## .NET resolves a relative path from the wrong directory

Use an absolute path generated by `Resolve-Path`:

```powershell
$file = (
  Resolve-Path ".\docker_predicted_water_mask.png"
).Path

$bytes = [System.IO.File]::ReadAllBytes($file)
```

Compare the PowerShell and .NET working directories:

```powershell
Get-Location
[System.IO.Directory]::GetCurrentDirectory()
```

Synchronize them when needed:

```powershell
[System.IO.Directory]::SetCurrentDirectory(
  (Get-Location).Path
)
```

---

## Invalid input unexpectedly returns a mask

The service should reject unsupported media types.

Expected response:

```text
HTTP 415 Unsupported Media Type
```

Inspect the request content type and API logs before marking validation complete.

---

# Stop and clean up

## Stop the Compose deployment

```powershell
docker compose `
  -f ".\deployment\compose.yaml" `
  down `
  --remove-orphans
```

## Remove a standalone container

```powershell
docker rm -f aereo-water-segmentation
```

## Remove temporary validation files

```powershell
Remove-Item `
  ".\invalid.txt", `
  ".\docker_predicted_water_mask.png", `
  ".\docker_predicted_water_mask_after_restart.png" `
  -Force `
  -ErrorAction SilentlyContinue
```

## Optional image cleanup

```powershell
docker image rm aereo-water-segmentation:v3
```

Only remove the image when intentionally forcing the next run to rebuild or repull it.

---

# Docker validation checklist

Before marking Docker deployment complete, confirm:

- [ ] Docker Desktop starts successfully.
- [ ] `docker info` shows client and server information.
- [ ] The Compose configuration resolves successfully.
- [ ] The checkpoint bundle is complete.
- [ ] `preprocessor_config.json` is present.
- [ ] `selected_threshold.json` is present.
- [ ] The host checkpoint SHA-256 matches the registry.
- [ ] The image builds successfully.
- [ ] The image contains the embedded checkpoint.
- [ ] The embedded checkpoint hash is correct.
- [ ] The embedded registry contains threshold `0.45`.
- [ ] The container starts successfully.
- [ ] `/health` succeeds.
- [ ] `/ready` succeeds.
- [ ] `/metadata` succeeds.
- [ ] Valid-image inference returns HTTP `200`.
- [ ] The prediction has a valid PNG signature.
- [ ] Invalid input returns HTTP `415`.
- [ ] Readiness succeeds after restart.
- [ ] Inference succeeds after restart.
- [ ] Logs and API responses are preserved.
- [ ] The Docker Hub `v3.0.0` image is pushed.
- [ ] The Docker Hub image is verified after pulling.
- [ ] The immutable Docker Hub digest is recorded.
- [ ] The service is shut down cleanly.

---

# Reproduce the experiments

The supported full reproduction environment is Kaggle because the complete profile requires GPU compute and substantial temporary storage.

## Notebook lineage

The complete assignment is preserved across the original comparative
experiments and the later governed production extension.

### Original comparative study — Experiments A–D

The historical notebooks contain:

- Experiment A: zero-shot SAM under controlled prompt modes;
- Experiment B: water-specific fine-tuned SAM;
- Experiment C: the historical automatic SegFormer baseline;
- Experiment D: the automatic SegFormer-to-SAM hybrid.

### Governed production extension — Experiment E

The V3 notebooks contain:

- Optuna hyperparameter search;
- same-code historical baseline confirmation;
- top-candidate confirmation;
- three-seed stability analysis;
- resumable final training;
- validation-only threshold calibration;
- frozen held-out evaluation;
- paired statistics and failure slices;
- production inference and FastAPI validation;
- model registry and release export.

| Notebook | Scope | Purpose |
|---|---|---|
| `notebooks/aereo-task-original.ipynb` | Original assignment | Initial comparative implementation and experiment development |
| `notebooks/aereo-task-maanvi-bansal (1).ipynb` | Experiments A–D | Complete or executed comparative study; verify exact role from notebook contents |
| `notebooks/aereo_sam_production_pipeline.ipynb` | Experiments A–B | Zero-shot SAM and water-specific SAM fine-tuning |
| `notebooks/01_complete_segformer_production_pipeline_source.ipynb` | Experiment E, smoke/source | Low-cost validation of the governed V3 pipeline |
| `notebooks/01_complete_segformer_production_pipeline_full_run.ipynb` | Experiment E, full | Recreate governed optimization, evaluation, and production evidence |
| `notebooks/Aereo_Production_SegFormer_V3_EXECUTED.ipynb` | Experiment E, executed | Review measured V3 outputs without rerunning |

The V3 pipeline consumes the preserved historical A–D result tables. It
does not replace the original SAM, SegFormer, and hybrid experiments.
## Smoke profile

The smoke profile checks the complete stage graph using reduced data, trials, epochs, and evaluation rows. It must not be used for final scientific claims.

```python
RUN_PROFILE = "smoke"
RESET_OUTPUT_ROOT = False
```

## Full profile

The full profile performs:

- 12 completed Optuna trials, with up to 20 attempts;
- four HPO epochs on a fixed 1,000-image training subset;
- evaluation on all 429 validation images;
- historical same-code baseline and top-three confirmation;
- three-seed stability using 42, 2026, and 3407;
- up to 15 final-training epochs;
- validation-only threshold calibration;
- frozen 421-image held-out evaluation;
- full 2,841-image inference;
- paired statistics and performance slices;
- FastAPI validation and release export.

```python
RUN_PROFILE = "full"
RESET_OUTPUT_ROOT = False
```

## Resume

The stage controller loads valid completion artifacts and runs only missing, failed, or interrupted stages. Final training preserves model, optimizer, scheduler, scaler, epoch, best score, early-stopping state, history, and random states.

For detailed instructions, use:

- [Execution runbook](docs/SEGFORMER_V3_RUNBOOK.md)
- [System architecture](docs/architecture.md)

---

# Evidence, tracking, and registries

## MLflow

MLflow is the authoritative experiment system of record. The full run tracks:

- Optuna trials;
- historical and candidate confirmation;
- seed stability;
- final training;
- calibration;
- evaluation;
- checkpoints;
- parameters;
- metrics;
- figures;
- per-image result tables.

The complete external run preserves:

```text
tracking/mlflow.db
tracking/mlartifacts/
```

## W&B

W&B is retained as an optional offline visualization mirror and is not the authoritative tracker.

## Evidence directories

```text
evidence/acceptance/
evidence/api/
evidence/calibration/
evidence/environment/
evidence/evaluation/
evidence/inference/
evidence/registry/
evidence/results/
evidence/run_state/
evidence/slices/
evidence/statistics/
```

## Figures

```text
reports/figures/dataset_eda.png
reports/figures/preprocessing_and_augmentation.png
reports/figures/tiling_reconstruction.png
reports/figures/historical_model_comparison.png
reports/figures/threshold_calibration.png
reports/figures/paired_iou_comparison.png
reports/figures/performance_slices.png
reports/figures/qualitative_success_failure_analysis.png
reports/figures/production_inference_evidence.png
```

---

# Tests and continuous integration

Canonical workflow:

```text
.github/workflows/ci.yml
```

CI performs:

1. Python 3.11 setup;
2. CPU PyTorch installation;
3. dependency installation;
4. editable package installation;
5. critical import checks;
6. source and script compilation;
7. repository tests.

Local validation:

```powershell
python -m compileall -q src scripts
python -m pytest -q
python -m pip check
```

Expected executed evidence:

```text
43 passed
```

---

# Limitations and future work

## Known limitations

1. **RGB-only sensing:** NIR, SWIR, SAR, and temporal evidence are unavailable.
2. **Geographic generalization:** Cross-country, cross-sensor, and cross-season performance is not established.
3. **No empty test masks:** Empty-scene robustness cannot be measured from the current held-out split.
4. **Reference-label quality:** Some labels may contain automated or index-assisted uncertainty.
5. **Variable source dimensions:** Global metrics and raw-pixel boundary distances are size-dependent.
6. **Historical latency scope:** Historical stage-level latency is not directly comparable with V3 end-to-end latency.
7. **Docker runtime evidence:** Packaging is provided, but validation remains pending until the external lifecycle is executed.
8. **Serving hardening:** Authentication, rate limiting, distributed serving, and sustained load tests are not included.
9. **Optimized export:** ONNX, TensorRT, FP16, and INT8 deployment benchmarks are future work.

## Prioritized next work

- add NIR and SWIR bands;
- evaluate SAR-optical fusion for cloud robustness;
- create geographic and seasonal OOD splits;
- add manually verified dry-scene negative controls;
- use boundary-aware and topology-aware objectives;
- assess uncertainty and probability calibration;
- benchmark optimized exports;
- complete sustained load and concurrency testing.

---

# Documentation

- [Execution and repository runbook](docs/SEGFORMER_V3_RUNBOOK.md)
- [System architecture](docs/architecture.md)
- [Repository layout](docs/repository_layout.md)
- [Dataset card](docs/dataset_card.md)
- [Model card](docs/model_card.md)
- [Release notes](docs/release_notes_v3.md)
- [Evidence directory guide](evidence/README.md)
- [Legacy directory guide](legacy/README.md)
- [Reports directory guide](reports/README.md)

---

# Project provenance

Developed for the Aereo Data Scientist Intern water-body segmentation assignment.

The repository preserves maintained source code, historical experiments, measured evidence, model and data registries, an executed notebook, release artifacts, and CI history so that reported claims can be traced back to reproducible code and generated artifacts.
