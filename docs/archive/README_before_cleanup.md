# Water-Body Segmentation in Satellite Imagery

An end-to-end machine-learning system for segmenting water bodies in RGB
satellite imagery. The project compares zero-shot SAM, task-specific SAM
fine-tuning, automatic SegFormer segmentation, and an automatic
SegFormer-to-SAM refinement pipeline.

The complete workflow covers:

- scalable image-mask ingestion;
- deterministic train/validation/test splitting;
- preprocessing and augmentation;
- full-dataset model training;
- prompt-strategy benchmarking;
- automatic semantic segmentation;
- experiment tracking and model selection;
- per-image and aggregate evaluation;
- inference packaging;
- FastAPI serving;
- Docker deployment;
- unit testing and continuous integration.

---

## Assignment objective

The objective was to build a complete production-oriented machine-learning
solution for water-body segmentation using the **Satellite Images of Water
Bodies** dataset.

The required system includes:

1. Data ingestion and preprocessing
2. Model training and experiment tracking
3. Comparative evaluation
4. Optimized inference
5. API and containerized deployment
6. Reproducible code and documentation

---

## Dataset

The project uses the **Satellite Images of Water Bodies** dataset containing:

- **2,841 RGB satellite images**
- corresponding binary water masks
- variable image dimensions
- Sentinel-2-derived imagery
- water and non-water pixel labels

The deterministic parent-image split is:

| Split | Images |
|---|---:|
| Training | 1,991 |
| Validation | 429 |
| Test | 421 |
| **Total** | **2,841** |

Images are split before any optional tiling operation to prevent data leakage
between tiles originating from the same parent image.

---

# Experiments

## Experiment A — Zero-shot SAM baseline

A pretrained Segment Anything Model is evaluated without water-specific
training.

The experiment measures how well a general-purpose promptable segmentation
foundation model transfers directly to remote-sensing imagery.

Prompt strategies:

- one positive point;
- multiple positive and negative points;
- bounding box;
- bounding box plus points.

The prompts in this controlled experiment are generated from ground-truth
masks. Therefore, Experiment A is an **oracle-prompt benchmark**, not an
automatic deployment pipeline.

All four prompt types are evaluated across all 2,841 images.

Expected result rows:

```text
2,841 images × 4 prompt types = 11,364 rows
```

---

## Experiment B — Fine-tuned SAM

SAM is fine-tuned on the water-body training split to reduce the
remote-sensing domain gap.

Training strategy:

* pretrained SAM ViT-B backbone;
* frozen image encoder and prompt encoder by default;
* trainable mask decoder;
* mixed-prompt curriculum;
* BCE, Dice, focal, and IoU-head losses;
* mixed-precision training;
* gradient accumulation;
* validation-based checkpoint selection;
* validation-based threshold selection;
* resumable checkpoints.

The same four prompt strategies used in Experiment A are evaluated after
fine-tuning.

This experiment answers:

> How much does task-specific adaptation improve SAM, and which prompt
> strategy performs best?

---

## Experiment C — Automatic SegFormer baseline

SegFormer is fine-tuned as a direct semantic segmentation model.

Unlike SAM, SegFormer does not require:

* user clicks;
* bounding boxes;
* ground-truth-derived prompts;
* a separate prompt-generation stage.

The deployment flow is:

```text
Satellite image
      ↓
SegFormer
      ↓
Water probability map
      ↓
Validation-selected threshold
      ↓
Binary water mask
```

This is the strongest fully automatic experiment and the selected production
model.

---

## Experiment D — Automatic SegFormer–SAM pipeline

Experiment D tests whether SAM can improve an automatically generated
SegFormer mask.

Pipeline:

```text
Satellite image
      ↓
SegFormer coarse probability map
      ↓
Connected-component analysis
      ↓
Automatic box and point prompts
      ↓
Fine-tuned SAM refinement
      ↓
Probability fusion
      ↓
Final water mask
```

No ground-truth prompt is used during inference.

This experiment tests whether the additional SAM refinement stage justifies
its increased computational complexity and latency.

---

# Test results

The primary model-selection metric is **mean per-image IoU on the held-out
test split**.

| Experiment          | Best configuration |   Mean IoU |       Dice | Boundary F1 | Recorded latency |
| ------------------- | ------------------ | ---------: | ---------: | ----------: | ---------------: |
| A — Zero-shot SAM   | Best oracle prompt |     0.5667 |     0.6890 |      0.3544 |         235.1 ms |
| B — Fine-tuned SAM  | Box + points       |     0.6043 |     0.7348 |      0.4021 |         237.1 ms |
| **C — SegFormer**   | **Automatic**      | **0.7190** | **0.8191** |  **0.5985** |       **8.3 ms** |
| D — SegFormer + SAM | Automatic hybrid   |     0.6093 |     0.7352 |      0.4131 |        >242.8 ms |

## Main conclusion

Fine-tuning improved SAM over its zero-shot baseline across prompt types.
However, the fully automatic SegFormer model achieved:

* the highest test IoU;
* the highest Dice score;
* the strongest boundary quality;
* the lowest inference latency;
* the simplest production architecture.

The SegFormer–SAM hybrid increased complexity and latency without
outperforming SegFormer.

Therefore:

> **SegFormer was selected as the production model.**

A more complex pipeline was not selected merely because it used more models.
The final choice was based on accuracy, boundary quality, latency, and
deployment simplicity.

---

## Output tables

The complete experiments generate:

| File                                      |   Rows |
| ----------------------------------------- | -----: |
| `experiment_A_zero_shot_sam_all_2841.csv` | 11,364 |
| `experiment_B_finetuned_sam_all_2841.csv` | 11,364 |
| `experiment_C_segformer_all_2841.csv`     |  2,841 |
| `experiment_D_auto_sam_all_2841.csv`      |  2,841 |
| Combined long-form results                | 28,410 |

Each per-image result contains:

* image ID;
* dataset split;
* experiment name;
* prompt mode;
* threshold;
* inference latency;
* true positives;
* false positives;
* false negatives;
* true negatives;
* IoU;
* Dice;
* precision;
* recall;
* specificity;
* pixel accuracy;
* balanced accuracy;
* Matthews correlation coefficient;
* Cohen's kappa;
* boundary F1;
* boundary IoU;
* HD95;
* average symmetric surface distance;
* actual water fraction;
* predicted water fraction.

---

# Metrics

## Primary metric

### Mean per-image IoU

Mean IoU is used as the primary model-selection metric because the images have
different spatial dimensions. A global pixel-weighted metric would give
greater influence to larger images.

## Supporting metrics

* Dice coefficient
* Precision
* Recall
* Specificity
* Pixel accuracy
* Balanced accuracy
* Matthews correlation coefficient
* Cohen's kappa
* Boundary precision
* Boundary recall
* Boundary F1
* Boundary IoU
* HD95
* Average symmetric surface distance
* Inference latency

Boundary distances are reported in image pixels.

---

# Data pipeline

The ingestion pipeline:

1. Recursively discovers images and masks
2. Matches every image with its corresponding mask
3. Validates file readability
4. Checks image-mask dimensions
5. Calculates image and mask hashes
6. Detects duplicate content
7. Measures water-pixel coverage
8. Generates deterministic train, validation, and test splits
9. Preserves parent-image grouping
10. Supports leakage-safe materialized tiling

The pipeline supports common formats including:

* PNG
* JPEG
* TIFF
* GeoTIFF

---

# Training pipeline

The training system includes:

* deterministic seeding;
* configurable augmentations;
* mixed-precision training;
* gradient accumulation;
* gradient clipping;
* AdamW optimization;
* warm-up and cosine learning-rate scheduling;
* early stopping;
* validation threshold sweeps;
* best and last checkpoints;
* resumable optimizer and scheduler state;
* JSONL and CSV experiment logs;
* model registry metadata;
* local or optional W&B tracking.

---

# Repository layout

```text
.github/workflows/       Continuous integration and automated tests
configs/                 Reproducible experiment configurations
notebooks/               Full Kaggle orchestration notebook
requirements/            Training, testing, and serving dependencies
results/                 Aggregate metrics and per-image experiment tables
scripts/                 Training, inference, and API test utilities
src/waterseg/data/       Ingestion, validation, splitting, augmentation, tiling
src/waterseg/models/     SAM and SegFormer model wrappers
src/waterseg/            Losses, metrics, training, evaluation, inference, API
tests/                   Lightweight unit tests without model downloads
.dockerignore            Docker build exclusions
.gitignore               Git exclusions for weights and generated artifacts
Dockerfile               Production SegFormer API container
docker-compose.yml       Local production deployment configuration
pyproject.toml           Python package and command-line configuration
README.md                Project overview and reproduction instructions
README_SEGFORMER_DEPLOYMENT.md
                         Detailed deployment documentation
```

---

# Kaggle reproduction

## Requirements

* Kaggle GPU notebook
* Internet enabled for model download
* Satellite Images of Water Bodies dataset attached
* Repository source attached or uploaded

Run the notebook:

```text
notebooks/aereo_water_four_experiments_full_2841.ipynb
```

The notebook performs:

1. Dataset discovery
2. Manifest generation
3. Data validation
4. Split creation
5. Experiment A evaluation
6. Experiment B training and evaluation
7. Experiment C training and evaluation
8. Experiment D automatic refinement
9. Combined result generation
10. Model registry export
11. Artifact packaging

Training uses only the training split.

Validation is used for:

* checkpoint selection;
* early stopping;
* segmentation-threshold selection;
* hybrid-parameter selection.

The test split is used only for final reporting.

---

# Local installation

Create a virtual environment:

```powershell
py -3.11 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the package:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

For lightweight tests:

```powershell
python -m pip install pytest pandas opencv-python-headless
```

Run tests:

```powershell
python -m pytest -q
```

Compile the source:

```powershell
python -m compileall src
```

---

# Production deployment

The selected production service uses SegFormer only.

## Model checkpoint

Download the SegFormer checkpoint from the GitHub Release and extract it to:

```text
artifacts/checkpoints/segformer_best/
```

Expected checkpoint contents:

```text
metadata.pt
config.json
model.safetensors
```

The weight file may alternatively be named:

```text
pytorch_model.bin
```

Model artifacts are excluded from normal Git history.

---

## Docker deployment

Make sure Docker Desktop is running.

Build the image:

```powershell
docker compose build
```

Start the service:

```powershell
docker compose up -d
```

Check container status:

```powershell
docker compose ps
```

View logs:

```powershell
docker compose logs --tail=100
```

Stop the deployment:

```powershell
docker compose down
```

---

# API

The FastAPI service exposes:

| Endpoint    | Method | Purpose                         |
| ----------- | ------ | ------------------------------- |
| `/health`   | GET    | Process health                  |
| `/ready`    | GET    | Model readiness                 |
| `/metadata` | GET    | Model and threshold information |
| `/segment`  | POST   | Water-mask inference            |

## Health check

```powershell
Invoke-RestMethod http://localhost:8000/health
```

## Readiness check

```powershell
Invoke-RestMethod http://localhost:8000/ready
```

## Interactive documentation

Open:

```text
http://localhost:8000/docs
```

## Segment an image

```powershell
curl.exe `
  -X POST `
  "http://localhost:8000/segment" `
  -F "image=@sample_satellite_image.png" `
  --output predicted_water_mask.png
```

The returned PNG uses:

* black for non-water;
* white for predicted water.

---

# Continuous integration

GitHub Actions runs:

* lightweight unit tests;
* source compilation;
* dependency-independent metric tests;
* data-splitting tests;
* prompt-generation tests;
* tiling tests.

Local command:

```powershell
python -m pytest -q
```

Expected result:

```text
10 passed
```

---

# Results and artifacts

GitHub contains:

* source code;
* configurations;
* training notebook;
* per-image result CSVs;
* summary tables;
* threshold sweeps;
* experiment metadata;
* tests;
* Docker deployment code.

Large files are distributed separately through GitHub Releases:

* trained SegFormer checkpoint;
* checkpoint checksum;
* optional deployment artifact archive.

Full prediction masks and probability maps are not committed to Git history.

---

# Limitations

* The dataset contains RGB imagery only; Sentinel-2 NIR and SWIR bands are not
  available to the current models.
* Some reference masks were generated through water-index-based processing and
  should not be treated as perfect human annotations.
* Experiments A and B use oracle prompts for controlled SAM comparison.
* Boundary-distance metrics are measured in image pixels rather than metres.
* The dataset contains variable image dimensions.
* The hybrid latency recorded during evaluation excludes part of the previously
  materialized coarse-prediction stage.
* Performance should be revalidated before applying the model to another
  geography, season, sensor, or spatial resolution.

---

# Future improvements

Potential extensions include:

* leakage-safe patch-based SegFormer training;
* SegFormer-B2 or larger backbones;
* NIR and SWIR Sentinel-2 channels;
* normalized or geospatial boundary-distance metrics;
* test-time augmentation;
* cloud and haze augmentation;
* validation-selected morphological post-processing;
* model quantization;
* ONNX or TensorRT export;
* GPU-enabled container deployment;
* confidence calibration;
* geographic out-of-distribution evaluation.

---

# Author

 ## Maanvi Bansal :)

Data Science Intern Assignment — Water-Body Segmentation in Satellite Imagery

```

## After saving

Check the rendered Markdown locally:

```powershell
Get-Content .\README.md -TotalCount 40
