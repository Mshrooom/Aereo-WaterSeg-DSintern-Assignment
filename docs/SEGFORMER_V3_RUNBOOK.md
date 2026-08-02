# Aereo SegFormer V3 — Execution and Repository Runbook

## Purpose

This runbook applies the production-grade V3 patch, validates the repository,
runs the staged Kaggle workflow, resumes interrupted work, and promotes measured
artifacts into GitHub.

The original four-experiment notebook and historical SAM/SegFormer results remain
in the repository. V3 is the maintained training, evaluation, inference, API, and
deployment path.

---

## 1. Apply the repository patch on Windows

Commit or stash every current change first.

```powershell
$repo = "D:\aereotasksubmission_maanvibansal\Source\aereo-water-segmentation"
$patchZip = "$HOME\Downloads\Aereo_SegFormer_V3_Repo_Patch.zip"
$patchRoot = "D:\aereotasksubmission_maanvibansal\Deployment\aereo-v3-patch"

Set-Location $repo
git status --short
```

The status must be empty.

```powershell
Remove-Item $patchRoot -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive -Path $patchZip -DestinationPath $patchRoot -Force
Set-ExecutionPolicy -Scope Process Bypass

& "$patchRoot\apply_segformer_v3.ps1" -RepoPath $repo
```

The patch script:

1. copies `src/aereo_water/`, configurations, scripts, tests, notebooks, docs,
   V3 Docker files, and CI;
2. updates `pyproject.toml` package discovery for both `waterseg*` and
   `aereo_water*`;
3. installs lightweight CI dependencies;
4. installs the repository editable;
5. compiles source and scripts;
6. runs unit tests;
7. stages the changes without committing them.

Review and commit:

```powershell
git diff --cached --stat
git diff --cached -- pyproject.toml
python -m pytest -q
git commit -m "Add production-grade SegFormer V3 pipeline"
git push origin main
```

---

## 2. Kaggle inputs

Attach:

1. **Satellite Images of Water Bodies** — raw `Images/` and `Masks/`.
2. The repository is cloned from GitHub by the notebook.
3. Historical per-image CSVs should already exist in the repository under
   `evidence/results/full/`. The notebook also searches attached Kaggle inputs.
4. For resumption, optionally attach a prior
   `aereo-water-v3-resume.zip` extraction.

Enable:

- GPU accelerator;
- Internet, for the initial Hugging Face model download and package install;
- enough writable disk for HPO, MLflow, W&B offline runs, checkpoints,
  predictions, and exported bundles.

---

## 3. Notebook choices

### Source notebook

`Aereo_Production_SegFormer_V3_Source.ipynb`

Defaults to:

```python
RUN_PROFILE = "smoke"
RESET_OUTPUT_ROOT = False
```

Use it for repository review and a short end-to-end validation.

### Full-run notebook

`Aereo_Production_SegFormer_V3_Full_Run.ipynb`

Defaults to:

```python
RUN_PROFILE = "full"
RESET_OUTPUT_ROOT = False
```

It runs:

- 12 completed Optuna trials, with up to 20 attempts;
- four HPO epochs on a fixed 1,000-image training subset;
- all 429 validation images;
- same-code baseline plus the top three candidates;
- three-seed stability analysis;
- up to 15 epochs of resumable final training;
- validation-only threshold calibration;
- frozen test evaluation;
- full 2,841-image inference;
- statistics, slices, production inference, API tests, and exports.

Run the source notebook once before the full profile.

---

## 4. Automatic staged resumption

Use the full notebook with `RUN_PROFILE = "full"` and run from the beginning.
The notebook traverses the dependency graph automatically:

```text
data → hpo → confirmation → stability → final_train → calibrate
     → evaluate → inference → api_test → export
```

Completed stages are loaded; interrupted, failed, or missing stages resume or
rerun. Final training resumes from `last_state.pt`.

After a long Kaggle session, create a notebook version and preserve the output
root. Profile-specific output roots prevent smoke evidence from being reused as
final evidence:

```text
/kaggle/working/aereo-water-v3-smoke
/kaggle/working/aereo-water-v3-full
```

At the export stage, the notebook produces a resume ZIP containing Optuna,
MLflow, W&B, final training state, calibration, and registries.

To restore a prior stage:

1. attach the extracted resume bundle as a Kaggle input;
2. set `RESUME_ROOT_INPUT` to the attached root;
3. leave `RESET_OUTPUT_ROOT = False` for normal resumption;
4. run the notebook from the beginning.

The early cells reconstruct runtime paths from the portable registry and load
completed evidence. To deliberately invalidate prior work, set
`RESET_OUTPUT_ROOT = True`; this removes the complete profile-specific root so
no stale downstream artifact survives.

---

## 5. Scientific rules built into the notebook

- HPO, confirmation, seed stability, early stopping, and threshold selection
  never receive test rows.
- HPO optimizes original-resolution validation IoU at threshold 0.50.
- The final probability threshold is selected on validation only.
- A model-selection lock is written before the test dataframe is created.
- The historical SegFormer configuration is always rerun under V3 code.
- Historical SAM prompts are selected on validation and then reported on test.
- The selected production seed is declared before test evaluation.
- Empty-mask conventions and water-present metrics are both reported.
- Model-forward and end-to-end latency are kept separate.
- Acceptance criteria are loaded before final test results.
- Compliance is computed from evidence, not hard-coded booleans.

---

## 6. Critical output artifacts

### Data and leakage

```text
evidence/registry/validated_manifest.csv
evidence/registry/runtime_manifest.csv
evidence/registry/split_registry.csv
evidence/registry/data_registry.json
evidence/registry/near_duplicate_audit.csv
```

### Tiling

```text
tiling/tiling_manifest.csv
tiling/source_mosaic.png
tiling/reconstructed_mosaic.png
tiling/geotiff_tiling_manifest.csv
tiling/reconstructed_geotiff.tif
```

### Search and tracking

```text
hpo/optuna.db
hpo/trials.csv
hpo/best_trial.json
tracking/mlflow.db
tracking/mlartifacts/
tracking/wandb/
failure_ledger.csv
```

### Confirmation and final training

```text
confirmation/confirmation_results.csv
stability/seed_stability.csv
final_training/best_checkpoint/
final_training/last_checkpoint/
final_training/last_state.pt
final_training/history.csv
```

### Selection and evaluation

```text
evidence/run_state/selected_final_parameters.json
evidence/registry/model_selection_lock.json
evidence/calibration/validation_threshold_sweep.csv
evidence/calibration/selected_threshold.json
evidence/evaluation/segformer_v3_all_2841.csv
evidence/evaluation/segformer_v3_test_metrics.json
evidence/evaluation/historical_comparison.csv
evidence/statistics/paired_comparison.json
evidence/slices/
```

### Production evidence

```text
evidence/inference/predicted_water_mask.png
evidence/inference/predicted_water_overlay.png
evidence/inference/inference.jsonl
evidence/inference/latency_summary.json
evidence/registry/model_registry.csv
evidence/registry/selected_model.json
docs/model_card.md
docs/dataset_card.md
evidence/api/api_smoke_test.json
```

### Exported bundles

```text
/kaggle/working/aereo-water-v3-github-evidence.zip
/kaggle/working/aereo-water-v3-resume.zip
/kaggle/working/aereo-water-segformer-v3-deployment.zip
/kaggle/working/AEREO_V3_SHA256SUMS.txt
```

---

## 7. Promote measured evidence into GitHub

Download the GitHub evidence ZIP and extract it outside the repository.

Copy only measured CSV, JSON, Markdown, figures, logs, and sample predictions.
Do not commit:

- model weights;
- `mlflow.db`;
- `mlartifacts/`;
- W&B offline run directories;
- Optuna SQLite databases;
- bulk prediction masks.

Publish the deployment ZIP and SHA-256 file through a GitHub Release.

Keep the test metrics exactly as measured. Do not edit them manually.

---

## 8. Final Docker validation

After copying the deployment artifact to:

```text
artifacts/checkpoints/
├── segformer_best/
└── selected_model.json
```

build the V3 service:

```powershell
docker compose -f deployment/compose.yaml config
docker compose -f deployment/compose.yaml build --no-cache
docker compose -f deployment/compose.yaml up -d
docker compose -f deployment/compose.yaml ps
docker compose -f deployment/compose.yaml logs --tail=300
```

Validate:

```text
GET /health
GET /ready
GET /metadata
POST /segment
```

Test:

- one valid image;
- missing upload;
- invalid content type;
- corrupt image;
- oversized file;
- restart and post-restart inference.

Save Docker evidence under `docs/docker_validation/`.

---

## 9. Final submission gate

Do not claim completion until all items are true:

- full HPO contains the requested number of completed trials;
- same-code baseline and top-candidate confirmation exist;
- final training is resumable;
- threshold is validation-selected;
- the model-selection lock predates test evaluation;
- the full inference table has 2,841 unique image IDs;
- the test table has 421 unique image IDs;
- paired comparison and performance slices exist;
- checkpoint hash matches the registry;
- export–reload prediction parity passes;
- repository tests and CI pass;
- Docker valid/invalid/restart tests pass;
- report and presentation use the measured V3 outputs;
- limitations and failures remain visible.
