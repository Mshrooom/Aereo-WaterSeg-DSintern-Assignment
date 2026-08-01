# Aereo SegFormer V3 — Coverage and Quality Audit

## Status

The V3 source and full-run notebooks were rebuilt after the production-readiness
review. The notebooks contain 104 cells: 62 code and 42 markdown. Both pass
nbformat validation and every code cell passes Python AST parsing.

The repository patch compiles and its lightweight V3 test suite passes:

```text
33 passed
```

GPU training, HPO, full inference, online/offline tracker evidence, and Docker
runtime tests must still be executed in their target environments. No metrics
have been fabricated.

## Correctness and scientific controls

| Review requirement | V3 implementation |
|---|---|
| HPO metric matches final metric | Original-resolution mean per-image validation IoU at fixed threshold 0.50 |
| No test-based prompt selection | Historical prompt selected on validation, then reported on test |
| Test firewall | Test dataframe created only after immutable model-selection lock |
| Threshold leakage prevention | Final threshold calibrated on validation only |
| Same-code baseline | Documented historical optimizer/loss/batch configuration rerun under V3 engine |
| Seed stability | Selected configuration run with seeds 42, 2026, and 3407 |
| Statistical uncertainty | Paired bootstrap CI, Wilcoxon test, matched-pairs rank-biserial effect |
| Empty-mask transparency | All-image, water-present, empty-mask accuracy, and false-positive rate |
| Macro/global reporting | Mean per-image and pooled confusion-matrix metrics |
| Calibration | Test Brier score, ECE, predictive entropy, low-confidence fraction, reliability table/plot |
| Performance slices | Water coverage, image size, topology, and boundary complexity |
| Predeclared acceptance | Acceptance YAML loaded before test evaluation |

## Data engineering

| Requirement | V3 implementation |
|---|---|
| Paired discovery | Images and Masks selected from the same dataset root |
| Validation | Decode, dimensions, binary mask, water fraction, channels |
| Exact duplicate audit | SHA-256 crossing splits is rejected |
| Near-duplicate audit | Complete cross-split average-hash comparison with audit metadata |
| Portable registry | Relative paths, checksums, split counts, Git commit |
| Resume portability | Runtime paths rebuilt from the portable registry |
| Deterministic split | Historical split recovered when available; exact-count fallback |
| Leakage control | Split integrity asserted before model selection |
| Data limitations | RGB-only and missing geographic/acquisition metadata documented |

## Preprocessing and tiling

| Requirement | V3 implementation |
|---|---|
| Normalization | Hugging Face SegFormer processor with explicit tensor evidence |
| Augmentation | Synchronized spatial transforms; image-only brightness/contrast |
| Shape handling | Aspect-ratio-preserving letterbox |
| Padding correctness | Label 255 used for artificial padding and excluded from CE/Dice |
| Pixel tiling | Guaranteed multi-tile overlap demonstration and exact reconstruction |
| GeoTIFF tiling | CRS, transform, nodata, bands, bounds, and tile transforms preserved |
| Small raster edges | Boundless padding without resampling; tested |
| Split-before-tiling | Recorded in the data registry |

## Training and HPO

| Requirement | V3 implementation |
|---|---|
| PyTorch training | End-to-end SegFormer fine-tuning |
| Optimizer | AdamW with no weight decay on bias/norm parameters |
| Scheduler | Cosine decay with warm-up |
| AMP | Enabled on CUDA |
| Gradient controls | Accumulation, clipping, finite-loss checks |
| Diagnostics | Losses, IoU/Dice, gradient norm, throughput, GPU memory, duration |
| HPO | Optuna TPE, MedianPruner, completed-trial budget, failure classification |
| Study integrity | Fingerprint includes code, split, subsets, model, search space, training contract |
| Confirmation | Historical control plus top three unique non-baseline trials |
| Resumption | Model, optimizer, scheduler, scaler, DataLoader generator, RNG, history, early stopping |
| Checkpoints | Best, last, and epoch state |
| Failure evidence | Failure ledger with stage, root cause, resolution, rerun state |

## Tracking and registries

| Requirement | V3 implementation |
|---|---|
| MLflow | Authoritative SQLite backend and explicit artifact root |
| W&B | Optional offline visualization/artifact mirror |
| Trial evidence | Params, epoch metrics, runtime, memory, history, checkpoint |
| Final evaluation lineage | Linked MLflow evaluation run with frozen test evidence |
| Model registration | Portable registry always; MLflow model flavor attempted with controlled fallback |
| Model record | Version, model, hashes, Git commit, HPO fingerprint, params, epoch, threshold, validation/test metrics, latency, status |
| Deployment integrity | Predictor verifies registered checkpoint SHA-256 |
| Artifact portability | Runtime path, relative artifact path, MLflow URI, release URI field |

## Inference and deployment

| Requirement | V3 implementation |
|---|---|
| Production predictor | Checkpoint/processor/threshold loading and original-resolution mask |
| Output proof | Dimensions, binary values, water fraction, saved mask and overlay |
| Structured logs | Request ID, model/hash/device, dimensions, threshold, latency components, status/errors |
| Latency | Cold start, model-forward p50/p95, end-to-end p50/p95, throughput, memory |
| Export parity | Deployment ZIP extracted fresh and mask equality asserted |
| API | Health, readiness, metadata, segmentation |
| API hardening | Type, size, pixel limit, semaphore, request IDs, controlled errors |
| Container | Dedicated minimal serve requirements, non-root user, V3 API command, health check |
| CI | Compilation and tests on Python 3.11 |

## Notebook and deliverable quality

- Executive contract and acceptance criteria appear before model results.
- Every assignment requirement maps to a section, module, and evidence artifact.
- Smoke and full profiles use separate output roots.
- Completed stages load automatically; interrupted final training resumes.
- Source notebook defaults to smoke; full notebook is a separate explicit artifact.
- Compliance is computed from actual files, run counts, row counts, tests, and
  inference evidence rather than hard-coded completion.
- Historical work is preserved but does not control the maintained package.
- Report-ready figures, cards, cards/registries, checksum manifests, GitHub
  evidence, resume, and deployment bundles are exported.

## Runtime evidence still required

The following cannot be honestly marked complete until executed:

1. 12 completed full Optuna trials.
2. Same-engine confirmation and seed-stability results.
3. Final full training and selected threshold.
4. Frozen 421-image test metrics.
5. Full 2,841-image V3 inference.
6. Actual MLflow and W&B run directories.
7. Local no-cache Docker build and endpoint/restart validation.
8. Final report and presentation populated with measured V3 results.
