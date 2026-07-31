# Presentation outline

## Slide 1 — Problem and objective
Water-body segmentation from satellite RGB imagery; complete training-to-deployment system rather than a visual SAM demo.

## Slide 2 — Dataset and quality controls
2,841 expected pairs, dimensions, water-coverage distribution, corrupt/unpaired files, duplicates, and mask-source limitations.

## Slide 3 — Leakage-safe pipeline
Manifest, SHA-256 grouping, stratified split, augmentation, optional parent-first tiling, and artifact lineage.

## Slide 4 — Why SAM and how it was adapted
SAM ViT-B architecture, frozen encoder strategy, prompt encoder, mask decoder, no-prompt semantic adaptation, and optional last-block fine-tuning.

## Slide 5 — Prompt experiments
None, one point, positive/negative points, jittered box, and box plus points. Clearly separate deployable and oracle-assisted results.

## Slide 6 — Training system
Combined loss, AMP, accumulation, optimizer, scheduler, early stopping, resume, W&B/local tracking, and sweep search space.

## Slide 7 — Evaluation protocol
Validation-only threshold selection; full test set; macro and global region metrics; boundary, surface, calibration, latency, and confidence intervals.

## Slide 8 — Results
Training curves, threshold sweep, prompt comparison table, and confidence intervals. Populate only after the complete Kaggle run.

## Slide 9 — Failure analysis
Worst automatic predictions, area over/under-segmentation, fragmented water, shadows, vegetation, turbid water, and label noise.

## Slide 10 — Deployment architecture
CLI and GeoTIFF path, FastAPI endpoint, cached image embeddings for repeated prompts, Docker image, logs, health checks, and CI.

## Slide 11 — Limitations
RGB-only data, oracle prompts, random rather than geographic holdout, source-mask uncertainty, and untested deployment hardware.

## Slide 12 — Next steps
Multispectral NIR/SWIR, SegFormer/U-Net baseline, geographic split, active prompting, optimization/quantization, and drift monitoring.
