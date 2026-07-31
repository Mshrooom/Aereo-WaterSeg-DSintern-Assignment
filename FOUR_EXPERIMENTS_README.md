# Four-experiment full-dataset workflow

Start with `notebooks/aereo_water_four_experiments_full_2841.ipynb` on Kaggle.

The notebook produces predictions and per-image metrics for every discovered image under:

1. Zero-shot SAM with oracle prompt types.
2. Fine-tuned SAM with the same prompt types.
3. Automatic SegFormer semantic segmentation.
4. Automatic SegFormer-generated prompts followed by fine-tuned SAM refinement.

Training uses only the training split. Validation selects checkpoints and thresholds. The test split is the final comparison. Predictions are still exported for train, validation, and test so the requested all-image registry contains 2,841 images.

Primary result files:

- `experiment_A_zero_shot_sam_all_2841.csv`
- `experiment_B_finetuned_sam_all_2841.csv`
- `experiment_C_segformer_all_2841.csv`
- `experiment_D_auto_sam_all_2841.csv`
- `all_experiments_all_images.csv`
- `summary_test_only.csv`

Automatic deployment:

```bash
waterseg-hybrid-infer \
  --sam-checkpoint artifacts/checkpoints/best.pt \
  --segformer-checkpoint artifacts/checkpoints/segformer_best \
  --image input.tif \
  --output water_mask.tif
```

The automatic API can be built with `Dockerfile.hybrid` and serves `POST /segment`, `GET /health`, and `GET /ready`.
