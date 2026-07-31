# Experimental Results

All four experiments were evaluated on the same deterministic split of the
2,841-image Satellite Images of Water Bodies dataset.

- Training images: 1,991
- Validation images: 429
- Test images: 421
- Total images: 2,841

## Test Performance

| Experiment | Configuration | Mean IoU | Dice | Boundary F1 | Mean latency |
|---|---|---:|---:|---:|---:|
| A: Zero-shot SAM | Best oracle prompt | 0.5667 | 0.6890 | 0.3544 | 235.1 ms |
| B: Fine-tuned SAM | Box + points | 0.6043 | 0.7348 | 0.4021 | 237.1 ms |
| C: SegFormer | Fully automatic | **0.7190** | **0.8191** | **0.5985** | **8.3 ms** |
| D: SegFormer + SAM | Automatic hybrid | 0.6093 | 0.7352 | 0.4131 | >242.8 ms |

SegFormer achieved the highest automatic segmentation accuracy, strongest
boundary quality, and lowest inference latency. It was selected as the
production model.

The SAM experiments remain useful for measuring zero-shot transfer, prompt
sensitivity, and the effect of task-specific fine-tuning.

## Metric Notes

- Primary metric: macro mean IoU across test images.
- A and B use oracle prompts derived from ground-truth masks for controlled
  prompt comparison.
- C is fully automatic and requires no prompt.
- D generates prompts automatically from SegFormer predictions.
- The recorded hybrid latency excludes part of the precomputed coarse-model
  stage, so real end-to-end hybrid latency is higher.
- Boundary distances are measured in pixels.
