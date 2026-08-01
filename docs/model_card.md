# Model Card — Water SegFormer V3

## Model
- Version: `segformer-v3.0.0`
- Base model: `nvidia/segformer-b0-finetuned-ade-512-512`
- Classes: non-water (0), water (1)
- Input policy: aspect-ratio-preserving letterbox to 512 × 512
- Output: original-resolution binary mask
- Checkpoint SHA-256: `dc0af932cfbcb2f3ea9cfa0e18e0f8438636027de2579d534412f7472b4521a4`
- Split registry SHA-256: `d7f0988bfd5e36c39fb5e1cffef323d2637a2b94daaa466f9a85e130432aa9d5`

## Selection
- HPO objective: original-resolution validation IoU at threshold 0.50
- Same-code historical configuration included during confirmation
- Final threshold selected on validation only: `0.4500`
- Final seed fixed before test evaluation: `42`

## Held-out test
- Images: `421`
- IoU: `0.727625`
- Dice: `0.825825`
- Precision: `0.860920`
- Recall: `0.815868`
- Boundary F1: `0.655815`

## Operations
- Model-forward p50/p95: `12.049` /
  `20.808` ms
- End-to-end p50/p95: `31.910` /
  `39.532` ms
- Deployment status: production-candidate; Docker runtime validation pending

## Limitations
- RGB-only inference can confuse shadows, dark soil, cloud shadows, and narrow water channels.
- No geographic OOD benchmark is available.
- Empty-mask conventions are disclosed separately.
- Docker runtime validation remains a separate local validation requirement.
