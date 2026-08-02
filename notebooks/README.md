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
