# Configuration Files

This directory will contain the maintained SegFormer configuration and HPO search space.

Planned files:

- `segformer.yaml`: full training configuration
- `hpo.yaml`: constrained Optuna study configuration
- `inference.yaml`: optional serving defaults

Configuration files should record:

- model identifier;
- image resolution;
- optimizer and scheduler;
- learning rate and weight decay;
- batch size and gradient accumulation;
- loss weights;
- epoch and early-stopping policy;
- random seed;
- threshold candidates;
- output paths.
