# Legacy Research Code

This directory preserves the original experimental modules used during the four-model investigation.

The legacy code is intentionally kept outside the active `waterseg` package because parts of the earlier training bundle are not present in the cleaned repository. Keeping broken imports in the production package would make the public project appear reproducible when it is not.

The archived modules support:

- methodological review;
- comparison with exported results;
- future restoration from the original source bundle;
- transparent documentation of the SAM and hybrid experiments.

They are not the production inference path.

The active production direction is SegFormer-B0.
