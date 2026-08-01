# Dataset Card — Satellite Images of Water Bodies

## Source
- Dataset: Satellite Images of Water Bodies
- Source: Kaggle
- Valid pairs: 2841

## Split
- Train: 1991
- Validation: 429
- Test: 421
- Split origin: recovered from historical Experiment C registry
- Split registry SHA-256: `d7f0988bfd5e36c39fb5e1cffef323d2637a2b94daaa466f9a85e130432aa9d5`

## Validation
- Image-mask pairing by case-insensitive filename stem
- Every file decoded and dimension-checked
- Exact SHA-256 duplicate leakage rejected
- Perceptual-hash cross-split audit exported separately
- Split performed before any materialized tiling

## Data limitations
- The maintained model uses RGB only; no NIR or SWIR bands are available to the model.
- Geographic region, acquisition date, atmosphere, season, and sensor metadata are not
  consistently available in the registry.
- Geographic out-of-distribution generalization is therefore not established.
- GeoTIFF metadata is preserved where present, but many samples may be non-georeferenced PNGs
  or TIFFs.

## Intended use
Binary water/non-water segmentation research and controlled inference demonstrations.
