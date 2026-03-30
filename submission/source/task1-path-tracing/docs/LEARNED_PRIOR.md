# Learned Prior Guide (DeepLabV3+/U-Net)

## Objective
Replace deterministic path prior with a labeled-model output raster.

## Expected Raster

`path_prior_prob.tif` with:
- single band
- either probabilities `[0,1]` or logits/scores
- georeferenced CRS/transform

Pipeline behavior:
- reprojects/resamples raster to DEM grid
- if values are outside `[0,1]`, applies sigmoid
- clips final prior to `[0,1]`

## Recommended Training Setup

Input channels (example):
- SAR intensity / orthophoto bands
- slope band from DEM

Model options:
- DeepLabV3+
- U-Net

Output target:
- binary/soft path mask from labeled fragments

## Runtime Integration

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --path-prior-mode learned \
  --path-prior-raster path_prior_prob.tif
```

## Notes

1. Use deterministic mode only for debugging or data-unavailable scenarios.
2. Prefer path-prior raster generated from manually validated labels.
