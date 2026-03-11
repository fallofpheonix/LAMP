# Run Quality Report

Generated: 2026-03-11 12:18:32 UTC

## Artifact Check
Status: PASS

## Deterministic Viewshed Ratios
- observer_1: `0.1411`
- observer_2: `0.0997`
- observer_3: `0.1283`

## Probability Raster Stats
- min: `0.0000`
- max: `1.0000`
- mean: `0.1230`

## Model Metrics
- label mode: `3d`
- validation accuracy: `0.7693`
- validation F1 @0.5: `0.4089`
- validation IoU @0.5: `0.2570`
- best threshold: `0.780`
- validation F1 @best: `0.5549`
- validation IoU @best: `0.3840`

## Readiness
- Runtime readiness: PASS
- Technical scope: baseline complete (2.5D + voxel-3D LOS + trained surrogate)
- Remaining for full architectural 3D claim: aperture-rich mesh raycasting and calibrated opening dataset
