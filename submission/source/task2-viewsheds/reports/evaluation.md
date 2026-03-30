# Evaluation Summary

Source metrics are synchronized from `outputs/viewshed_model_metrics.json` and `outputs/task2_completion.md`.

## Current Status
- Required scope: PASS
- Required + optional scope: PASS

## Key Validation Numbers
- Deterministic observer ratios: 0.1411, 0.0997, 0.1283
- 3D deterministic observer ratios: 0.1280, 0.0637, 0.0975
- Model F1 @ best threshold: 0.5549
- Model IoU @ best threshold: 0.3840

## Remaining Technical Gap
- Aperture-rich mesh/BVH raycasting is not yet integrated.
