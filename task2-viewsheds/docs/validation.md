# Validation Protocol

## Mandatory Checks
1. **CRS alignment**
- Confirm all outputs are `EPSG:32636`.
- Overlay `viewshed.tif` and `dem_original.tif` in QGIS.

2. **Observer placement correctness**
- Confirm each observer is inside raster bounds.
- Confirm observer elevation is terrain-relative (`z = surface + eye_height`).

3. **Occlusion behavior**
- Verify reduced visibility behind dense building clusters.
- Verify topography-induced shielding in low/behind-slope areas.

4. **Raster/vector consistency**
- `viewshed.shp` polygons must match binary connected regions in `viewshed.tif`.

## Current Run Reference (Task_2 Data)
Observed per-observer ratios:
- Observer 1: `0.1411`
- Observer 2: `0.0997`
- Observer 3: `0.1283`

Interpretation: these are `visible_cells / checked_cells` in current configuration.

## Derived Output Semantics
- `viewshed_observer_<id>.tif`: binary visibility per observer
- `viewshed_probability.tif`: mean of observer binaries

For `N=3` observers, valid values are `{0, 1/3, 2/3, 1}`.

## Model Baseline Reference
`train_viewshed_model.py` currently reports validation metrics against deterministic labels from the three project marks.

Latest reference run:
- Accuracy: `0.7525`
- Precision: `0.3055`
- Recall: `0.7948`
- F1 @ threshold 0.5: `0.4414`
- IoU @ threshold 0.5: `0.2832`
- Best threshold (validation-selected): `0.700`
- F1 @ best threshold: `0.5424`
- IoU @ best threshold: `0.3721`

Metric file:
- `outputs/viewshed_model_metrics.json`
- `outputs/run_quality_report.md`
