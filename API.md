# API

## Root CLI
Entry point: `src/lamp/api/cli.py`

Commands:
- `python -m lamp.api.cli validate-dataset`
- `python -m lamp.api.cli security-audit`
- `python -m lamp.api.cli benchmark-raycast`
- `python -m lamp.api.cli ml-diagnostics`

## Task 1 CLI
Entry point: `scripts/run_path_tracing.py`

Core arguments:
- `--dem`
- `--sar`
- `--marks`
- `--buildings`
- `--known-paths`
- `--path-prior-raster`
- `--path-prior-mode`
- `--out`
- `--samples`
- `--top-k`
- `--temperature`
- `--w-slope`
- `--w-roughness`
- `--w-surface`
- `--w-path-prior`
- `--w-visibility`
- `--visibility-raster`
- `--visibility-source`
- `--calibrate-weights`
- `--compare-visibility-coupling`

Task 1 config object:
- file: `src/lamp/tasks/path_tracing/config.py`
- type: `PipelineConfig`

Task 1 behavior:
- default behavior:
  - identical to prior pipeline when `w_visibility == 0` or no visibility raster is supplied
- comparison mode:
  - runs baseline and coupled scenarios in one execution

Task 1 primary outputs per scenario:
- `predicted_paths.geojson`
- `movement_cost.tif`
- `path_density.tif`
- `topk_path_mask.tif` when labels exist
- `summary.json`
- `preprocess_report.json`

Comparison-mode outputs:
- `baseline/...`
- `visibility_coupled/...`
- `comparison_density_delta.tif`
- `comparison_summary.json`
- `comparison_visibility_coupling.png`

## Task 2 CLI
Entry point: `scripts/run_viewsheds.py`

Arguments:
- `--data-dir`
- `--output-dir`
- `--scene-mode`
- `--observer-height`
- `--target-height`
- `--max-distance`

Task 2 outputs:
- `viewshed_probability.tif`
- `viewshed_all_observers.tif`
- `viewshed.tif`
- `viewshed.shp`
- `viewshed_observer_<id>.tif`
- `viewshed_observer_<id>.gpkg`
- `occlusion_surface.tif`

## Internal Interfaces
- Task 1 cost surface:
  - file: `src/lamp/tasks/path_tracing/simulation/cost_surface.py`
  - function: `compute_cost_surface(...)`
- Task 1 calibration:
  - file: `src/lamp/tasks/path_tracing/simulation/calibration.py`
  - functions:
    - `default_weight_grid(...)`
    - `calibrate_weights(...)`
    - `evaluate_topk_metrics(...)`
- Task 2 aggregate visibility:
  - file: `src/lamp/tasks/viewsheds/visibility.py`
  - function: `compute_multi_observer_visibility(...)`

## Contract: Visibility Coupling
- Input raster semantics:
  - mean visibility probability per cell
- Required value range:
  - `[0,1]`
- Required alignment:
  - CRS, transform, width, height must match Task 1 DEM or be reprojected to it
