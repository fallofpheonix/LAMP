# SETUP

## Environment
- working directory:
  - `/Users/fallofpheonix/Project/Human AI/LAMP`
- primary virtualenv:
  - `.venv/`
- Python path for local execution:
  - `PYTHONPATH=src`

## Required Python Packages
- core:
  - `numpy`
  - `rasterio`
  - `geopandas`
  - `shapely`
  - `pytest`
- optional / task-specific:
  - `matplotlib` for comparison figures
  - `scikit-image` for skeletonization fallback path
  - `osgeo` / GDAL Python bindings for `scripts/run_viewsheds.py`

## Install Pattern
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Verification Commands
- tests:
```bash
PYTHONPATH=src .venv/bin/python -m pytest
```
- Task 1 compare smoke:
```bash
PYTHONPATH=src .venv/bin/python scripts/run_path_tracing.py \
  --path-prior-mode deterministic \
  --visibility-raster outputs/tmp/synthetic_visibility.tif \
  --w-visibility 0.2 \
  --compare-visibility-coupling \
  --out outputs_production/visibility_compare_smoke
```

## Current Environment Limitation
- `scripts/run_viewsheds.py` requires `from osgeo import gdal`
- current local environments do not provide `osgeo`
- consequence:
  - Task 2 real-data execution is blocked until GDAL Python bindings are installed

## Expected Runtime Surfaces
- Task 1:
  - local CPU execution
  - file-based outputs in `outputs_production/`
- Task 2:
  - local CPU execution
  - GDAL-backed raster writes

## Generated Proof Artifacts
- current verified comparison outputs:
  - `outputs_production/visibility_compare_smoke/comparison_summary.json`
  - `outputs_production/visibility_compare_smoke/comparison_density_delta.tif`
  - `outputs_production/visibility_compare_smoke/comparison_visibility_coupling.png`
