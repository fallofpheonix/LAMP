# LAMP Views and Visibilities

Deterministic multi-observer visibility pipeline for the El Bagawat Task 2 dataset.

This repository implements reproducible **2.5D and voxel-3D line-of-sight (LOS)** solvers, then exports GIS layers, model predictions, and review-ready figures.

## Scope
- Included:
  - Multi-observer LOS visibility on DEM-aligned grid
  - Voxelized 3D LOS visibility engine and 3D volume export
  - Building occlusion via fused terrain/building elevation surface
  - Trainable NumPy visibility model (logistic baseline) from deterministic labels
  - GIS outputs (`GeoTIFF`, `Shapefile`, `GeoPackage`)
  - Reviewer visuals (`3D scene`, `2D overlay`, `histogram`)
- Not included:
  - True volumetric 3D visibility field
  - Window/translucency transmissivity model
  - High-capacity nonlinear ML surrogate model (beyond logistic baseline)

## Repository Layout
- `data/`: input rasters and shapefiles
- `src/`: core modules
- `scripts/run_viewshed.py`: main computation pipeline
- `scripts/run_viewshed_3d.py`: voxelized 3D viewshed and volume pipeline
- `scripts/train_viewshed_model.py`: train model from deterministic viewshed labels
- `scripts/predict_viewshed_model.py`: infer model viewsheds and export GIS layers
- `scripts/make_figures.py`: documentation/review figure generation
- `scripts/report_quality.py`: automatic run-quality report
- `scripts/check_task2_completion.py`: requirement-level completion check
- `configs/`: pipeline/model/validation defaults
- `tests/`: smoke validation suite
- `reports/`: reviewer-facing evaluation notes
- `outputs/`: generated GIS rasters/vectors
- `figures/`: exported PNG figures
- `docs/`: technical documentation

## Quick Start
```bash
cd lamp-3d-viewshed
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
./scripts/run_all.sh
```

Use the virtualenv. System Python may reject package install (`externally-managed-environment`, PEP 668).

## Acceptance Check
```bash
python3 scripts/check_task2_completion.py \
  --output-dir outputs \
  --report-path outputs/task2_completion.md \
  --metrics-path outputs/viewshed_model_metrics.json
```

Expected result for submission: `Required scope: PASS` and `Required + optional scope: PASS`.

## Submission Status
- Task 2 acceptance checker: PASS (required + optional)
- Submission state: submittable

## Primary Outputs
- `outputs/viewshed.tif`: binary viewshed for observer 1
- `outputs/viewshed.shp`: polygonized vector from `viewshed.tif`
- `outputs/viewshed_observer_<id>.tif`: per-observer binary viewshed
- `outputs/viewshed_observer_<id>.gpkg`: per-observer vectorized layer
- `outputs/viewshed_all_observers.tif`: binary union across observers
- `outputs/viewshed_probability.tif`: observer consensus probability
  - Values for 3 observers: `{0, 1/3, 2/3, 1}`
- `outputs/viewshed3d.tif`: 3D-engine binary viewshed for observer 1
- `outputs/viewshed3d.shp`: vectorized 3D-engine viewshed
- `outputs/viewshed3d_probability.tif`: 3D-engine observer consensus
- `outputs/viewshed3d_volume.vtk`: optional 3D visibility volume
- `outputs/viewshed_model.npz`: trained model parameters
- `outputs/viewshed_model_metrics.json`: train/validation metrics
- `outputs/viewshed_model_probability.tif`: model-predicted mean probability
- `outputs/viewshed_model_union.shp`: model-predicted binary union (vector)
- `outputs/run_quality_report.md`: artifact and metric readiness report
- `outputs/task2_completion.md`: required vs optional completion status
- `figures/3D_visibility_scene.png`
- `figures/2D_viewshed_overlay.png`
- `figures/visibility_histogram.png`

## Documentation Index
- [Architecture](docs/architecture.md)
- [Usage](docs/usage.md)
- [Validation](docs/validation.md)
- [Limitations and Roadmap](docs/limitations.md)
- [Evaluation Summary](reports/evaluation.md)
