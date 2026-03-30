# ARCH

## Top-Level Surfaces

- `src/lamp/`
- unified package surface for CLI, services, shared models, and task code
- `scripts/`
- root command entry points
- `tests/`
- root validation tests for service/report layers
- `submission/source/`
- legacy runnable submission package
- active CI surface for PR `#5`

## Module Map

- `lamp.api`
- CLI routing
- `lamp.services`
- report generation and operational jobs
- `lamp.core`
- dataclasses and exceptions
- `lamp.config`
- default path and pipeline settings
- `lamp.tasks.path_tracing`
- Task 1 pipeline
- `lamp.tasks.viewsheds`
- Task 2 pipeline
- `submission/source/task1-path-tracing`
- legacy Task 1 pipeline and tests
- `submission/source/task2-viewsheds`
- legacy Task 2 pipeline and tests

## Data Flow

### Task 1

- DEM + SAR + marks + buildings
- raster alignment / feature extraction
- slope + roughness + surface penalty + path prior
- cost surface
- probabilistic path sampling
- raster/vector export

### Task 2

- DEM + DEM-with-buildings + marks + footprints
- building-height rasterization
- occlusion surface build
- per-observer line-of-sight / viewshed
- aggregate visibility products
- raster/vector export

### Operations

- dataset files
- validators / scanners / benchmarks
- Markdown reports in `outputs/` or user-selected targets

## Interface Boundaries

- `lamp.api.*` calls `lamp.services.*`
- root task scripts call `lamp.tasks.*`
- legacy submission tests import task-local `src/`
- some root services still depend on legacy import paths via `sys.path` injection

## External Dependencies

- GDAL / `osgeo`
- Rasterio
- GeoPandas
- Shapely
- NumPy / SciPy / scikit-image / scikit-learn
- Matplotlib
