# SPEC

## Problem

- Deliver one repository for:
- Task 1: terrain-aware path tracing from DEM, SAR, marks, and building footprints.
- Task 2: multi-observer viewshed analysis over terrain and buildings.
- Operational tooling: dataset validation, ML diagnostics, security audit, and raycast benchmarking.

## Inputs

- Task 1 rasters:
- DEM
- SAR
- Task 1 vectors:
- observer/terminal marks
- building footprints
- optional known path labels for calibration and diagnostics
- Task 2 rasters:
- bare-earth DEM
- DEM with buildings
- Task 2 vectors:
- observer marks
- building footprints

## Outputs

- Task 1:
- movement cost raster
- probability heatmap raster
- predicted path vectors
- centerline vectors
- run summary and optional calibration report
- Task 2:
- occlusion surface raster
- per-observer viewsheds
- aggregate probability / any-visible rasters
- required alias outputs `viewshed.tif` and `viewshed.shp`
- optional model metrics and completion report
- Operations:
- Markdown reports for validation, diagnostics, security, and benchmarking

## Constraints

- Python `>=3.10`
- Geospatial stack required: GDAL bindings, Rasterio, GeoPandas, Shapely
- CRS alignment is mandatory across all rasters and vectors
- Large rasters are processed in-memory; runtime and memory scale with raster area
- Current repository contains both unified `src/lamp` code and legacy `submission/source` task trees

## Success Metrics

- Task 1 unit tests and synthetic integration run without manual path edits
- Task 2 unit tests run in CI and produce no import-time dependency failures
- Root operational CLIs emit reports from repository root
- Required GIS artifacts are generated with projected CRS and non-empty dimensions
