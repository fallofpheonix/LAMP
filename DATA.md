# DATA

## Task 1 Required Inputs

- DEM: `DEM_Subset-Original.tif`
- SAR: `SAR-MS.tif`
- observer / terminal marks: `Marks_Brief1.shp`
- building footprints: `BuildingFootprints.shp`

## Task 1 Optional Inputs

- training labels: `known_paths_train.shp`
- evaluation labels: `known_paths_eval.shp`

## Task 2 Required Inputs

- bare-earth DEM: `dem_original.tif`
- DEM with buildings: `dem_with_buildings.tif`
- observer marks: `Marks_Brief2.shp`
- building footprints: `BuildingFootprints.shp`

## Invariants

- rasters and vectors must use a projected CRS
- Task 1 rasters are aligned to the DEM grid before feature generation
- nodata values are converted to `NaN` where applicable
- observer marks outside raster bounds are discarded
- building footprints are rasterized onto the reference DEM grid

## Active Data Locations

- root config still references legacy Task 1 paths such as `task1-path-tracing/Task_1/...`
- legacy submission Task 2 test data lives under `submission/source/task2-viewsheds/data/`
- repository-level `data/` is not yet authoritative on this branch

## Edge Cases

- empty rasters
- invalid / empty geometries
- CRS mismatch
- no valid observer marks after reprojection / clipping
- missing known-path labels while calibration is enabled
