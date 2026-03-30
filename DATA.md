# DATA

## Canonical Dataset Roots
- Task 1: `data-briefs/task1/`
- Task 2: `data-briefs/task2/`

## Task 1 Files
- terrain:
  - `DEM_Subset-Original.tif`
  - `DEM_Subset-WithBuildings.tif`
- imagery:
  - `SAR-MS.tif`
  - `OrthoImage_Subset.tif`
- vectors:
  - `Marks_Brief1.shp`
  - `BuildingFootprints.shp`
- reference material:
  - `Site_Map_With_ROI.png`
  - `Site_Plan.pdf`
  - `LAMP_Project_Page.md`

## Task 2 Files
- terrain:
  - `DEM_Subset-Original.tif`
  - `DEM_Subset-WithBuildings.tif`
- imagery:
  - `OrthoImage_Subset.tif`
- vectors:
  - `Marks_Brief2.shp`
  - `BuildingFootprints.shp`
- reference material:
  - `Site_Map_With_ROI.png`
  - `Site_Plan.pdf`

## Alignment Assumptions
- Current default configuration assumes:
  - Task 1 rasters share a common CRS and grid
  - Task 2 rasters share a common CRS and grid
- Coupling rule:
  - Task 1 DEM is the target grid
  - Task 2 visibility raster is reprojected if needed

## Feature Semantics
- slope:
  - derived from Task 1 DEM
- roughness:
  - derived from local DEM variation
- surface penalty:
  - derived from SAR plus terrain features
- path prior:
  - deterministic or learned raster in `[0,1]`
- visibility probability:
  - mean fraction of observer viewsheds covering a cell

## Label Availability
- default config:
  - `known_paths_train: null`
  - `known_paths_eval: null`
- implication:
  - evaluation metrics requiring path labels are optional until labels are supplied

## Output Data Contract
- rasters:
  - GeoTIFF
- vectors:
  - GeoJSON, GPKG, or SHP depending on pipeline step
- comparison artifact root:
  - `outputs_production/<run_name>/`

## Data Quality Risks
- missing labels
- raster misalignment across tasks
- nodata propagation
- environment-specific GDAL behavior
