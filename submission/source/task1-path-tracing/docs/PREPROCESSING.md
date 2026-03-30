# Preprocessing Guide

## 1. Metadata Inspection

Validate CRS, bounds, and resolution for DEM and imagery.

Pipeline output: `outputs/preprocess_report.json`

## 2. Grid Alignment

SAR is reprojected/resampled to DEM grid using bilinear interpolation.

Target grid fields:
- CRS
- width/height
- affine transform
- pixel size

## 3. DEM Features

Derived from DEM:
- normalized slope
- roughness (local residual vs Gaussian smooth)

## 4. Obstacles and Terminals

- Buildings rasterized into `obstacle_mask`
- Terminals loaded from marks shapefile and snapped to raster cells
- Terminal cells set traversable

## 5. Required Inputs

- `--dem`
- `--sar`
- `--marks`
- `--buildings`
- `--path-prior-raster` (learned mode)

## 6. Failure Conditions

- <2 valid terminals
- learned mode without `--path-prior-raster`
- CRS/projection incompatibility
