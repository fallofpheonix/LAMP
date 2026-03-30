# Architecture

## Data Flow
1. Load DEM rasters (`dem_original.tif`, `dem_with_buildings.tif`).
2. Rasterize `BuildingFootprints.Elevation` to DEM grid.
3. Build occlusion surface (`provided`, `synthetic`, or `fused`).
4. Build voxelized 3D occupancy scene (terrain solids + building solids).
5. Optionally carve opening voxels from `openings.geojson` if provided.
6. Reproject observer marks (`EPSG:4326`) into DEM CRS (`EPSG:32636`).
7. Compute per-cell LOS visibility for each observer (2.5D and 3D variants).
8. Compute 3D visibility volume from selected observer and export VTK volume.
9. Train logistic visibility model on deterministic labels.
10. Infer model viewsheds for project observer marks.
11. Aggregate per-observer binaries into union and probability rasters.
12. Polygonize binary rasters to vector layers.

## Core Model
Two deterministic solvers are present:
- Raster LOS (2.5D)
- Voxel LOS (3D occupancy)

Definitions:
- `S(r,c)`: occlusion surface elevation at cell `(r,c)`
- Observer elevation:
  - `z_obs = S(r_obs,c_obs) + h_obs`
- Target elevation:
  - `z_tgt = S(r_tgt,c_tgt) + h_tgt`

For each target cell, LOS samples intermediate points between observer and target.
If any intermediate occlusion elevation exceeds LOS elevation, target is occluded.

3D voxel formulation:
- `V(k,r,c) ∈ {0,1}` occupancy grid where 1=solid.
- A target is visible iff all sampled voxels along segment `(obs_xyz, target_xyz)` are air.

## Complexity
Let:
- `O` = number of observers
- `H,W` = raster dimensions
- `Z` = voxel layers

Worst-case time:
- `O(O * H * W * max(H,W))`
- 3D ground viewshed: `O(O * H * W * L)` where `L` is sampled 3D segment length.
- 3D volume ray-march: `O(R * S)` (rays × steps per ray).

Space:
- `O(H*W)` per output raster
- `O(Z*H*W)` for occupancy/visibility volumes

## Module Map
- `src/load_data.py`: DEM loading, observer reprojection
- `src/buildings.py`: building height rasterization
- `src/scene.py`: occlusion surface construction
- `src/raycast.py`: LOS visibility kernel
- `src/voxel_scene.py`: 3D occupancy construction, 3D LOS, visibility volume
- `src/visibility.py`: multi-observer aggregation
- `src/export_gis.py`: raster/vector export
- `src/visualize.py`: hillshade generation
- `src/ml_features.py`: feature extraction for observer/cell pairs
- `src/ml_model.py`: NumPy logistic model training/inference
