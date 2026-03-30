# Pipeline Overview

## Problem
Infer probabilistic human movement pathways between buildings in El Bagawat and recover likely lost paths.

## Data Flow

1. `DEM` -> slope, roughness
2. `SAR/imagery` -> aligned to DEM grid
3. `learned path prior` (DeepLabV3+/U-Net probability raster) -> movement preference
4. `buildings` -> obstacle mask
5. `terminals` -> source/destination pair generation
6. `cost surface` -> Monte Carlo A* path sampling
7. `path density` -> threshold + skeletonization
8. `vectorization` -> GIS path layers

## Invariants

1. All raster operations occur on the DEM target grid.
2. Cost raster is finite only on traversable cells.
3. Terminals are forced traversable even if they overlap obstacle cells.
4. Path probabilities are bounded in `[0,1]`.

## Cost Model

`C = ws*slope + wr*roughness + wt*surface_penalty + wp*(1-path_prior)`

Weights are normalized internally.

## Lost Path Logic

`lost = (density >= percentile_92) AND (path_prior < 0.35)`

Centerlines are extracted from skeletonized dense corridors.

## Main Script

`/Users/fallofpheonix/Project/Human AI/Late Antiquity Modelling Project/Path Tracing Simulations/scripts/run_pipeline.py`
