# SPEC

## Purpose
- Repository scope: deterministic geospatial pipelines for:
  - Task 1: probabilistic path tracing at El Bagawat
  - Task 2: deterministic multi-observer viewshed analysis
  - Task coupling: feed Task 2 visibility into Task 1 movement cost
- Repository role: engineering and evaluation surface for reproducible archaeological analysis, not a general ML sandbox.

## Problem Statements
- Task 1:
  - infer likely pedestrian corridors between site terminals
  - generate GIS-ready raster and vector outputs
  - support terrain, roughness, surface penalty, path priors, and optional visibility coupling
- Task 2:
  - compute per-observer binary viewsheds
  - aggregate mean visibility probability across observers
  - export raster and vector GIS artifacts
- Coupling:
  - test whether visibility changes path distributions and agreement with known archaeological paths

## Inputs
- Task 1:
  - `data-briefs/task1/DEM_Subset-Original.tif`
  - `data-briefs/task1/SAR-MS.tif`
  - `data-briefs/task1/Marks_Brief1.shp`
  - `data-briefs/task1/BuildingFootprints.shp`
  - optional known-path labels
  - optional `viewshed_probability.tif`
- Task 2:
  - `data-briefs/task2/DEM_Subset-Original.tif`
  - `data-briefs/task2/DEM_Subset-WithBuildings.tif`
  - `data-briefs/task2/Marks_Brief2.shp`
  - `data-briefs/task2/BuildingFootprints.shp`

## Outputs
- Task 1 baseline or coupled run:
  - movement cost raster
  - path density raster
  - vectorized predicted paths
  - run summary JSON
- Task 2 run:
  - per-observer viewshed rasters and polygons
  - `viewshed_probability.tif`
  - `viewshed_all_observers.tif`
  - required alias `viewshed.tif` and `viewshed.shp`
- Coupled comparison mode:
  - `baseline/`
  - `visibility_coupled/`
  - `comparison_density_delta.tif`
  - `comparison_summary.json`
  - `comparison_visibility_coupling.png`

## Functional Requirements
- Maintain current Task 1 behavior when visibility is disabled.
- Accept a visibility raster aligned or alignable to the Task 1 DEM grid.
- Use the extended cost model:

\[
C = w_s S + w_r R + w_t T + w_p(1-P) + w_v(1-V)
\]

- Enforce:
  - `w_s + w_r + w_t + w_p + w_v = 1`
  - `P, V ∈ [0,1]`
  - nodata and obstacle cells remain `inf`
- Support baseline-vs-coupled comparison in one execution.

## Success Metrics
- Engineering:
  - repo tests pass
  - comparison mode completes on aligned inputs
  - Task 1 still runs without visibility input
- Scientific:
  - compare baseline vs coupled using:
    - top-k recall
    - IoU
    - F1
    - path-density divergence
- Acceptance:
  - either improved agreement with known paths
  - or a valid negative-result coupling framework with reproducible outputs

## Non-Goals
- new deep learning model training
- RL / GNN integration
- audibility modeling
- Unreal / Unity / web visualization
- multi-temporal reconstruction
