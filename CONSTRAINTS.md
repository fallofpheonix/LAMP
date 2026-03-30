# CONSTRAINTS

## Engineering Constraints
- deterministic behavior is preferred over stochastic black-box redesigns
- existing task CLIs and file outputs must remain usable
- visibility coupling must be optional
- root config file remains the default control surface

## Data Constraints
- coupling depends on raster compatibility across tasks
- visibility input must be clipped to `[0,1]`
- obstacle and nodata handling must dominate all soft costs
- default repo config has no path labels enabled

## Environment Constraints
- local execution only
- current environment lacks `osgeo`, blocking Task 2 runtime verification
- no assumption of GPU availability

## Scope Constraints
- in scope:
  - Task 1 / Task 2 coupling
  - deterministic viewshed reuse
  - calibration and evaluation
  - GIS artifact generation
- out of scope:
  - new CV models
  - RL / GNN movement systems
  - audibility
  - game-engine visualization
  - multi-temporal scene graphs

## Compatibility Constraints
- backward compatibility:
  - `w_visibility = 0` preserves prior Task 1 behavior
- comparison mode must not alter the single-scenario execution path
- legacy docs under `submission/` are not authoritative

## Scientific Constraints
- visibility is treated as a measurable movement constraint, not as a replacement for terrain cost
- results must be reported as baseline-vs-coupled comparisons
- negative results are acceptable if the framework is valid and reproducible
