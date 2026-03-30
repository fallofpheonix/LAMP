# CONSTRAINTS

## Repository Constraints

- Two active code layouts exist:
- root `src/lamp`
- legacy `submission/source`
- CI for PR `#5` executes legacy task trees, not the root package

## Dependency Constraints

- Task 2 code using `osgeo` requires a system GDAL install plus matching Python bindings
- Some root service code still depends on legacy modules and path injection

## Environment Constraints

- Current GitHub token has `repo` scope but not `workflow`
- consequence: workflow-file edits cannot be pushed from this environment
- CI runners use Python `3.11`

## Operational Constraints

- Raster operations are array-wide and memory-bound on large DEMs
- Full viewshed generation cost scales with raster size and observer count
- Calibration requires known path labels; it must be disabled when labels are absent

## Scope Constraints

- Current work is CI stabilization and context capture
- Full architectural unification is not complete on this branch
