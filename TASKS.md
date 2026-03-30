# TASKS

## IN PROGRESS
- Convert root docs into the canonical context surface.
- Keep Task 1 visibility coupling aligned with current config and CLI behavior.

## TODO
- Validate real Task 2 -> Task 1 handoff using `viewshed_probability.tif` from `scripts/run_viewsheds.py`.
- Install or provision GDAL / `osgeo` for the viewshed runtime.
- Run coupled evaluation against real known-path labels once label paths are available.
- Add regression test covering end-to-end compare mode artifact creation.
- Remove or clearly deprecate conflicting legacy docs under `submission/`.
- Add a single authoritative output schema description for generated artifacts.

## BLOCKED
- Real Task 2 execution:
  - blocked by missing `osgeo`
- Real archaeological metric comparison:
  - blocked by absent default known-path labels in `pipeline.yaml`

## DONE
- Removed inactive duplicate code under `submission/source/`.
- Repointed CI to the root package and root tests.
- Added visibility as an optional fifth cost term in Task 1.
- Added constrained visibility-weight calibration.
- Added raster alignment and reprojection for visibility coupling.
- Added `--compare-visibility-coupling`.
- Added smoke outputs for baseline vs coupled comparison.
- Added coupling-focused tests.
- Restored missing source shims in `src/lamp/core/`.

## Deferred
- new ML models
- RL / GNN movement modeling
- audibility pipeline
- multi-temporal scene management
- engine / visualization integrations
