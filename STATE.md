# STATE

## Snapshot
- Date: `2026-03-30`
- Repository mode: active local development
- Source of truth: root docs plus current `src/lamp/` and `scripts/`

## Completed
- Task 1 deterministic / learned-prior path tracing pipeline exists.
- Task 2 deterministic multi-observer viewshed pipeline exists.
- Duplicate archived code under `submission/source/` has been removed.
- Visibility coupling is implemented in Task 1:
  - optional `visibility_raster`
  - optional `cost_w_visibility`
  - optional comparison mode
- Cost surface supports 5-term weighting.
- Calibration supports constrained visibility-weight search.
- Task 1 compare mode emits:
  - `baseline/`
  - `visibility_coupled/`
  - `comparison_density_delta.tif`
  - `comparison_summary.json`
  - `comparison_visibility_coupling.png`
- Root test suite passes locally.

## Verified Locally
- `PYTHONPATH=src .venv/bin/python -m pytest`
  - result: `11 passed`
- smoke run:
  - `PYTHONPATH=src .venv/bin/python scripts/run_path_tracing.py --path-prior-mode deterministic --visibility-raster outputs/tmp/synthetic_visibility.tif --w-visibility 0.2 --compare-visibility-coupling --out outputs_production/visibility_compare_smoke`
  - result: success
- proof artifacts exist under:
  - `outputs_production/visibility_compare_smoke/`

## Partially Verified
- Task 2 artifact contract is code-level valid:
  - `scripts/run_viewsheds.py` writes `viewshed_probability.tif`
- Real Task 2 end-to-end execution is not currently verified in this environment.

## Current Blockers
- Missing `osgeo` bindings in both system Python and repo virtualenv prevent direct execution of:
  - `scripts/run_viewsheds.py`
- Real known-path evaluation remains dataset-dependent:
  - `known_paths_train`
  - `known_paths_eval`
  - currently `null` in default config

## Known Risks
- Legacy docs under `submission/` and old reports may contradict current root docs.
- Task coupling quality depends on visibility raster quality and alignment.
- Comparison mode is validated with synthetic aligned visibility input, not yet with real Task 2 output in this environment.

## Working Tree
- There are uncommitted source changes in:
  - `scripts/run_path_tracing.py`
  - `src/lamp/config/__init__.py`
  - `src/lamp/config/pipeline.yaml`
  - `src/lamp/tasks/path_tracing/config.py`
  - `src/lamp/tasks/path_tracing/simulation/calibration.py`
  - `src/lamp/tasks/path_tracing/simulation/cost_surface.py`
  - `tests/test_dataset_validation_service.py`
  - `tests/test_gis_outputs.py`
  - `tests/test_security_audit_service.py`
  - `tests/test_visibility_coupling.py`
  - `src/lamp/core/config.py`
  - `src/lamp/core/io.py`
  - `src/lamp/core/terrain.py`

## Last Reliable Commands
- tests:
  - `PYTHONPATH=src .venv/bin/python -m pytest`
- compare smoke:
  - `PYTHONPATH=src .venv/bin/python scripts/run_path_tracing.py --path-prior-mode deterministic --visibility-raster outputs/tmp/synthetic_visibility.tif --w-visibility 0.2 --compare-visibility-coupling --out outputs_production/visibility_compare_smoke`
