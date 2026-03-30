# STATE

## Snapshot

- Date: `2026-03-30`
- Active PR target: `#5`
- PR head branch: `copilot/check-ci-cd-error`
- Local integration branch: `codex/pr5-codefix`
- Last locally validated commit before this patch: `59b05aa`

## Completed

- Root `src/lamp` package exists for CLI, services, shared models, and task code
- Legacy Task 1 CI surface imports resolve via task-local `src/lamp_core`
- Legacy Task 2 CI surface imports resolve via task-local `src/lamp_core`
- Task 1 synthetic integration no longer requires calibration labels
- Task 2 artifact smoke tests skip when generated outputs are absent
- Local validation previously passed for:
- `submission/source/task1-path-tracing` unit tests
- `submission/source/task1-path-tracing` synthetic integration
- `submission/source/task2-viewsheds` unit tests

## In Progress

- PR `#5` CI stabilization
- Task 2 Python dependency closure for CI (`rasterio` requirement added in working tree)
- Context document set for GPT upload / repo orientation

## Broken or Incomplete

- Root services are not fully decoupled from legacy code:
- [src/lamp/services/ml_diagnostics_service.py](/Users/fallofpheonix/Project/Human%20AI/LAMP/src/lamp/services/ml_diagnostics_service.py) injects legacy paths and imports `lamp_core`
- [src/lamp/services/raycast_benchmark_service.py](/Users/fallofpheonix/Project/Human%20AI/LAMP/src/lamp/services/raycast_benchmark_service.py) injects legacy paths and imports `src.*`
- Root path defaults still point at legacy dataset locations, not a clean centralized `data/` layout
- Repository contains duplicated execution surfaces: root package and `submission/source`

## Known Operational Constraints

- Updating `.github/workflows/*` requires GitHub token scope `workflow`
- Current GitHub token lacks that scope
- Task 2 runtime paths that import `osgeo` still require system GDAL bindings
