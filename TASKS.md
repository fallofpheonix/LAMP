# TASKS

## In Progress

- Close PR `#5` CI failures by fixing legacy submission test/runtime assumptions
- Add GPT-ingestion context documents at repository root

## TODO

- Replace legacy `sys.path` injection in root services with package imports under `lamp.tasks.*`
- Align root default paths with a single repository-level data layout
- Remove or freeze duplicate code between `src/lamp/` and `submission/source/`
- Add root-level tests for root CLI entry points and script entry points
- Normalize packaging metadata in root `pyproject.toml` to match actual build/install flow

## Blocked

- Workflow-file edits from this environment
- blocker: GitHub token lacks `workflow` scope
- End-to-end Task 2 runtime verification on machines without GDAL Python bindings
- blocker: `osgeo` import availability depends on system install

## Done Recently

- Vendored `lamp_core` into legacy Task 1 and Task 2 `src/` trees
- Fixed Task 1 synthetic integration configuration
- Hardened Task 2 smoke tests against missing generated artifacts
- Pushed CI-fix commit `59b05aa` to PR `#5`
