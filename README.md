# LAMP Engineering Repository

Deterministic geospatial pipelines for El Bagawat path tracing, viewshed analysis, and visibility-coupled path inference.

## Canonical Docs
- [SPEC.md](SPEC.md): system boundary
- [ARCH.md](ARCH.md): module map and data flow
- [STATE.md](STATE.md): current verified state
- [TASKS.md](TASKS.md): active execution frontier
- [API.md](API.md): CLI and interface surface
- [DATA.md](DATA.md): dataset and artifact contract
- [SETUP.md](SETUP.md): local execution requirements
- [CONSTRAINTS.md](CONSTRAINTS.md): non-negotiable limits

These root docs are the primary context surface. Legacy material under `submission/` is not authoritative.

## Repository Layout
- `src/lamp/`
  - `core/`
  - `api/`
  - `services/`
  - `tasks/path_tracing/`
  - `tasks/viewsheds/`
- `scripts/`
- `tests/`
- `data-briefs/`
- `outputs_production/`

## Primary Entry Points
- root operations:
```bash
PYTHONPATH=src python -m lamp.api.cli validate-dataset
PYTHONPATH=src python -m lamp.api.cli security-audit
PYTHONPATH=src python -m lamp.api.cli benchmark-raycast
PYTHONPATH=src python -m lamp.api.cli ml-diagnostics
```

- Task 1:
```bash
PYTHONPATH=src python scripts/run_path_tracing.py
```

- Task 2:
```bash
PYTHONPATH=src python scripts/run_viewsheds.py --data-dir data-briefs/task2 --output-dir outputs/task2
```

## Current Status
- Task 1 and Task 2 pipelines exist.
- Visibility coupling is implemented in Task 1.
- Local tests pass.
- Real Task 2 runtime validation is blocked by missing `osgeo`.
