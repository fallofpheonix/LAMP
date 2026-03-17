# LAMP Engineering Repository

This repository contains two geospatial pipelines and a lightweight operations layer for validation, diagnostics, security scanning, and benchmark reporting.

## What it does
- Runs Task 1 path tracing and Task 2 viewshed workflows.
- Validates core raster/vector inputs before pipeline runs.
- Generates ML diagnostics and raycasting benchmark reports.
- Performs a basic source-level security hygiene scan.

## Modular Layout
- `core/`: domain models and shared exceptions.
- `services/`: business logic for validation, diagnostics, audit, and benchmarks.
- `api/`: CLI adapters and command routing.
- `utils/`: narrow helper functions.
- `config/`: environment-driven defaults.
- `tests/`: repository-level smoke and critical-path unit tests.

Task-specific pipelines remain in `task1-path-tracing/` and `task2-viewsheds/`.

## Quick start
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r task1-path-tracing/requirements.txt
pip install -r task2-viewsheds/requirements.txt
```

Run repository-level tools:
```bash
python validate_dataset.py
python ml_diagnostics.py
python security_audit.py
python benchmark_raycast.py --samples 100
```

Or via unified CLI:
```bash
python -m api.cli validate-dataset
python -m api.cli security-audit
```

## Key decisions
- Kept top-level script names stable for compatibility with existing docs.
- Moved all operational logic into service modules for testability.
- Chose pragmatic validation and audit checks instead of full policy engines.
