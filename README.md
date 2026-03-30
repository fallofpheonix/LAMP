# LAMP Engineering Repository

A unified geospatial engineering platform for path tracing, viewshed analysis, and operational diagnostics.

## Modular Architecture

This repository follows a professional, modular "big project" structure:

- `src/lamp/`: Main source package
  - `core/`: Domain models, exceptions, and shared configuration.
  - `services/`: Operational logic (ML diagnostics, validation, benchmarks).
  - `api/`: CLI adapters and command-line routing.
  - `tasks/`: Pipeline implementations.
    - `path_tracing/`: Task 1 path tracing logic.
    - `viewsheds/`: Task 2 visibility logic.
  - `shared/`: Common geospatial and terrain utilities.
  - `utils/`: Filesystem and I/O helpers.
- `scripts/`: Unified command-line entry points.
- `data/`: (Consolidation in progress) Centralized dataset storage.
- `outputs/`: Task-specific result folders.
- `tests/`: Consolidated test suite.

## Installation

```bash
pip install -e .
```

Requires: Python 3.10+, GDAL, and Geospatial Python stack (Rasterio, GeoPandas, etc.).

## Usage

### Root CLI
```bash
lamp validate-dataset
lamp security-audit
lamp ml-diagnostics
```

### Task 1: Path Tracing
```bash
python scripts/run_path_tracing.py --dem data/dem.tif --sar data/sar.tif ...
```

### Task 2: Viewsheds
```bash
python scripts/run_viewsheds.py --data-dir data/task2 --output-dir outputs/task2
```

