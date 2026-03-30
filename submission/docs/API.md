# API Documentation

This project exposes a CLI API via `python -m api.cli` and compatible script entrypoints.

## Unified CLI

```bash
python -m api.cli <command> [options]
```

### Commands

#### `validate-dataset`
Validates raster/vector integrity and writes a markdown report.

Key options:
- `--dem`
- `--sar`
- `--marks`
- `--buildings`
- `--out-report`

#### `security-audit`
Runs heuristic path traversal checks and writes a markdown report.

Key options:
- `--root`
- `--out`

#### `benchmark-raycast`
Runs mesh and voxel raycast benchmarks and writes markdown output.

Key options:
- `--samples`
- `--out`

#### `ml-diagnostics`
Runs feature/PR diagnostics for task1 learned prior inputs.

Key options:
- `--dem`
- `--sar`
- `--paths`
- `--eval-paths`
- `--out-dir`

## Backward-compatible script entrypoints
- `python validate_dataset.py`
- `python security_audit.py`
- `python benchmark_raycast.py`
- `python ml_diagnostics.py`
