# CLI Reference

Main entrypoint:

```bash
PYTHONPATH=src python scripts/run_pipeline.py [options]
```

## Required for normal runs

- `--dem PATH`
- `--sar PATH`
- `--marks PATH`
- `--buildings PATH`

## Path prior

- `--path-prior-mode {learned,deterministic}`
- `--path-prior-raster PATH` (recommended when mode is `learned`; otherwise deterministic fallback is used)

## Sampling

- `--samples INT` (default: 256)
- `--max-pairs INT` (default: 0 => all pairs)
- `--top-k INT` (default: 8)
- `--temperature FLOAT` (default: 0.08)

## Cost weights

- `--w-slope FLOAT`
- `--w-roughness FLOAT`
- `--w-surface FLOAT`
- `--w-path-prior FLOAT`

## Calibration

- `--known-paths PATH`
- `--calibrate-weights`
- `--calibration-samples INT`

## Output

- `--out PATH`
- `--seed INT`
