# SETUP

## Root Package

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

## Legacy Submission CI Surfaces

### Task 1

```bash
cd submission/source/task1-path-tracing
python -m pip install --upgrade pip
pip install -r requirements.txt
PYTHONPATH=src python -m pytest tests/ -v --tb=short
```

### Task 2

```bash
cd submission/source/task2-viewsheds
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "GDAL==$(gdal-config --version)"
python -m pytest tests/ -v --tb=short
```

## System Dependencies

- `gdal-bin`
- `libgdal-dev`

## Validation Commands

```bash
PYTHONPATH=src python tests/integration/test_task1_synthetic.py
python scripts/run_path_tracing.py --help
python scripts/run_viewsheds.py --help
```
