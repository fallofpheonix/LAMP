# LAMP Submission Package

## Project overview
This package delivers a production-ready implementation of the LAMP screening project:
- **Task 1**: terrain-aware path tracing with GIS exports
- **Task 2**: multi-observer 2.5D/3D viewshed computation with model outputs
- **Repo operations layer**: dataset validation, diagnostics, security audit, and benchmarking

## Package structure
- `source/` — complete runnable source code
- `docs/` — architecture, API, and usage documentation
- `validation/` — generated validation and acceptance reports
- `samples/` — output images for reviewer verification

## Setup instructions
From `submission/source/`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run instructions
From `submission/source/`:

### Repository-level tooling
```bash
python validate_dataset.py --out-report ../validation/dataset_integrity_submission.md
python security_audit.py --out ../validation/security_audit_submission.md
python benchmark_raycast.py --samples 25 --out ../validation/raycast_benchmark_submission.md
```

### Task 1 acceptance check
```bash
cd task1-path-tracing
python scripts/check_task1_completion.py \
  --output-dir outputs \
  --report-path ../../validation/task1_completion_submission.md \
  --run-summary outputs/run_summary.json \
  --preprocess-report outputs/preprocess_report.json \
  --prior-report outputs/prior_training_report.json
cd ..
```

### Task 2 acceptance check
```bash
cd task2-viewsheds
python scripts/check_task2_completion.py \
  --output-dir outputs \
  --report-path ../../validation/task2_completion_submission.md \
  --metrics-path outputs/viewshed_model_metrics.json
cd ..
```

## Dependencies
Declared in `source/requirements.txt`:
- Task pipelines: requirements inherited from both tasks
- Repo operations: `numpy`, `rasterio`, `geopandas`, `matplotlib`, `scikit-learn`

## Key decisions
- Preserved original task pipelines and outputs; avoided risky rewrites.
- Added modular service/api layers for operational scripts.
- Kept validation artifacts in `validation/` for direct reviewer access.
