# Demo / Usage Guide

## 1) Environment setup
```bash
cd source
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Run repository-level quality checks
```bash
python validate_dataset.py --out-report ../validation/dataset_integrity_submission.md
python security_audit.py --out ../validation/security_audit_submission.md
python benchmark_raycast.py --samples 25 --out ../validation/raycast_benchmark_submission.md
```

## 3) Verify task completion status
```bash
cd task1-path-tracing
python scripts/check_task1_completion.py --output-dir outputs --report-path ../../validation/task1_completion_submission.md --run-summary outputs/run_summary.json --preprocess-report outputs/preprocess_report.json --prior-report outputs/prior_training_report.json
cd ../task2-viewsheds
python scripts/check_task2_completion.py --output-dir outputs --report-path ../../validation/task2_completion_submission.md --metrics-path outputs/viewshed_model_metrics.json
```

## 4) Review sample outputs
- `../samples/2D_viewshed_overlay.png`
- `../samples/3D_visibility_scene.png`
- `../samples/visibility_histogram.png`
- `../validation/*.md`
