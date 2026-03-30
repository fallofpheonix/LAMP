#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"
python3 scripts/run_viewshed.py
python3 scripts/run_viewshed_3d.py
python3 scripts/train_viewshed_model.py
python3 scripts/predict_viewshed_model.py
python3 scripts/make_figures.py
python3 scripts/report_quality.py
python3 scripts/check_task2_completion.py

echo "Done: deterministic(2D+3D) outputs, model artifacts, figures, quality report, and completion check refreshed"
