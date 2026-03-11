# LAMP Screening Submission

This repository contains both LAMP screening tasks with runnable pipelines and acceptance checks.

## Layout
- `task1-path-tracing/`: Task 1 path tracing simulation pipeline
- `task2-viewsheds/`: Task 2 2.5D/3D viewshed pipeline
- `data-briefs/`: brief/task input manifests

## Reproducibility
Task 1:
```bash
cd task1-path-tracing
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/train_cv_prior.py --dem Task_1/DEM_Subset-Original.tif --sar Task_1/SAR-MS.tif --train-paths known_paths_train.shp --eval-paths known_paths_eval.shp --out-prior path_prior_prob.tif --out-report outputs/prior_training_report.json --out-eval-mask outputs/prior_eval_mask.tif --buffer-m 4.5 --neg-pos-ratio 2.0 --n-estimators 200 --seed 11
PYTHONPATH=src python scripts/run_pipeline.py --dem Task_1/DEM_Subset-Original.tif --sar Task_1/SAR-MS.tif --marks Task_1/Marks_Brief1.shp --buildings Task_1/BuildingFootprints.shp --known-paths known_path_fragments.shp --path-prior-mode learned --path-prior-raster path_prior_prob.tif --calibrate-weights --calibration-samples 64 --samples 64 --max-pairs 0 --top-k 4 --out outputs
python scripts/check_task1_completion.py --output-dir outputs --report-path outputs/task1_completion.md --run-summary outputs/run_summary.json --preprocess-report outputs/preprocess_report.json --prior-report outputs/prior_training_report.json
```

Task 2:
```bash
cd task2-viewsheds
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
./scripts/run_all.sh
python3 scripts/check_task2_completion.py --output-dir outputs --report-path outputs/task2_completion.md --metrics-path outputs/viewshed_model_metrics.json
```
