# LAMP Task 1: Path Tracing Simulations

Terrain-aware probabilistic pathway inference for El Bagawat.

## Scope

Inputs:
- DEM raster
- SAR/imagery raster
- Building footprints
- Entrance-near terminals (`Marks_Brief1`)
- Optional: known path fragments (for calibration/evaluation)
- Optional: learned path-prior raster from DeepLabV3+/U-Net

Primary outputs:
- Predicted path vectors
- Path probability heatmap
- Lost path candidates
- Top-k recall/IoU (if known path fragments are provided)

## Repository Structure

```
.
├── docs/
├── scripts/
│   └── run_pipeline.py
│   └── check_task1_completion.py
├── src/
│   ├── preprocessing/
│   ├── vision/
│   ├── simulation/
│   ├── gis/
│   └── config.py
├── tests/
├── Task_1/
├── outputs/
├── requirements.txt
└── README.md
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use the virtualenv. System Python may reject package install (`externally-managed-environment`, PEP 668).

## Quick Start

### 0) Activate environment

```bash
source .venv/bin/activate
```

### 1) Train learned path prior (required for learned mode)

```bash
python scripts/train_cv_prior.py \
  --dem Task_1/DEM_Subset-Original.tif \
  --sar Task_1/SAR-MS.tif \
  --train-paths known_paths_train.shp \
  --eval-paths known_paths_eval.shp \
  --out-prior path_prior_prob.tif \
  --out-report outputs/prior_training_report.json \
  --out-eval-mask outputs/prior_eval_mask.tif \
  --buffer-m 4.5 \
  --neg-pos-ratio 2.0 \
  --n-estimators 200 \
  --seed 11
```

### 2) Learned prior mode (recommended)

`--path-prior-raster` should be a model output aligned/reprojectable to the DEM CRS.

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --dem Task_1/DEM_Subset-Original.tif \
  --sar Task_1/SAR-MS.tif \
  --marks Task_1/Marks_Brief1.shp \
  --buildings Task_1/BuildingFootprints.shp \
  --path-prior-mode learned \
  --path-prior-raster path_prior_prob.tif \
  --samples 512 \
  --max-pairs 0 \
  --top-k 8 \
  --out outputs
```

### 3) Calibration + metrics (requires known paths)

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --dem Task_1/DEM_Subset-Original.tif \
  --sar Task_1/SAR-MS.tif \
  --marks Task_1/Marks_Brief1.shp \
  --buildings Task_1/BuildingFootprints.shp \
  --path-prior-mode learned \
  --path-prior-raster path_prior_prob.tif \
  --known-paths known_path_fragments.shp \
  --calibrate-weights \
  --calibration-samples 128 \
  --samples 512 \
  --max-pairs 0 \
  --top-k 8 \
  --out outputs
```

### 4) Deterministic fallback mode (debug only)

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --path-prior-mode deterministic \
  --samples 256 \
  --max-pairs 0
```

If `--path-prior-mode learned` is used without `--path-prior-raster`, the runner automatically falls back to deterministic prior and records `"mode": "deterministic-fallback"` in `preprocess_report.json`.

## Core Algorithm

1. Align SAR to DEM grid.
2. Compute slope + roughness from DEM.
3. Load learned path-prior raster (`learned` mode) or deterministic proxy (`deterministic` mode).
4. Build weighted movement cost surface with obstacle mask.
5. Sample probabilistic routes per terminal pair using Monte Carlo A*.
6. Aggregate route density and derive dense corridors.
7. Extract skeletonized centerlines and vector layers.
8. If known paths are provided:
- calibrate weights on candidate grids
- report top-k recall and IoU

## Output Artifacts

Vectors:
- `outputs/predicted_paths.geojson`
- `outputs/predicted_paths.gpkg`
- `outputs/predicted_centerlines.geojson`
- `outputs/lost_path_centerlines.geojson`
- `outputs/lost_path_candidates.geojson`

Rasters:
- `outputs/probability_heatmap.tif`
- `outputs/movement_cost.tif`
- `outputs/detected_paths.tif`
- `outputs/movement_dense_mask.tif`
- `outputs/skeleton_mask.tif`
- `outputs/lost_paths_mask.tif`
- `outputs/known_paths_mask.tif` (if provided)
- `outputs/predicted_topk_mask.tif` (if provided)

Reports:
- `outputs/preprocess_report.json`
- `outputs/run_summary.json`
- `outputs/calibration_report.json` (if `--calibrate-weights`)
- `outputs/task1_completion.md` (from acceptance checker)

## Acceptance Check

```bash
python3 scripts/check_task1_completion.py \
  --output-dir outputs \
  --report-path outputs/task1_completion.md \
  --run-summary outputs/run_summary.json \
  --preprocess-report outputs/preprocess_report.json \
  --prior-report outputs/prior_training_report.json
```

Expected result for submission: `Required scope: PASS` and `Required + optional scope: PASS`.

## Submission Status

- Task 1 acceptance checker: PASS (required + optional)
- Submission state: submittable

## Complexity

Let:
- `V`: traversable cells
- `E`: 8-neighbor edges (`~8V`)
- `P`: processed terminal pairs
- `S`: Monte Carlo samples per pair

Runtime: `O(P * S * E log V)`
Memory: `O(V + E)`

## Documentation Index

- [Pipeline Overview](docs/PIPELINE.md)
- [Preprocessing Guide](docs/PREPROCESSING.md)
- [Learned Prior Guide (DeepLabV3+/U-Net)](docs/LEARNED_PRIOR.md)
- [Calibration and Metrics](docs/CALIBRATION.md)
- [CLI Reference](docs/CLI.md)
