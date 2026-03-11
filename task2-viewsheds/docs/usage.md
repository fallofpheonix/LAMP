# Usage

## 1. Install
```bash
cd lamp-3d-viewshed
python3 -m pip install -r requirements.txt
```

## 2. Full Pipeline (Deterministic + Model + Figures + Report)
```bash
./scripts/run_all.sh
```

## 3. Run Viewshed Pipeline
```bash
python3 scripts/run_viewshed.py
```

CLI options:
```text
--data-dir <path>          default: data
--output-dir <path>        default: outputs
--scene-mode <mode>        provided|synthetic|fused (default: fused)
--observer-height <m>      default: 1.6
--target-height <m>        default: 0.0
--max-distance <m>         default: none
```

Example:
```bash
python3 scripts/run_viewshed.py --scene-mode fused --observer-height 1.6 --max-distance 120
```

## 4. Run 3D Voxel Viewshed + Volume
```bash
python3 scripts/run_viewshed_3d.py
```

Primary 3D outputs:
- `outputs/viewshed3d.tif`
- `outputs/viewshed3d.shp`
- `outputs/viewshed3d_probability.tif`
- `outputs/viewshed3d_volume.vtk`

## 5. Generate Figures
```bash
python3 scripts/make_figures.py
```

CLI options:
```text
--data-dir <path>          default: data
--output-dir <path>        default: outputs
--figures-dir <path>       default: figures
--observer-height <m>      default: 1.6
```

## 6. Open in QGIS
Load these layers:
- `outputs/occlusion_surface.tif`
- `outputs/viewshed_probability.tif`
- `outputs/viewshed_all_observers.tif`
- `outputs/viewshed.shp`
- `outputs/viewshed3d_probability.tif`
- `outputs/viewshed3d.shp`

Expected CRS: `EPSG:32636`.

## 7. Train and Predict Model Viewsheds
Train:
```bash
python3 scripts/train_viewshed_model.py
```

Predict:
```bash
python3 scripts/predict_viewshed_model.py
```

Model outputs:
- `outputs/viewshed_model.npz`
- `outputs/viewshed_model_metrics.json`
- `outputs/viewshed_model_prob_observer_<id>.tif`
- `outputs/viewshed_model_bin_observer_<id>.tif`
- `outputs/viewshed_model_bin_observer_<id>.gpkg`
- `outputs/viewshed_model_probability.tif`
- `outputs/viewshed_model_union.shp`

## 8. Generate Quality Report
```bash
python3 scripts/report_quality.py
```

Output:
- `outputs/run_quality_report.md`
