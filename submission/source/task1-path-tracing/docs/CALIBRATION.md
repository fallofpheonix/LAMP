# Calibration and Metrics

## Inputs

- Known path fragments layer via `--known-paths`
- Candidate cost weights

## Calibration Process

1. Rasterize known path fragments to DEM grid.
2. Evaluate candidate weight tuples.
3. For each tuple:
- run Monte Carlo A* over all terminal pairs
- build predicted top-k mask
- compute recall/IoU/precision/F1
4. Select best tuple by `0.65*IoU + 0.35*Recall`.

## Metrics

- `topk_recall`: overlap recall on known path mask
- `iou`: intersection-over-union
- `precision`
- `f1`

## Command

```bash
PYTHONPATH=src python scripts/run_pipeline.py \
  --known-paths known_path_fragments.shp \
  --calibrate-weights \
  --calibration-samples 128 \
  --samples 512 \
  --max-pairs 0
```

## Reports

- `outputs/calibration_report.json`
- `outputs/run_summary.json` (selected weights + metrics)
- `outputs/predicted_topk_mask.tif`
- `outputs/known_paths_mask.tif`
