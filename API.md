# API

## Root CLI

- `lamp validate-dataset`
- args: `--dem`, `--sar`, `--marks`, `--buildings`, `--out-report`
- effect: writes Markdown integrity report
- `lamp security-audit`
- args: `--root`, `--out`
- effect: scans Python files for heuristic path-traversal patterns and writes Markdown report
- `lamp benchmark-raycast`
- args: `--samples`, `--out`
- effect: benchmarks raycast implementations and writes Markdown report
- `lamp ml-diagnostics`
- args: `--dem`, `--sar`, `--paths`, `--eval-paths`, `--out-dir`
- effect: trains diagnostic model and writes figures / metrics

## Service Contracts

- `validate_raster_layer(path: Path) -> RasterValidation`
- reads one raster and returns CRS, shape, resolution, nodata stats, bounds
- `validate_vector_layer(path: Path) -> VectorValidation`
- reads one vector layer and returns CRS, feature counts, geometry validity, bounds
- `find_crs_mismatches(reference_crs: str, rasters: list[RasterValidation], vectors: list[VectorValidation]) -> list[Path]`
- returns asset paths whose CRS differs from the reference
- `run_raycast_benchmark(samples: int = 100) -> BenchmarkResult`
- returns timing metrics for mesh and voxel visibility operations
- `run_diagnostics(dem_path: Path, sar_path: Path, train_paths_path: Path, eval_paths_path: Path, out_dir: Path) -> None`
- writes figures and `metrics.json`

## Task Entry Points

- [scripts/run_path_tracing.py](/Users/fallofpheonix/Project/Human%20AI/LAMP/scripts/run_path_tracing.py)
- primary Task 1 runner from repo root
- outputs raster/vector artifacts and summary JSON
- [scripts/run_viewsheds.py](/Users/fallofpheonix/Project/Human%20AI/LAMP/scripts/run_viewsheds.py)
- primary Task 2 runner from repo root
- outputs occlusion and viewshed rasters/vectors

## Failure Modes

- missing raster/vector files
- CRS mismatch across layers
- marks outside raster bounds
- missing GDAL / Rasterio / GeoPandas runtime dependencies
