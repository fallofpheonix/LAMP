# Architectural Weakness Analysis

## 1. Coupling Analysis
- **Task Interdependence**: While logic is modularized, `task2-viewsheds` still maintains its own `src/` directory which overlaps in purpose with `lamp_core`.
- **Inference Coupling**: ML models (RF in Task 1, Logistic in Task 2) are tightly bound to their respective training scripts, making it difficult to swap models without script modification.

## 2. Scalability Risks
- **Task 1 Parallelization**: Path simulation for terminal pairs is currently serial in `run_pipeline.py`, which becomes a bottleneck as the number of markers increases.
- **Voxel Scene Memory**: 3D voxel scenes in `task2` scale cubically. Large ROI with high resolution will rapidly deplete system RAM.

## 3. Performance Bottlenecks
- **GIS I/O**: Rasterio/GDAL loading is performed repeatedly for the same layers across different pipeline stages. High-frequency I/O on large DEMs (>1GB) will lag.
- **Raycasting**: The current mesh tessellation (`dem_to_mesh`) creates a very high triangle count. Without aperture sampling or GPU acceleration, this remains O(N log N) but with a large constant factor.

## 4. Configuration Management
- **Fragmented Schema**: Task 1 uses `config.json`, while Task 2 uses `argparse` defaults. This leads to configuration drift and makes systematic hyperparameter tuning difficult.

## 5. Memory Fragmentation
- **Intermediate Raster Buffers**: Feature extraction in `ml_features.py` and `dem_processing.py` creates multiple float32 copies of the ROI, which can lead to heap fragmentation during long simulation runs.

## Refactor Recommendations
- **Centralize Configuration**: Implement `config/pipeline.yaml` to unify all hyperparameters.
- **Expand lamp_core**: Move all shared ML feature engineering and GIS I/O into `lamp_core`.
- **Parallelize Task 1**: Use `concurrent.futures` to sample terminal pairs in parallel.
- **Aperture Sampling**: Integrate stochastic raycasting in `mesh_raycast.py` to reduce mesh density requirements while maintaining fidelity.
