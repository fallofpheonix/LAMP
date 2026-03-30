# Raycasting Benchmark

## Setup Time
- Mesh generation and BVH build: 0.9539s

## Performance (per query)
- Baseline Mesh LOS: 0.001699s
- Aperture Mesh LOS (8 samples): 0.007848s

## Scaling
- Voxel-based (100x100 full viewshed): 0.0296s
- Mesh-based is O(log N) per ray and still better for sparse long-range queries.
