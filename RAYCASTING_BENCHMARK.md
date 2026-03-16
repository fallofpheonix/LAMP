# Raycasting Benchmark

## Setup Time
- Mesh generation and BVH build: 9.6916s

## Performance (per query)
- Baseline Mesh LOS: 0.001893s
- Aperture Mesh LOS (8 samples): 0.014864s

## Scaling
- Voxel-based (100x100 full viewshed): 0.1323s
- Mesh-based is O(log N) per ray, making it superior for sparse long-range queries.
