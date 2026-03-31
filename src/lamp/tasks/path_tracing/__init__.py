"""Task 1 path-tracing pipeline: probabilistic pedestrian-path inference.

Loads DEM, SAR, and known-mark shapefiles; derives terrain features;
and samples Monte Carlo A* paths weighted by slope, roughness, surface
penalty, a path prior, and an optional visibility coupling term from
Task 2 outputs.
"""
