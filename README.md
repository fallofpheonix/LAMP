# LAMP: Late Antiquity Modeling Project

[![CI](https://github.com/fallofpheonix/LAMP/actions/workflows/ci.yml/badge.svg)](https://github.com/fallofpheonix/LAMP/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deterministic geospatial pipelines for archaeological path tracing, 3D viewshed analysis, and visibility-coupled movement inference.

## Overview

LAMP models movement and visibility in the El Bagawat necropolis using two explicit task surfaces:

- Task 1: probabilistic path tracing over slope, roughness, surface penalty, path priors, and optional visibility coupling
- Task 2: 2D and voxel-3D viewshed generation over terrain and building geometry

The coupling contract is raster-based: Task 2 produces `viewshed_probability.tif`, and Task 1 can consume that aligned raster as an additional movement-cost term.

## Repository Layout

```text
.
├── assets/           Curated figures used in the README
├── data/             Shipped sample GIS inputs for Task 1 and Task 2
├── docs/research/    Research and proposal material kept for project context
├── scripts/          Compatibility wrappers and research utilities
├── src/lamp/         Package source
│   ├── api/          Unified CLI entry points
│   ├── core/         Shared config, IO, terrain, and exception utilities
│   ├── services/     Validation, diagnostics, benchmark, and audit services
│   └── tasks/        Task 1 and Task 2 implementations
└── tests/            Unit and integration tests
```

## Installation

Requirements:

- Python 3.10+
- GDAL/OGR runtime for Task 2 and GDAL-backed utilities

Recommended local install:

```bash
python -m pip install -e ".[dev]"
```

On Linux, install GDAL system packages before Python dependencies. The GitHub Actions workflow shows the expected Ubuntu setup.

## CLI

The package CLI is the primary interface:

```bash
lamp path-tracing --max-pairs 1 --samples 8
lamp viewsheds-2d
lamp viewsheds-3d
lamp validate-dataset
lamp security-audit
lamp benchmark-raycast
```

Notes:

- Task 1 defaults to a deterministic path prior so a clean clone does not require an untracked `path_prior_prob.tif`.
- Task 2 defaults to the shipped `data/task2/` dataset layout.
- `lamp ml-diagnostics` requires explicit training and evaluation path labels because those labels are not shipped in this repository.

The legacy script entry points remain available as wrappers:

```bash
python scripts/run_path_tracing.py
python scripts/run_viewsheds.py
python scripts/run_viewsheds_3d.py
```

## Data

Shipped datasets live under:

- `data/task1/` for path-tracing inputs
- `data/task2/` for viewshed inputs

The repository intentionally does not ship all derived labels and trained artifacts. Commands that require those artifacts accept them explicitly as arguments.

## Testing

Run the test suite from the repository root:

```bash
python -m pytest -q
```

## Research Context

Research and proposal material is preserved under `docs/research/` to document project intent, evaluation framing, and coupling rationale without mixing it into the runtime package surface.

## Visualizations

![3D Visibility](assets/3D_visibility_scene.png)

![Path Comparison](assets/2D_viewshed_overlay.png)
