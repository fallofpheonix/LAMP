---
layout: default
title: Late Antiquity Modelling Project (LAMP)
---

# HumanAI Activities: Late Antiquity Modelling Project

## Description
The Late Antiquity Modeling Project (LAMP) is an international, interdisciplinary research collective. LAMP uses computational methods to reconstruct embodied experiences of ancient buildings and landscapes and to rewrite early Christian history with spatially grounded evidence.

Current focus: the late antique necropolis of El Bagawat in Egypt's Kharga Oasis.

## Duration
- Total project length: **350 hours**

## Task 1: Path Tracing Simulations
### Goal
Infer likely movement pathways between buildings, including pathways no longer visible today.

### Required modeling constraints
- Existing known pathways
- Building neighborhood/topology relationships
- Entrance locations/orientation
- Terrain and walking surface constraints

### Expected results
- Train a CV model for path identification from site data (DEM + SAR/imagery + vectors).
- Export inferred pathways as GIS vector layers.

## Task 2: Views and Visibilities
### Goal
Compute viewsheds and visibility gradients at El Bagawat using full 3D site context.

### Required modeling constraints
- Hilly terrain topography
- Building footprints and heights
- 3D line-of-sight constraints (not only planimetric 2D)

### Expected results
- Train a model that projects 3D viewsheds from the site scene.
- Export viewsheds as GIS vector layers.
- Optional: export a 3D visibility volume.

## Requirements
- GIS experience (DEM/SAR preferred)
- Image processing / computer vision experience
- QGIS (+ Orfeo Toolbox), ERDAS IMAGINE, or equivalent GIS/SAR stack
- Difficulty level: **Medium/Hard**

## Mentors
### Project Directors
1. Camille Leon Angelo (University of Alabama)
2. Joshua Silver (Karlsruhe Institute of Technology)

### Project Collaborators
1. Rachel Dubose (University of Alabama)
2. Jefferey Turner (University of Alabama)
3. Richard Newton (University of Alabama)

## Recommended Project Structure
### Page information architecture
1. Summary (description + duration + scope)
2. Task definitions (Task 1 and Task 2 goals/constraints/expected outputs)
3. Data + methods (inputs, pipeline stages, model assumptions)
4. Progress dashboard (percent complete, dated evidence)
5. Validation status (metrics, thresholds, known limitations)
6. Delivery artifacts (GIS layers, reports, figures)
7. Team/mentors and references

### Repository structure
```text
Late Antiquity Modelling Project/
  data/
    raw/
    processed/
  task1-path-tracing/
    src/
    scripts/
    configs/
    tests/
    outputs/
    docs/
  task2-viewsheds/
    src/
    scripts/
    configs/
    tests/
    outputs/
    docs/
  shared/
    gis/
    io/
    metrics/
  reports/
  site/   # Jekyll pages/docs
```

### Execution structure (350 hours)
1. Hours 1-60: data audit, CRS harmonization, baseline preprocessing
2. Hours 61-160: Task 1 model + path inference + GIS exports
3. Hours 161-250: Task 2 2.5D/3D LOS + GIS exports
4. Hours 251-310: model calibration, acceptance checks, error analysis
5. Hours 311-350: documentation, reproducibility scripts, final packaging

## Progress Dashboard (Updated 2026-03-11 UTC)

| Task | Completion (/100) | Evidence |
|---|---:|---|
| Task 1: Path Tracing | **100** | `outputs/task1_completion.md` (Required: PASS, Required+Optional: PASS), `outputs/predicted_paths.gpkg`, `outputs/lost_path_candidates.geojson`, `outputs/probability_heatmap.tif` |
| Task 2: Viewsheds | **100** | `outputs/task2_completion.md` (Required: PASS, Required+Optional: PASS), `outputs/viewshed3d_volume.vtk`, `outputs/viewshed_model_metrics.json` |

### Scoring Basis
- Task 1 acceptance status: complete (`outputs/task1_completion.md` = Required PASS + Optional PASS)
- Task 1 key metrics: `topk_recall=0.5614`, `f1=0.3033`, `prior_iou=0.1091`
- Task 2 scoring rubric:
1. Required artifacts complete: 25/25
2. 3D label mode and voxel outputs complete: 25/25
3. Model acceptance thresholds pass (`F1_best=0.5549`, `IoU_best=0.3840`): 25/25
4. Optional 3D volume artifact present: 25/25

## Data and Repositories
- Task 1 implementation root: `Path Tracing Simulations/`
- Task 2 implementation root: `Visibilties/lamp-3d-viewshed/`

---
Built with GitHub Pages, Jekyll, and Bootstrap.
