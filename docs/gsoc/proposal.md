# GSoC Proposal — LAMP Project

---

## Project Idea: LAMP (Late Antiquity Modeling Project)

### Abstract

The Late Antiquity Modeling Project (LAMP) provides deterministic pipelines for archaeological path inference and 3D visibility analysis at the El Bagawat necropolis. While both capabilities exist, the interaction between visibility and movement remains insufficiently validated. The repository already supports visibility-coupled path inference, but its impact on emergent movement patterns and agreement with archaeological evidence has not been systematically evaluated.

This project proposes to conduct a structured validation and calibration of visibility-coupled path inference within the existing LAMP framework. The work will reuse the deterministic multi-observer viewshed outputs and integrate them into the path cost model to compare terrain-only and visibility-aware movement scenarios. A controlled evaluation pipeline will be developed to measure how visibility influences path distributions using metrics such as IoU, F1 score, top-k recall, and path-density divergence. Additionally, the project will perform sensitivity analysis over the visibility weight parameter to quantify its effect on inferred pathways and identify stable operating regimes.

The expected outcome is a reproducible evaluation framework that establishes whether visibility acts as a meaningful constraint on movement in archaeological landscapes. Even in the absence of performance improvement, the project will deliver a validated coupling interface, calibrated parameter ranges, and comparative outputs that support further archaeological interpretation. This contribution strengthens LAMP by transforming an implemented feature into a rigorously tested and scientifically grounded component of the modeling pipeline.

---

## Problem Statement

### Existing LAMP Capabilities

The LAMP repository implements two core deterministic geospatial pipelines for the El Bagawat necropolis. **Task 1** performs probabilistic path tracing using a multi-factor cost surface derived from terrain slope, roughness, surface conditions, and path priors. It produces GIS-ready outputs such as cost rasters, path density maps, and vectorized predicted pathways. **Task 2** computes multi-observer visibility through line-of-sight analysis over terrain and building geometry, aggregating per-observer viewsheds into a continuous visibility probability raster.

These pipelines are modular and CLI-driven, with a shared configuration system and file-based artifact exchange. Task 2 exports a `viewshed_probability.tif` raster, and Task 1 supports an extended cost model that can incorporate visibility as an additional term. Comparison mode is also available, allowing baseline (terrain-only) and visibility-coupled runs to be generated within a single execution.

### Limitations of Current Pipeline

Despite the presence of visibility coupling in the codebase, its role within the modeling pipeline remains largely unvalidated. Current verification is limited to synthetic or smoke-test scenarios, and real end-to-end integration between Task 2 outputs and Task 1 inputs has not been fully executed due to environmental constraints (e.g., missing GDAL bindings).

Additionally, the repository lacks systematic evaluation using archaeological ground truth. Default configurations do not include known-path labels, preventing direct computation of metrics such as IoU, F1 score, and top-k recall on real datasets. As a result, the effect of visibility on inferred movement patterns has not been quantitatively established.

Calibration of the visibility weight is also underexplored. While constrained weight search exists, there is no structured sensitivity analysis to determine how variations in the visibility term influence path distributions or stability of results. Consequently, it is unclear whether visibility meaningfully contributes to path inference or introduces noise under certain parameter regimes.

### Identified Gap: Visibility–Movement Decoupling

The core gap in the current LAMP pipeline is not the absence of visibility coupling, but the absence of its rigorous validation and characterization. Visibility and movement are connected at the implementation level yet remain effectively decoupled at the level of scientific interpretation.

Specifically, the pipeline does not currently answer:

1. Whether visibility-aware cost functions produce pathways that better align with archaeological evidence.
2. How sensitive inferred paths are to the visibility weighting parameter.
3. Where and how visibility redistributes movement probability across the landscape.

Without these answers, visibility remains an unverified feature rather than a validated modeling constraint. This limits its usefulness for archaeological analysis and undermines the goal of reconstructing embodied spatial experience. This project addresses this gap by transforming visibility coupling from an implemented capability into a systematically evaluated and calibrated component of the LAMP modeling framework.

---

## System Design

### System Overview

The system extends the existing LAMP pipelines by introducing a structured evaluation and calibration layer over the already implemented visibility-coupled path inference. Rather than modifying core functionality, the design orchestrates Task 1 (path tracing) and Task 2 (viewshed computation) into a unified experimental workflow.

The system operates in two modes:

- **Baseline mode:** Path inference using terrain, surface, and prior-based cost only.
- **Coupled mode:** Identical pipeline with an additional visibility-derived penalty term.

Both modes are executed under identical conditions and compared using shared metrics and outputs, functioning as a controlled experimental framework.

### Core Components

The design builds on existing modules and introduces minimal extensions:

1. **Visibility Generator (Task 2):** Produces `viewshed_probability.tif` from multi-observer line-of-sight analysis.
2. **Raster Alignment Layer:** Ensures compatibility between Task 2 visibility raster and Task 1 DEM grid (CRS, transform, shape validation).
3. **Cost Surface Engine (Task 1 Extension):** Constructs movement cost using terrain features and optional visibility term.
4. **Path Simulation Engine:** Performs probabilistic path sampling using the constructed cost surface.
5. **Calibration Module:** Performs constrained search over visibility weight parameter.
6. **Comparison and Evaluation Layer:** Executes baseline and coupled scenarios and produces comparative metrics and density delta maps.

### Visibility-Coupled Cost Model

The system uses an extended deterministic cost formulation:

**C = ws S + wr R + wt T + wp (1 - P) + wv (1 - V)**

#### Model Parameters

- **S:** slope (normalized [0, 1])
- **R:** roughness (normalized [0, 1])
- **T:** surface penalty (normalized [0, 1])
- **P:** path prior probability [0, 1]
- **V:** visibility probability [0, 1]

All weights are normalized such that **Σw = 1**. Visibility acts as a penalty term (high visibility reduces cost). When **wv = 0**, the model reduces to the baseline pipeline.

---

## Implementation Plan

### Stepwise Development Plan

#### Step 1: Environment Stabilization and Baseline Verification

- Resolve environment dependencies (GDAL bindings) and ensure Task 1/2 run end-to-end.
- Execute baseline path tracing and verify outputs (path density, vectorized pathways).

#### Step 2: Task 2 Execution Enablement

- Execute `scripts/run_viewsheds.py` on the real El Bagawat dataset to generate `viewshed_probability.tif`.

#### Step 3: Raster Alignment and Coupling Verification

- Load Task 1 DEM as reference and align visibility raster (CRS, transform, dimensions).
- Perform initial coupling test with real data.

#### Step 4: Baseline vs. Coupled Execution

- Run side-by-side comparison runs using `--compare-visibility-coupling`.
- Collect difference delta maps and summary metrics.

#### Step 5: Calibration and Sensitivity Analysis

- Controlled variation of wv across a defined search space (e.g., [0.0, 0.4]).
- Identify stable operating regimes and quantify the shift in path distributions.

#### Step 6: Evaluation with Available Metrics

- Compute IoU, F1 score, and recall against archaeological ground truth (if available).
- Otherwise, compute path-density divergence and redistribution patterns.

#### Step 7: Artifact Generation and Reporting

- Generate comparison figures, metric tables, and GIS layers for the final submission.

---

## Timeline (350-hour structure)

- **Weeks 0–1 (Bonding):** Environment setup, resolving GDAL issues, and aligning evaluation strategy with mentors.
- **Weeks 2–6 (Phase I: Integration):** Running real-data baselines, generating visibility rasters, and implementing the alignment layer for end-to-end coupling.
- **Midterm Checkpoint:** End-to-end pipeline working with real visibility data.
- **Weeks 7–11 (Phase II: Analysis):** Automated calibration loops, weight sensitivity analysis, and metric computation.
- **Week 12 (Final):** Final documentation, reproducibility guide, and preparing the final submission report.

---

## Deliverables

- **Core:** End-to-end pipeline execution from Task 2 to Task 1 with real data.
- **Analysis:** Sensitivity analysis report on the impact of visibility on archaeological movement.
- **GIS Artifacts:** Path density difference maps and calibrated vector pathways.
- **Documentation:** CLI usage guide and experimental reproducibility report.

---

## References

1. Pérez-García et al. (2021). Modelling archaeological works in Qubbet el-Hawa.
2. Wróżyński et al. (2024). 3D visibility analysis workflows in GIS.
3. Kiourt et al. (2026). Agent-based and terrain-aware mobility modeling.
4. Jaturapitpornchai et al. (2024). Model sensitivity in archaeological reconstruction.
