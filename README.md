# LAMP: Late Antiquity Modeling Project

[![CI](https://github.com/fallofpheonix/LAMP/actions/workflows/ci.yml/badge.badge.svg)](https://github.com/fallofpheonix/LAMP/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Deterministic geospatial pipelines for archaeological path tracing, 3D viewshed analysis, and visibility-coupled movement inference.**

---

## 1. Project Overview

The **Late Antiquity Modeling Project (LAMP)** is an interdisciplinary research initiative applying computational methods to reconstruct spatial experiences in ancient landscapes. 

The project focuses on the **El Bagawat necropolis** in Egypt's Kharga Oasis. By integrating 3D visibility analysis with probabilistic path tracing, LAMP provides a rigorous framework for validating archaeological hypotheses against spatially grounded evidence.

### Core Capabilities
*   **Probabilistic Path Tracing:** Inferred movement pathways between ancient structures using multi-factor cost surfaces.
*   **3D Viewshed Analysis:** High-fidelity visibility computation considering terrain and building geometry.
*   **Visibility-Coupled Inference:** Integration of visibility as a measurable constraint in movement modeling.
*   **Operational Diagnostics:** Built-in tools for dataset validation, security auditing, and performance benchmarking.

---

## 2. Repository Structure

The repository is organized into a modular, production-ready layout:

```text
/
├── assets/             # Visualization samples for documentation
├── data/               # GIS sample datasets (DEMs, Shapefiles, Orthoimagery)
├── docs/               # Research documentation and GSoC proposals
├── scripts/            # Pipeline execution and analysis scripts
├── src/lamp/           # Core implementation modules
│   ├── api/            # CLI and service orchestration
│   ├── core/           # Mathematical models and IO logic
│   ├── services/       # Domain services (validation, diagnostics)
│   ├── tasks/          # Task-specific pipeline logic (Path Tracing, Viewsheds)
│   └── utils/          # Filesystem and utility helpers
├── tests/              # Unit and integration test suite
├── Dockerfile          # Containerized execution environment
└── pyproject.toml      # Project metadata and dependencies
```

---

## 3. Getting Started

### Prerequisites
*   Python 3.10+
*   GDAL/OGR system dependencies (for GIS operations)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fallofpheonix/LAMP.git
    cd LAMP
    ```

2.  **Install dependencies:**
    ```bash
    pip install .
    ```

3.  **Environment Setup:**
    Copy the example environment file and adjust paths if necessary:
    ```bash
    cp .env.example .env
    ```

---

## 4. Usage

The primary interface is the unified `lamp` CLI.

### Core Operations
*   **Validate Dataset:**
    ```bash
    lamp validate-dataset
    ```
*   **Security Audit:**
    ```bash
    lamp security-audit
    ```
*   **Performance Benchmark:**
    ```bash
    lamp benchmark-raycast
    ```

### Running the Pipeline
Execute individual task scripts for path tracing or viewshed analysis:

```bash
# Path Tracing
python scripts/run_path_tracing.py

# Viewshed Computation
python scripts/run_viewsheds.py
```

---

## 5. Visualizations

### 3D Visibility Scene
![3D Visibility](/assets/3D_visibility_scene.png)
*Reconstruction of visibility volumes in the El Bagawat necropolis.*

### Path Density Comparison
![Path Comparison](/assets/2D_viewshed_overlay.png)
*Comparison of movement patterns with and without visibility coupling.*

---

## 6. Testing

Run the full test suite using `pytest`:

```bash
pytest tests/ -v
```

---

## 7. License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 8. Acknowledgments

LAMP is a collaborative effort involving archaeologists, historians, and computer scientists dedicated to digital heritage and spatial modeling.
