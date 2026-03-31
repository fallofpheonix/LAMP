from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lamp.core.config import load_config


def _optional_path(value: str | None) -> Path | None:
    if value in (None, "", "null"):
        return None
    return Path(value)


@dataclass(frozen=True)
class PipelineConfig:
    # Defaults taken from centralized config
    _raw = load_config()
    dataset = _raw.get("dataset", {})
    ml = _raw.get("ml_training", {})
    sim = _raw.get("simulation", {})
    viewshed = _raw.get("viewshed", {})
    
    dem_path: Path = Path(dataset.get("dem_path", "data/task1/DEM_Subset-Original.tif"))
    sar_path: Path = Path(dataset.get("sar_path", "data/task1/SAR-MS.tif"))
    marks_path: Path = Path(dataset.get("marks_path", "data/task1/Marks_Brief1.shp"))
    buildings_path: Path = Path(dataset.get("buildings_path", "data/task1/BuildingFootprints.shp"))
    known_paths_path: Path | None = _optional_path(dataset.get("known_paths_train"))
    path_prior_raster: Path | None = Path("path_prior_prob.tif")
    path_prior_mode: Literal["learned", "deterministic"] = "learned"
    out_dir: Path = Path("outputs_production")
    samples_per_pair: int = int(sim.get("samples_per_pair", 128))
    max_pairs: int = 0
    top_k_paths: int = int(sim.get("top_k_paths", 12))
    noise_temperature: float = float(sim.get("noise_temperature", 0.08))
    cost_w_slope: float = float(sim.get("cost_w_slope", 0.55))
    cost_w_roughness: float = float(sim.get("cost_w_roughness", 0.30))
    cost_w_surface: float = float(sim.get("cost_w_surface", 0.10))
    cost_w_path_prior: float = float(sim.get("cost_w_path_prior", 0.05))
    cost_w_visibility: float = float(sim.get("cost_w_visibility", 0.0))
    calibrate_weights: bool = bool(sim.get("calibrate_weights", False))
    calibration_samples: int = 64
    compare_visibility_coupling: bool = bool(sim.get("compare_visibility_coupling", False))
    visibility_raster: Path | None = _optional_path(viewshed.get("visibility_raster"))
    visibility_source: Literal["deterministic", "model"] = str(viewshed.get("visibility_source", "deterministic"))  # type: ignore[assignment]
    rng_seed: int = int(ml.get("seed", 11))

DEFAULT_CONFIG = PipelineConfig()
