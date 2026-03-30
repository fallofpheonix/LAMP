from lamp.core.config import load_config
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PipelineConfig:
    # Defaults taken from centralized config
    _raw = load_config()
    dataset = _raw.get("dataset", {})
    ml = _raw.get("ml_training", {})
    sim = _raw.get("simulation", {})
    
    dem_path: Path = Path(dataset.get("dem_path", "Task_1/DEM_Subset-Original.tif"))
    sar_path: Path = Path(dataset.get("sar_path", "Task_1/SAR-MS.tif"))
    marks_path: Path = Path(dataset.get("marks_path", "Task_1/Marks_Brief1.shp"))
    buildings_path: Path = Path(dataset.get("buildings_path", "Task_1/BuildingFootprints.shp"))
    known_paths_path: Path | None = Path(dataset.get("known_paths_train", "known_path_fragments.shp"))
    path_prior_raster: Path | None = Path("path_prior_prob.tif")
    path_prior_mode: Literal["learned", "deterministic"] = "learned"
    out_dir: Path = Path("outputs_production")
    samples_per_pair: int = int(sim.get("samples_per_pair", 128))
    max_pairs: int = 0
    top_k_paths: int = int(sim.get("top_k_paths", 12))
    noise_temperature: float = float(sim.get("noise_temperature", 0.08))
    cost_w_slope: float = 0.55
    cost_w_roughness: float = 0.30
    cost_w_surface: float = 0.10
    cost_w_path_prior: float = 0.05
    calibrate_weights: bool = True
    calibration_samples: int = 64
    rng_seed: int = int(ml.get("seed", 11))

DEFAULT_CONFIG = PipelineConfig()
