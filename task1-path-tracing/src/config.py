from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class PipelineConfig:
    dem_path: Path = Path("Task_1/DEM_Subset-Original.tif")
    sar_path: Path = Path("Task_1/SAR-MS.tif")
    marks_path: Path = Path("Task_1/Marks_Brief1.shp")
    buildings_path: Path = Path("Task_1/BuildingFootprints.shp")
    known_paths_path: Path | None = None
    path_prior_raster: Path | None = None
    path_prior_mode: Literal["learned", "deterministic"] = "learned"
    out_dir: Path = Path("outputs")
    samples_per_pair: int = 256
    max_pairs: int = 0  # 0 => all terminal pairs
    top_k_paths: int = 8
    noise_temperature: float = 0.08
    cost_w_slope: float = 0.50
    cost_w_roughness: float = 0.25
    cost_w_surface: float = 0.15
    cost_w_path_prior: float = 0.10
    calibrate_weights: bool = False
    calibration_samples: int = 64
    rng_seed: int = 7


DEFAULT_CONFIG = PipelineConfig()
