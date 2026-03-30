from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from lamp.core.config import REPOSITORY_ROOT, load_config


def _resolve_path(raw: str | None, default: str) -> Path:
    value = raw or default
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _resolve_optional_path(raw: str | None, default: str | None) -> Path | None:
    value = raw if raw is not None else default
    if value in (None, "", "null"):
        return None
    path = Path(value)
    return path if path.is_absolute() else REPOSITORY_ROOT / path


_raw_config = load_config()
_dataset = _raw_config.get("dataset", {})
_ml = _raw_config.get("ml_training", {})
_sim = _raw_config.get("simulation", {})
_weights = _sim.get("cost_weights", {})
_path_prior_mode = str(_dataset.get("path_prior_mode", "learned")).strip().lower()


@dataclass(frozen=True)
class PipelineConfig:
    dem_path: Path = _resolve_path(_dataset.get("dem_path"), "data/task1/DEM_Subset-Original.tif")
    sar_path: Path = _resolve_path(_dataset.get("sar_path"), "data/task1/SAR-MS.tif")
    marks_path: Path = _resolve_path(_dataset.get("marks_path"), "data/task1/Marks_Brief1.shp")
    buildings_path: Path = _resolve_path(_dataset.get("buildings_path"), "data/task1/BuildingFootprints.shp")
    known_paths_path: Path | None = _resolve_optional_path(
        _dataset.get("known_paths_train"),
        "data/task1/known_paths_train.shp",
    )
    path_prior_raster: Path | None = _resolve_optional_path(
        _dataset.get("path_prior_raster"),
        "data/task1/path_prior_prob.tif",
    )
    path_prior_mode: Literal["learned", "deterministic"] = (
        "deterministic"
        if _path_prior_mode == "deterministic"
        else "learned"
    )
    out_dir: Path = _resolve_path(
        str(_dataset.get("output_dir", "outputs_production")),
        "outputs_production",
    )
    samples_per_pair: int = int(_sim.get("samples_per_pair", 128))
    max_pairs: int = int(_sim.get("max_pairs", 0))
    top_k_paths: int = int(_sim.get("top_k_paths", 12))
    noise_temperature: float = float(_sim.get("noise_temperature", 0.08))
    cost_w_slope: float = float(_weights.get("slope", 0.55))
    cost_w_roughness: float = float(_weights.get("roughness", 0.30))
    cost_w_surface: float = float(_weights.get("surface", 0.10))
    cost_w_path_prior: float = float(_weights.get("path_prior", 0.05))
    calibrate_weights: bool = bool(_sim.get("calibrate_weights", True))
    calibration_samples: int = int(_sim.get("calibration_samples", 64))
    rng_seed: int = int(_ml.get("seed", 11))

DEFAULT_CONFIG = PipelineConfig()
