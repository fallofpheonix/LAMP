from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PathDefaults:
    dem_path: Path
    sar_path: Path
    marks_path: Path
    buildings_path: Path
    train_paths: Path
    eval_paths: Path
    diagnostics_output_dir: Path


def load_defaults(base_dir: str | Path = ".") -> PathDefaults:
    base = Path(base_dir).resolve()
    return PathDefaults(
        dem_path=base / os.getenv("LAMP_DEM_PATH", "task1-path-tracing/Task_1/DEM_Subset-Original.tif"),
        sar_path=base / os.getenv("LAMP_SAR_PATH", "task1-path-tracing/Task_1/SAR-MS.tif"),
        marks_path=base / os.getenv("LAMP_MARKS_PATH", "task1-path-tracing/Task_1/Marks_Brief1.shp"),
        buildings_path=base / os.getenv("LAMP_BUILDINGS_PATH", "task1-path-tracing/Task_1/BuildingFootprints.shp"),
        train_paths=base / os.getenv("LAMP_TRAIN_PATHS", "task1-path-tracing/known_paths_train.shp"),
        eval_paths=base / os.getenv("LAMP_EVAL_PATHS", "task1-path-tracing/known_paths_eval.shp"),
        diagnostics_output_dir=base / os.getenv("LAMP_DIAG_OUT", "outputs/diagnostics"),
    )
