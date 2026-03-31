from __future__ import annotations

import argparse
from pathlib import Path

from lamp.services.ml_diagnostics_service import run_diagnostics


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ML diagnostics for Task 1 path prior features")
    parser.add_argument("--dem", default="data/task1/DEM_Subset-Original.tif")
    parser.add_argument("--sar", default="data/task1/SAR-MS.tif")
    parser.add_argument("--paths", required=True)
    parser.add_argument("--eval-paths", required=True)
    parser.add_argument("--out-dir", default="outputs/path_tracing/diagnostics")
    args = parser.parse_args(argv)

    run_diagnostics(
        dem_path=Path(args.dem),
        sar_path=Path(args.sar),
        train_paths_path=Path(args.paths),
        eval_paths_path=Path(args.eval_paths),
        out_dir=Path(args.out_dir),
    )
    return 0
