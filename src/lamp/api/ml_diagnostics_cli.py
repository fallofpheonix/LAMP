from __future__ import annotations

import argparse
from pathlib import Path

from lamp.config import load_defaults
from lamp.services.ml_diagnostics_service import run_diagnostics


def run(argv: list[str] | None = None) -> int:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="Run ML diagnostics for Task 1 path prior features")
    parser.add_argument("--dem", default=str(defaults.dem_path))
    parser.add_argument("--sar", default=str(defaults.sar_path))
    parser.add_argument("--paths", default=str(defaults.train_paths))
    parser.add_argument("--eval-paths", default=str(defaults.eval_paths))
    parser.add_argument("--out-dir", default=str(defaults.diagnostics_output_dir))
    args = parser.parse_args(argv)

    run_diagnostics(
        dem_path=Path(args.dem),
        sar_path=Path(args.sar),
        train_paths_path=Path(args.paths),
        eval_paths_path=Path(args.eval_paths),
        out_dir=Path(args.out_dir),
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)
