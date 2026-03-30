from __future__ import annotations

import argparse
from pathlib import Path

from lamp.services.raycast_benchmark_service import render_benchmark_report, run_raycast_benchmark


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark raycast implementations")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--out", default="RAYCASTING_BENCHMARK.md")
    args = parser.parse_args(argv)

    result = run_raycast_benchmark(samples=args.samples)
    report = render_benchmark_report(result)
    Path(args.out).write_text(report, encoding="utf-8")
    print(report)
    return 0
