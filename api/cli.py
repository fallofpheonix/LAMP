from __future__ import annotations

import argparse
import sys

from api.dataset_validation_cli import run as run_dataset_validation
from api.ml_diagnostics_cli import run as run_ml_diagnostics
from api.raycast_benchmark_cli import run as run_raycast_benchmark
from api.security_audit_cli import run as run_security_audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lamp-tools", description="Repository operations CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("validate-dataset")
    subparsers.add_parser("security-audit")
    subparsers.add_parser("benchmark-raycast")
    subparsers.add_parser("ml-diagnostics")

    args, extra = parser.parse_known_args(argv)

    if args.command == "validate-dataset":
        return run_dataset_validation(extra)
    if args.command == "security-audit":
        return run_security_audit(extra)
    if args.command == "benchmark-raycast":
        return run_raycast_benchmark(extra)
    if args.command == "ml-diagnostics":
        return run_ml_diagnostics(extra)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
