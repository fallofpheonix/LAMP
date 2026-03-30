from __future__ import annotations

import argparse
import sys

from lamp.api.dataset_validation_cli import run as run_dataset_validation
from lamp.api.ml_diagnostics_cli import run as run_ml_diagnostics
from lamp.api.raycast_benchmark_cli import run as run_raycast_benchmark
from lamp.api.security_audit_cli import run as run_security_audit


COMMANDS = {
    "validate-dataset": run_dataset_validation,
    "security-audit": run_security_audit,
    "benchmark-raycast": run_raycast_benchmark,
    "ml-diagnostics": run_ml_diagnostics,
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="lamp", description="Repository operations CLI")
    parser.add_argument("command", nargs="?", choices=sorted(COMMANDS))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    parser = _build_parser()

    if not args or args[0] in {"-h", "--help"}:
        parser.print_help()
        return 0 if args else 1

    command = args[0]
    handler = COMMANDS.get(command)
    if handler is None:
        parser.print_help()
        return 1

    return handler(args[1:])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
