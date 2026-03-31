from __future__ import annotations

import argparse
import sys


def _run_path_tracing(argv: list[str]) -> int:
    from lamp.tasks.path_tracing.pipeline import main as run_path_tracing

    return run_path_tracing(argv)


def _run_viewsheds_2d(argv: list[str]) -> int:
    from lamp.tasks.viewsheds.pipeline_2d import main as run_viewsheds_2d

    return run_viewsheds_2d(argv)


def _run_viewsheds_3d(argv: list[str]) -> int:
    from lamp.tasks.viewsheds.pipeline_3d import main as run_viewsheds_3d

    return run_viewsheds_3d(argv)


def _run_validate_dataset(argv: list[str]) -> int:
    from lamp.api.dataset_validation_cli import run as run_dataset_validation

    return run_dataset_validation(argv)


def _run_security_audit(argv: list[str]) -> int:
    from lamp.api.security_audit_cli import run as run_security_audit

    return run_security_audit(argv)


def _run_benchmark_raycast(argv: list[str]) -> int:
    from lamp.api.raycast_benchmark_cli import run as run_raycast_benchmark

    return run_raycast_benchmark(argv)


def _run_ml_diagnostics(argv: list[str]) -> int:
    from lamp.api.ml_diagnostics_cli import run as run_ml_diagnostics

    return run_ml_diagnostics(argv)


COMMANDS = {
    "path-tracing": _run_path_tracing,
    "viewsheds-2d": _run_viewsheds_2d,
    "viewsheds-3d": _run_viewsheds_3d,
    "validate-dataset": _run_validate_dataset,
    "security-audit": _run_security_audit,
    "benchmark-raycast": _run_benchmark_raycast,
    "ml-diagnostics": _run_ml_diagnostics,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(argv or [])
    if argv and argv[0] in COMMANDS:
        try:
            return COMMANDS[argv[0]](argv[1:])
        except ModuleNotFoundError as exc:
            if exc.name == "osgeo":
                print(
                    "viewshed pipelines require GDAL Python bindings (`osgeo`) in the active environment.",
                    file=sys.stderr,
                )
                return 2
            raise

    parser = argparse.ArgumentParser(prog="lamp", description="LAMP command line interface")
    parser.add_argument(
        "command",
        nargs="?",
        choices=sorted(COMMANDS),
        help="Subcommand to execute",
    )
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 1
    return COMMANDS[args.command]([])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
