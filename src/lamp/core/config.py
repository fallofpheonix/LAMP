from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback is covered instead.
    yaml = None

from lamp.core.shared_config import load_config as load_config_fallback


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"
REPOSITORY_ROOT = DEFAULT_CONFIG_PATH.parents[2]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load repository configuration from YAML with a source fallback.

    The repo previously relied on a removed ``lamp.core.config`` source file
    while only a bytecode artifact remained in local environments.  This source
    implementation restores a stable import target and keeps a minimal fallback
    parser when PyYAML is unavailable.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_absolute():
        config_path = REPOSITORY_ROOT / config_path

    if not config_path.exists():
        return {}

    if yaml is None:
        return load_config_fallback(config_path)

    with open(config_path, "r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Configuration root must be a mapping: {config_path}")
    return data
