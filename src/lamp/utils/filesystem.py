from __future__ import annotations

from pathlib import Path


def resolve_existing_path(path_value: str | Path) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def read_text_safely(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")
