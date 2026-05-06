"""Config loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    text = config_path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - optional dependency
            raise ValueError(f"{config_path} is not valid JSON and PyYAML is unavailable.") from exc
        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise ValueError(f"{config_path} must contain a mapping/object.")
        return loaded


def project_path(raw_path: str | Path, *, root: str | Path | None = None) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (Path.cwd() if root is None else Path(root)) / path
