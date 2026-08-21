"""Small shared helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def read_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def write_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
