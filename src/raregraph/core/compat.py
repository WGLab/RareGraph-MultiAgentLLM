"""Small compat helpers."""
from __future__ import annotations

from typing import Any


def to_dict(x: Any) -> Any:
    """Convert a Pydantic model or mapping-like object to a plain dict.

    Returns x unchanged if it's already a dict. For Pydantic v2 models, uses
    `model_dump()`. For v1-style models (with `dict()` attribute but no
    `model_dump`), falls back to `dict()`. For anything else, returns as-is.
    """
    if isinstance(x, dict):
        return x
    # Pydantic v2
    if hasattr(x, "model_dump"):
        try:
            return x.model_dump()
        except Exception:
            pass
    # Pydantic v1 fallback
    if hasattr(x, "dict") and callable(getattr(x, "dict")):
        try:
            return x.dict()
        except Exception:
            pass
    return x
