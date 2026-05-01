"""Configuration loader.

Loads YAML config into a nested attribute-accessible object. Kept compatible
with the existing rare_dx_mcp style (cfg.models.text_llm.model_name etc.).
"""
from __future__ import annotations

import yaml
from pathlib import Path
from typing import Any, Dict


class AttrDict(dict):
    """Dict that supports attribute access and recursive conversion."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v in self.items():
            self[k] = self._wrap(v)

    @classmethod
    def _wrap(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return cls(v)
        if isinstance(v, list):
            return [cls._wrap(x) for x in v]
        return v

    def __getattr__(self, key: str) -> Any:
        if key in self:
            return self[key]
        raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in self.items():
            if isinstance(v, AttrDict):
                out[k] = v.to_dict()
            elif isinstance(v, list):
                out[k] = [x.to_dict() if isinstance(x, AttrDict) else x for x in v]
            else:
                out[k] = v
        return out


# Alias for external readability.
AppConfig = AttrDict


def cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    """Read a config key from either an AttrDict/object or a plain dict."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def retrieval_initial_top_k(cfg: Any) -> int:
    """Initial deterministic retrieval pool size.

    Accepts both the clean key (retrieval.initial_top_k) and older aliases so
    existing external configs do not break.
    """
    retrieval = cfg_get(cfg, "retrieval", {})
    ranking = cfg_get(cfg, "ranking", {})
    return int(
        cfg_get(
            retrieval,
            "initial_top_k",
            cfg_get(retrieval, "top_k", cfg_get(ranking, "top_k", 2000)),
        )
    )


def retrieval_retain_top_k(cfg: Any) -> int:
    """Number of Stage 3 candidates retained for downstream processing."""
    retrieval = cfg_get(cfg, "retrieval", {})
    ranking = cfg_get(cfg, "ranking", {})
    return int(
        cfg_get(
            retrieval,
            "retain_top_k",
            cfg_get(retrieval, "rerank_top_k", cfg_get(ranking, "rerank_top_k", 300)),
        )
    )


def audit_top_n_candidates(cfg: Any) -> int:
    """Number of post-Stage-3 candidates sent to the audit stage."""
    audit = cfg_get(cfg, "audit", {})
    frontier = cfg_get(cfg, "frontier", {})
    return int(
        cfg_get(
            audit,
            "top_n_candidates",
            cfg_get(frontier, "top_n_candidates", retrieval_retain_top_k(cfg)),
        )
    )


def load_config(path: str | Path) -> AttrDict:
    """Load a YAML config file and return an attribute-accessible AttrDict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping at top level: {path}")

    return AttrDict(data)


def save_config(cfg: AttrDict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.dump(cfg.to_dict() if isinstance(cfg, AttrDict) else cfg, f, sort_keys=False)
