"""Co-occurrence pair score.

Thin wrapper around the retrieval-side output. Each candidate's pair_score
from `retrieve_by_cooccurrence` is already the per-disease signal — this
component simply normalizes and exposes it in the composite ranker.
"""
from __future__ import annotations

from typing import Any, Dict


def cooccurrence_score(
    disease_id: str,
    cooccurrence_candidates: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    info = cooccurrence_candidates.get(disease_id)
    if not info:
        return {"cooccurrence_pairs_score": 0.0, "matched_pairs": []}
    return {
        "cooccurrence_pairs_score": float(info.get("pair_score", 0.0)),
        "matched_pairs": info.get("matched_pairs", []),
    }
