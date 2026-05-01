"""Inheritance pattern inference.

From family-history structured output, derive a prior distribution over
inheritance modes. The prior is then used in inheritance_score to modulate
per-candidate disease inheritance matches.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


# Default distribution when NO family history is available
NULL_PRIOR = {"AD": 0.0, "AR": 0.0, "XLR": 0.0, "XLD": 0.0, "Mito": 0.0}


def _contains_any(haystack: str, needles: List[str]) -> bool:
    s = haystack.lower()
    return any(n in s for n in needles)


def _relation_has(relation: str, terms: List[str]) -> bool:
    cleaned = re.sub(r"[^a-z\s]+", " ", str(relation or "").lower())
    tokens = set(cleaned.split())
    return any(term in tokens for term in terms)


def infer_inheritance_prior(family_history: List[Dict[str, Any]]) -> Dict[str, float]:
    """Infer an inheritance-mode prior from parsed family history entries.

    Returns an all-zero prior when no family info is available (so the
    downstream scorer gives this component zero weight).
    """
    if not family_history:
        return dict(NULL_PRIOR)

    # Consanguinity may be noted in an unaffected entry — still informative.
    has_consanguinity = any(
        _contains_any(f.get("evidence", ""), ["consang", "related", "cousin marr"])
        for f in family_history
    )

    affected = [f for f in family_history if f.get("affected", True)]
    if not affected and not has_consanguinity:
        return dict(NULL_PRIOR)

    # Simple signal features
    has_parents_affected = any(
        _relation_has(f.get("relation", ""), ["mother", "father", "parent"])
        for f in affected
    )
    has_siblings_affected = any(
        _relation_has(f.get("relation", ""), ["brother", "sister", "sibling"])
        for f in affected
    )
    has_maternal_male_pattern = any(
        _relation_has(f.get("relation", ""), ["uncle", "grandfather", "brother"])
        and _contains_any(f.get("evidence", ""), ["maternal"])
        for f in family_history
    )

    # Rule-based prior (order matters — first strong signal wins)
    if has_consanguinity:
        return {"AD": 0.05, "AR": 0.80, "XLR": 0.05, "XLD": 0.05, "Mito": 0.05}
    if has_parents_affected:
        return {"AD": 0.70, "AR": 0.05, "XLR": 0.05, "XLD": 0.10, "Mito": 0.10}
    if has_maternal_male_pattern:
        return {"AD": 0.05, "AR": 0.05, "XLR": 0.70, "XLD": 0.10, "Mito": 0.10}
    if has_siblings_affected and not has_parents_affected:
        return {"AD": 0.05, "AR": 0.70, "XLR": 0.10, "XLD": 0.05, "Mito": 0.10}

    # Only proband affected (or equivocal): uninformative
    return {"AD": 0.30, "AR": 0.30, "XLR": 0.10, "XLD": 0.10, "Mito": 0.20}


def inheritance_compatibility_flag(
    prior: Dict[str, float],
    disease_modes: List[str],
) -> Tuple[str, float]:
    """Compatibility of a disease's inheritance modes with the patient prior.

    Returns (flag, numeric_weight) where flag ∈
    {"COMPATIBLE", "PARTIAL_MATCH", "WEAK_MATCH", "UNKNOWN"}.
    """
    if not prior or sum(prior.values()) == 0:
        return ("UNKNOWN", 0.0)
    if not disease_modes:
        return ("UNKNOWN", 0.0)

    # Normalize mode strings ("autosomal dominant" → AD, etc.)
    disease_keys = []
    for m in disease_modes:
        s = re.sub(r"[^a-z0-9]+", " ", (m or "").lower()).strip()
        compact = s.replace(" ", "")
        if compact in {"ad", "autdom"}:
            disease_keys.append("AD")
        elif compact in {"ar", "autrec"}:
            disease_keys.append("AR")
        elif compact in {"xlr", "xlinkedrecessive"}:
            disease_keys.append("XLR")
        elif compact in {"xld", "xlinkeddominant"}:
            disease_keys.append("XLD")
        elif "dominant" in s and "auto" in s:
            disease_keys.append("AD")
        elif "recess" in s and "auto" in s:
            disease_keys.append("AR")
        elif "x linked" in s and "recess" in s:
            disease_keys.append("XLR")
        elif "x linked" in s and "domin" in s:
            disease_keys.append("XLD")
        elif "mito" in s:
            disease_keys.append("Mito")
        elif "x linked" in s:
            disease_keys.append("XLR")
        elif "dominant" in s:
            disease_keys.append("AD")
        elif "recess" in s:
            disease_keys.append("AR")

    if not disease_keys:
        return ("UNKNOWN", 0.0)

    # Max probability over disease_keys
    p = max(prior.get(k, 0.0) for k in disease_keys)
    if p >= 0.6:
        return ("COMPATIBLE", p)
    if p >= 0.3:
        return ("PARTIAL_MATCH", p)
    if p > 0:
        return ("WEAK_MATCH", p)
    return ("UNKNOWN", 0.0)
