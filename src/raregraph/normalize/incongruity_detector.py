"""Incongruity detection.

For each patient, compute:
  1. Per-branch IC-weighted signal
  2. Dominant organ-system branch (max weighted sum)
  3. Outlier phenotypes (those outside the dominant branch)
  4. Overall incongruity strength: strong | moderate | weak | none

This is a RareGraph-specific signal used as:
  - A retrieval/scoring component (incongruity_match_score)
  - A trigger condition for the frontier consultation call
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List

from .hpo_ontology import HpoOntology, TOP_LEVEL_BRANCHES


def compute_patient_branch_profile(
    hpos: List[str],
    hpo: HpoOntology,
) -> Dict[str, float]:
    """IC-weighted branch profile of the patient's phenotypes."""
    profile: Dict[str, float] = defaultdict(float)
    for hid in hpos:
        ic = hpo.get_ic(hid)
        for b in hpo.get_branches(hid):
            profile[b] += ic
    return dict(profile)


def detect_incongruity(
    present_hpos: List[Dict[str, Any]],
    hpo: HpoOntology,
) -> Dict[str, Any]:
    """Detect incongruity from a list of {hpo_id, mention} dicts."""
    hpo_ids = [h.get("hpo_id") for h in present_hpos if h.get("hpo_id")]
    if not hpo_ids:
        return {
            "dominant_branch": None,
            "dominant_branch_name": None,
            "branch_profile": {},
            "incongruous_phenotypes": [],
            "overall_incongruity_strength": "none",
        }

    profile = compute_patient_branch_profile(hpo_ids, hpo)
    if not profile:
        return {
            "dominant_branch": None,
            "dominant_branch_name": None,
            "branch_profile": {},
            "incongruous_phenotypes": [],
            "overall_incongruity_strength": "none",
        }

    dominant = max(profile, key=profile.get)
    dominant_ic = profile[dominant]

    incongruous = []
    for h in present_hpos:
        hid = h.get("hpo_id")
        if not hid:
            continue
        branches = hpo.get_branches(hid)
        if dominant in branches:
            continue  # belongs to dominant pattern
        ic = hpo.get_ic(hid)
        strength = _classify_strength(ic, hpo)
        incongruous.append({
            "mention": h.get("mention") or h.get("hpo_name") or hpo.get_name(hid),
            "hpo_id": hid,
            "branch": list(branches)[0] if branches else None,
            "branch_name": (hpo.get_branch_names(hid) or [""])[0],
            "ic": ic,
            "strength": strength,
            "incongruity_ratio": ic / dominant_ic if dominant_ic > 0 else 0.0,
        })
    incongruous.sort(key=lambda x: x["ic"], reverse=True)

    overall = _overall_strength(incongruous, profile, dominant_ic)

    return {
        "dominant_branch": dominant,
        "dominant_branch_name": TOP_LEVEL_BRANCHES.get(dominant, dominant),
        "branch_profile": profile,
        "incongruous_phenotypes": incongruous,
        "overall_incongruity_strength": overall,
    }


def _classify_strength(ic: float, hpo: HpoOntology) -> str:
    """Classify a single incongruous phenotype's strength by IC."""
    if ic >= hpo.ic_p75:
        return "strong"
    if ic >= hpo.ic_median:
        return "moderate"
    if ic >= hpo.ic_p25:
        return "weak"
    return "trace"


def _overall_strength(
    incongruous: List[Dict[str, Any]],
    profile: Dict[str, float],
    dominant_ic: float,
) -> str:
    """Aggregate incongruity strength at the patient level."""
    if not incongruous:
        return "none"
    # Any strong outlier → strong incongruity
    if any(x["strength"] == "strong" for x in incongruous):
        return "strong"
    # Any moderate outlier → moderate
    if any(x["strength"] == "moderate" for x in incongruous):
        return "moderate"
    # Non-trivial proportion of signal outside dominant branch
    other = sum(v for k, v in profile.items() if v != dominant_ic)
    if dominant_ic > 0 and (other / (other + dominant_ic)) > 0.2:
        return "weak"
    return "none"
