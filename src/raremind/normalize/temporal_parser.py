"""Parse onset strings to approximate age-in-months.

Turns phrases like "at birth", "infancy", "6 months", "since age 2", "childhood"
into a comparable numeric age (months from conception). Used for temporal view
and temporal lens reasoning.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Symbolic anchors
ANCHORS = {
    "prenatal": -3,
    "fetal": -3,
    "at conception": -9,
    "in utero": -3,
    "at birth": 0,
    "birth": 0,
    "neonatal": 0,
    "neonate": 0,
    "newborn": 0,
    "perinatal": 0,
    "infancy": 6,
    "infantile": 6,
    "infant": 6,
    "early childhood": 24,
    "toddler": 24,
    "childhood": 60,
    "school age": 84,
    "adolescence": 144,
    "adolescent": 144,
    "teenage": 156,
    "young adult": 228,
    "adulthood": 216,
    "adult": 216,
    "middle age": 480,
    "elderly": 720,
    "late adulthood": 720,
    "congenital": 0,
    "juvenile": 120,
}

# Numeric patterns
_NUM = r"(\d+(?:\.\d+)?)"

_YEAR_RE = re.compile(rf"{_NUM}\s*(?:year|yr|yrs|y/o|years old)", flags=re.IGNORECASE)
_MONTH_RE = re.compile(rf"{_NUM}\s*(?:month|mo|months)", flags=re.IGNORECASE)
_WEEK_RE = re.compile(rf"{_NUM}\s*(?:week|wk|weeks)", flags=re.IGNORECASE)
_DAY_RE = re.compile(rf"{_NUM}\s*(?:day|days)", flags=re.IGNORECASE)

# "age 2" / "at 2" / "when 2"
_BARE_AGE_RE = re.compile(
    rf"(?:age|at|when)\s*{_NUM}\b",
    flags=re.IGNORECASE,
)


def parse_onset_to_months(text: Any) -> Optional[float]:
    """Parse a free-text onset string to approximate age-in-months from birth.

    Negative values mean prenatal.
    Returns None if nothing can be parsed.
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s or s in {"unknown", "none", "null"}:
        return None

    # Anchor check first (longer phrases before shorter ones)
    for phrase in sorted(ANCHORS.keys(), key=len, reverse=True):
        if phrase in s:
            return float(ANCHORS[phrase])

    # Numeric patterns (most specific unit first)
    m = _YEAR_RE.search(s)
    if m:
        return float(m.group(1)) * 12.0

    m = _MONTH_RE.search(s)
    if m:
        return float(m.group(1))

    m = _WEEK_RE.search(s)
    if m:
        return float(m.group(1)) / 4.0

    m = _DAY_RE.search(s)
    if m:
        return float(m.group(1)) / 30.0

    m = _BARE_AGE_RE.search(s)
    if m:
        return float(m.group(1)) * 12.0

    return None


def build_temporal_view(phenotypes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build TemporalView data from a list of phenotype dicts.

    Each phenotype may have an 'onset' string field and a 'mention' / 'hpo_name'.
    """
    timeline = []
    for p in phenotypes:
        onset_text = p.get("onset")
        onset_months = parse_onset_to_months(onset_text) if onset_text else None
        timeline.append({
            "mention": p.get("mention") or p.get("hpo_name") or p.get("hpo_id", ""),
            "hpo_id": p.get("hpo_id") or p.get("hpo", ""),
            "onset_text": onset_text,
            "onset_months": onset_months,
        })

    with_onset = [t for t in timeline if t["onset_months"] is not None]
    with_onset.sort(key=lambda t: t["onset_months"])

    earliest_features = [t["mention"] for t in with_onset[:3]]

    return {
        "earliest_features": earliest_features,
        "onset_ordering": with_onset,
        "age_at_presentation": None,  # filled by caller if demographics known
    }
