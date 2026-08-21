"""Demographics score component.

Matches patient age / sex / ethnicity against a disease's demographic annotations
(if present in KG). Preserves the structure from the original rare_dx_mcp
demographics_score but adapted to accept an already-normalized ethnicity
lookup table and simpler KG access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class DemographicsConfig:
    w_sex: float = 0.4
    w_age: float = 0.3
    w_ethnicity: float = 0.3

    kg_high_multiplier: float = 1.0
    kg_moderate_multiplier: float = 0.5
    kg_low_multiplier: float = 0.1
    kg_unknown_multiplier: float = 0.5


def _norm(x: Any) -> str:
    return str(x).strip().lower() if x is not None else ""


def _kg_measure_multiplier(measure_type: Any, cfg: DemographicsConfig) -> float:
    s = _norm(measure_type)
    if "high" in s:
        return cfg.kg_high_multiplier
    if "moderate" in s:
        return cfg.kg_moderate_multiplier
    if "low" in s:
        return cfg.kg_low_multiplier
    return cfg.kg_unknown_multiplier


def _iter_field(kg_demo: Dict[str, Any], field: str) -> List[Dict[str, Any]]:
    v = kg_demo.get(field, [])
    if isinstance(v, list):
        return [x for x in v if isinstance(x, dict)]
    if isinstance(v, dict):
        return [v]
    return []


def normalize_sex(text: Any) -> str:
    s = _norm(text)
    if any(k in s for k in ["female", "woman", "women", "girl"]):
        return "female"
    if any(k in s for k in ["male", "man", "men", "boy"]):
        return "male"
    return "unknown"


def normalize_age_group(age_value: Any) -> str:
    s = _norm(age_value)
    if not s:
        return "unknown"
    if any(k in s for k in ["prenatal", "fetal", "congenital", "at birth"]):
        return "prenatal"
    if any(k in s for k in ["neonate", "neonatal", "newborn"]):
        return "neonatal"
    if "infant" in s:
        return "infantile"
    if "child" in s:
        return "childhood"
    if any(k in s for k in ["adolescent", "teen", "juvenile"]):
        return "adolescent"
    if "adult" in s:
        return "adult"
    if "elderly" in s:
        return "elderly"
    return "unknown"


def demographics_score(
    disease_id: str,
    patient_demographics: Dict[str, Any],
    kg: Dict[str, Dict[str, Any]],
    ethnicity_normalized: Dict[str, List[str]],
    cfg: DemographicsConfig | None = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = DemographicsConfig()

    entry = kg.get(disease_id, {})
    meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}
    kg_demo = entry.get("demographics") or meta.get("demographics") or {}
    if not isinstance(kg_demo, dict):
        kg_demo = {}

    def _pull_value(d: Dict[str, Any], key: str) -> Any:
        v = d.get(key)
        if isinstance(v, dict):
            return v.get("value")
        return v

    p_sex = normalize_sex(_pull_value(patient_demographics, "sex"))
    p_eth = _norm(_pull_value(patient_demographics, "ethnicity"))
    age_raw = _pull_value(patient_demographics, "age_group") or _pull_value(patient_demographics, "age")
    p_age = normalize_age_group(age_raw)

    matched_eth = set(ethnicity_normalized.get("similar", []))
    partial_eth = set(ethnicity_normalized.get("partial", []))

    score = 0.0
    hits = {"sex": [], "age_group": [], "ethnicity": []}

    # SEX
    if p_sex not in {"unknown", "none"}:
        for item in _iter_field(kg_demo, "sex"):
            kg_val = normalize_sex(item.get("sex"))
            if kg_val == p_sex:
                w = cfg.w_sex * _kg_measure_multiplier(item.get("measure_type"), cfg)
                score += w
                hits["sex"].append({"kg_value": item.get("sex"), "weight": w})
                break

    # AGE
    if p_age not in {"unknown", "none"}:
        for item in _iter_field(kg_demo, "age"):
            kg_val = normalize_age_group(item.get("age_group"))
            if kg_val == p_age:
                w = cfg.w_age * _kg_measure_multiplier(item.get("measure_type"), cfg)
                score += w
                hits["age_group"].append({"kg_value": item.get("age_group"), "weight": w})
                break

    # ETHNICITY
    if p_eth not in {"unknown", "none"}:
        for item in _iter_field(kg_demo, "ethnicity"):
            kg_val = item.get("ethnicity")
            if not kg_val or _norm(kg_val) in {"unknown", "none"}:
                continue
            if kg_val in matched_eth:
                w = cfg.w_ethnicity * _kg_measure_multiplier(item.get("measure_type"), cfg)
                score += w
                hits["ethnicity"].append({"kg_value": kg_val, "weight": w, "match": "similar"})
                break
            if kg_val in partial_eth:
                w = cfg.w_ethnicity * _kg_measure_multiplier(item.get("measure_type"), cfg) * 0.8
                score += w
                hits["ethnicity"].append({"kg_value": kg_val, "weight": w, "match": "partial"})
                break

    return {
        "demographics_score": score,
        "patient_norm": {"sex": p_sex, "age_group": p_age, "ethnicity": p_eth},
        "hits": hits,
    }
