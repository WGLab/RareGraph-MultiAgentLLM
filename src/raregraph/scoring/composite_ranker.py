"""Composite ranker: Stage 3 of RareGraph.

Combines 8 scoring components with adaptive, evidence-dependent weights:
  1. Phenotype (IC-weighted + freq-aware negation + competitive IC)
  2. Genotype (Bayesian log-LR)
  3. Inheritance (prior × compatibility)
  4. Demographics (age/sex/ethnicity)
  5. Cases (PubCaseFinder)
  6. Specific signal (high-IC hallmark match bonus)
  7. Incongruity match (bridges dominant + outlier branches)
  8. Co-occurrence pairs (rare pair matches)

Normalization strategy:
  - Heavy-tailed components (phenotype, cases) → log1p + min-max rescaling
  - Other components → min-max with 0.5×weight tie floor

Output: pd.DataFrame with all per-component scores + total + rank.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from raregraph.core.compat import to_dict
from raregraph.core.config import retrieval_retain_top_k
from raregraph.kg.kg_precompute import KGIndex
from raregraph.normalize.hpo_ontology import HpoOntology

from .phenotype_score import phenotype_score, PhenotypeScoreConfig
from .specific_signal_score import specific_signal_score
from .incongruity_match_score import incongruity_match_score
from .cooccurrence_score import cooccurrence_score
from .gene_variant_score import genotype_score, GenotypeConfig
from .family_evidence_score import prepare_family_evidence, family_evidence_score
from .demographics_score import demographics_score, DemographicsConfig

logger = logging.getLogger(__name__)


def _min_max_rescale(values: np.ndarray, tie_floor: float = 0.5) -> np.ndarray:
    if len(values) == 0:
        return values
    lo = float(values.min())
    hi = float(values.max())
    if hi <= lo:
        return np.full_like(values, fill_value=tie_floor, dtype=np.float32)
    return (values - lo) / (hi - lo)


def _log_min_max_rescale(values: np.ndarray, tie_floor: float = 0.0) -> np.ndarray:
    """Compress heavy-tailed positive scores without clipping away rank signal."""
    if len(values) == 0:
        return values
    clipped = np.clip(values.astype(float), 0.0, None)
    logged = np.log1p(clipped)
    return _min_max_rescale(logged, tie_floor=tie_floor)


FAMILY_EVIDENCE_SHORTLIST_FLOOR = 500


def _adaptive_weights(
    base_weights: Dict[str, float],
    has_genes: bool,
    has_family: bool,
    has_demographics: bool,
    has_cases: bool,
    n_patient_hpos: int,
    has_vcf_positive: bool = False,
) -> Dict[str, float]:
    """Zero out components with no input data; scale by evidence availability."""
    w = dict(base_weights)

    # Phenotype weight scales with number of patient HPOs
    if n_patient_hpos >= 5:
        w["phenotype"] = base_weights.get("phenotype", 4.0)
    elif n_patient_hpos >= 3:
        w["phenotype"] = base_weights.get("phenotype", 4.0) * 0.8
    elif n_patient_hpos >= 1:
        w["phenotype"] = base_weights.get("phenotype", 4.0) * 0.6
    else:
        w["phenotype"] = 0.0

    # Genotype
    if has_vcf_positive:
        w["genotype"] = base_weights.get("genotype", 3.0)
    elif has_genes:
        w["genotype"] = base_weights.get("genotype", 3.0) * 0.5
    else:
        w["genotype"] = 0.0

    # Inheritance — zero if no family history at all
    family_weight = base_weights.get("family_evidence", base_weights.get("inheritance", 1.0))
    w["family_evidence"] = family_weight if has_family else 0.0
    w.pop("inheritance", None)

    # Demographics — zero if nothing known
    w["demographics"] = base_weights.get("demographics", 1.0) if has_demographics else 0.0

    # Cases — zero if disabled / unavailable
    w["cases"] = base_weights.get("cases", 1.0) if has_cases else 0.0

    # Specific signal / incongruity / co-occurrence: always on when data exists
    w["specific_signal"] = base_weights.get("specific_signal", 2.0) if n_patient_hpos > 0 else 0.0
    w["incongruity_match"] = base_weights.get("incongruity_match", 2.0) if n_patient_hpos > 0 else 0.0
    w["cooccurrence_pairs"] = base_weights.get("cooccurrence_pairs", 1.5) if n_patient_hpos >= 2 else 0.0

    return w


def score_candidates(
    candidate_ids: List[str],
    patient_state: Any,  # PatientCaseState
    kg: Dict[str, Dict[str, Any]],
    kg_index: KGIndex,
    hpo: HpoOntology,
    cfg: Any,
    hpo_normalizer: Any | None = None,
    ethnicity_normalized: Optional[Dict[str, List[str]]] = None,
    cooccurrence_candidates: Optional[Dict[str, Dict[str, Any]]] = None,
    cases_scores: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Score all candidates, return DataFrame sorted by total_score descending."""

    ethnicity_normalized = ethnicity_normalized or {}
    cooccurrence_candidates = cooccurrence_candidates or {}
    cases_scores = cases_scores or {}

    all_phens = [to_dict(h) for h in patient_state.normalized_hpo]
    present_hpos = [h for h in all_phens if h.get("present", True)]
    negated_hpos = [h for h in all_phens if not h.get("present", True)]
    gene_mentions = patient_state.gene_mentions or []
    vcf_summary = patient_state.vcf_summary or []
    demographics = to_dict(patient_state.demographics) if patient_state.demographics else {}
    if not isinstance(demographics, dict):
        demographics = {}
    family_history = patient_state.family_history or []
    inheritance_prior = patient_state.inheritance_prior or {}
    incongruity = to_dict(patient_state.incongruity) if patient_state.incongruity else {}
    if not isinstance(incongruity, dict):
        incongruity = {}

    total_candidates = len(candidate_ids)

    # Adaptive weights
    base_weights = cfg.scoring.weights
    if hasattr(base_weights, "to_dict"):
        base_weights = base_weights.to_dict()
    else:
        base_weights = dict(base_weights)

    has_genes = bool(gene_mentions) or bool(vcf_summary)
    has_vcf_positive = any(str(g.get("result", "")).lower().find("pos") != -1 for g in vcf_summary)
    family_mentions = [
        to_dict(m)
        for m in (patient_state.phenotype_mentions_text or [])
        if to_dict(m).get("attribution") == "family"
    ]
    has_family = bool(family_history or family_mentions)
    has_demographics = any([
        demographics.get("sex", {}).get("value") if isinstance(demographics.get("sex"), dict) else demographics.get("sex"),
        demographics.get("age", {}).get("value") if isinstance(demographics.get("age"), dict) else demographics.get("age"),
        demographics.get("ethnicity", {}).get("value") if isinstance(demographics.get("ethnicity"), dict) else demographics.get("ethnicity"),
    ])
    has_cases = bool(cases_scores)

    if cfg.scoring.use_adaptive_weights:
        weights = _adaptive_weights(
            base_weights,
            has_genes=has_genes,
            has_family=has_family,
            has_demographics=has_demographics,
            has_cases=has_cases,
            n_patient_hpos=len(present_hpos),
            has_vcf_positive=has_vcf_positive,
        )
    else:
        weights = base_weights

    logger.info(f"Adaptive weights: {weights}")

    ic_high = cfg.scoring.ic_high_threshold
    if ic_high is None:
        ic_high = hpo.ic_p75

    # Compute raw per-component scores for every candidate
    rows: List[Dict[str, Any]] = []
    for did in candidate_ids:
        name = kg_index.disease_name.get(did, did)
        group_id = kg_index.disease_group.get(did, "") or did
        group_name = kg_index.disease_name.get(group_id, group_id)

        pheno = phenotype_score(
            did, present_hpos, negated_hpos, kg_index, kg, hpo, total_candidates, hpo_normalizer=hpo_normalizer
        )
        specific = specific_signal_score(did, present_hpos, kg_index, kg, hpo, ic_high) if cfg.scoring.use_specific_signal else {"specific_signal_score": 0.0}
        incong = incongruity_match_score(did, incongruity, kg_index, hpo) if cfg.scoring.use_incongruity_match else {"incongruity_match_score": 0.0}
        cooc = cooccurrence_score(did, cooccurrence_candidates) if cfg.scoring.use_cooccurrence_pairs else {"cooccurrence_pairs_score": 0.0}
        geno = genotype_score(did, kg, gene_mentions, vcf_summary)
        demo = demographics_score(did, demographics, kg, ethnicity_normalized) if has_demographics else {"demographics_score": 0.0}
        cases = cases_scores.get(did, 0.0)

        rows.append({
            "disease_id": did,
            "disease_name": name,
            "group_id": group_id,
            "group_name": group_name,
            "raw_phenotype_score": pheno["phenotype_score"],
            "phenotype_raw_sum": pheno.get("raw_score_sum", 0.0),
            "negation_penalty": pheno.get("negation_penalty", 0.0),
            "raw_genotype_score": geno["genotype_score"],
            "raw_inheritance_score": 0.0,
            "raw_family_evidence_score": 0.0,
            "raw_pedigree_mode_score": 0.0,
            "family_gene_support": 0.0,
            "family_disease_support": 0.0,
            "family_phenotype_support": 0.0,
            "family_system_support": 0.0,
            "raw_demographics_score": demo["demographics_score"],
            "raw_cases_score": cases,
            "raw_specific_signal_score": specific["specific_signal_score"],
            "raw_incongruity_match_score": incong["incongruity_match_score"],
            "raw_cooccurrence_pairs_score": cooc["cooccurrence_pairs_score"],
            "matched_hpo_count": len(pheno.get("matched_hpos", [])),
            "matched_gene_count": geno.get("gene_count", 0),
        })

    df = pd.DataFrame(rows)
    if len(df) == 0:
        # Return empty frame with expected columns to avoid downstream crashes
        empty_cols = [
            "disease_id", "disease_name", "group_id", "group_name",
            "raw_phenotype_score", "phenotype_raw_sum", "negation_penalty",
            "raw_genotype_score", "raw_inheritance_score", "raw_demographics_score",
            "raw_family_evidence_score", "raw_pedigree_mode_score",
            "family_gene_support", "family_disease_support", "family_phenotype_support",
            "family_system_support",
            "raw_cases_score", "raw_specific_signal_score", "raw_incongruity_match_score",
            "raw_cooccurrence_pairs_score", "matched_hpo_count", "matched_gene_count",
            "phenotype_score", "cases_score", "genotype_score", "inheritance_score",
            "family_evidence_score",
            "demographics_score", "specific_signal_score", "incongruity_match_score",
            "cooccurrence_pairs_score", "total_score", "rank",
        ]
        return pd.DataFrame(columns=empty_cols)

    # Rescale
    df["phenotype_score"] = _log_min_max_rescale(df["raw_phenotype_score"].values.astype(float), tie_floor=0.0)
    df["cases_score"] = _log_min_max_rescale(df["raw_cases_score"].values.astype(float), tie_floor=0.0)
    df["genotype_score"] = _min_max_rescale(df["raw_genotype_score"].values.astype(float), tie_floor=0.0)
    df["family_evidence_score"] = 0.0
    df["inheritance_score"] = 0.0
    df["demographics_score"] = _min_max_rescale(df["raw_demographics_score"].values.astype(float), tie_floor=0.0)
    df["specific_signal_score"] = _min_max_rescale(df["raw_specific_signal_score"].values.astype(float), tie_floor=0.0)
    df["incongruity_match_score"] = _min_max_rescale(df["raw_incongruity_match_score"].values.astype(float), tie_floor=0.0)
    df["cooccurrence_pairs_score"] = _min_max_rescale(df["raw_cooccurrence_pairs_score"].values.astype(float), tie_floor=0.0)

    # Combine
    def _total(row: pd.Series) -> float:
        return (
            weights.get("phenotype", 0.0) * row["phenotype_score"]
            + weights.get("genotype", 0.0) * row["genotype_score"]
            + weights.get("family_evidence", weights.get("inheritance", 0.0)) * row["family_evidence_score"]
            + weights.get("demographics", 0.0) * row["demographics_score"]
            + weights.get("cases", 0.0) * row["cases_score"]
            + weights.get("specific_signal", 0.0) * row["specific_signal_score"]
            + weights.get("incongruity_match", 0.0) * row["incongruity_match_score"]
            + weights.get("cooccurrence_pairs", 0.0) * row["cooccurrence_pairs_score"]
        )

    df["total_score"] = df.apply(_total, axis=1)

    # Family evidence is disease-specific but more expensive than the core
    # signals. Use the core ranking to shortlist, then add family evidence only
    # where it can plausibly affect the top rerank/audit set.
    if weights.get("family_evidence", weights.get("inheritance", 0.0)) > 0 and has_family:
        family_evidence = prepare_family_evidence(
            patient_state,
            inheritance_prior,
            hpo_normalizer,
        )
        if family_evidence.terms or family_evidence.diseases or family_evidence.genes or family_evidence.systems:
            shortlist_n = min(
                len(df),
                max(
                    retrieval_retain_top_k(cfg),
                    FAMILY_EVIDENCE_SHORTLIST_FLOOR,
                ),
            )
            shortlist_idx = (
                df.sort_values("total_score", ascending=False)
                .head(shortlist_n)
                .index
            )
            for idx in shortlist_idx:
                did = df.at[idx, "disease_id"]
                family = family_evidence_score(did, family_evidence, kg_index, hpo, hpo_normalizer)
                score = float(family["family_evidence_score"])
                df.at[idx, "raw_inheritance_score"] = score
                df.at[idx, "raw_family_evidence_score"] = score
                df.at[idx, "raw_pedigree_mode_score"] = float(family["pedigree_mode_support"])
                df.at[idx, "family_gene_support"] = float(family["family_gene_support"])
                df.at[idx, "family_disease_support"] = float(family["family_disease_support"])
                df.at[idx, "family_phenotype_support"] = float(family["family_phenotype_support"])
                df.at[idx, "family_system_support"] = float(family["family_system_support"])

            df["family_evidence_score"] = df["raw_family_evidence_score"].astype(float).clip(0.0, 1.0)
            df["inheritance_score"] = df["family_evidence_score"]
            df["total_score"] = df.apply(_total, axis=1)
    df = df.sort_values("total_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    # Save the weights used (handy for investigation)
    df.attrs["weights"] = weights
    return df
