"""Composite ranker: Stage 3 of RareMind.

Combines the production phenotype/KG scoring components:
  1. Phenotype (IC-weighted + freq-aware negation + competitive IC)
  2. Genotype (Bayesian log-LR)
  3. Inheritance (prior × compatibility)
  4. Demographics (age/sex/ethnicity)
  5. Cases (PubCaseFinder)
  6. Specific signal (high-IC hallmark match bonus)
  7. Incongruity match (bridges dominant + outlier branches)
  8. Co-occurrence pairs (rare pair matches)
  9. Semantic similarity (HPO MICA best-match-average)
10. Coverage-aware phenotype profile (patient recall + branch coverage)
11. Evidence-quality controls (generic guard + anchor cluster)

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

from raremind.core.compat import to_dict
from raremind.core.config import cfg_get, retrieval_retain_top_k
from raremind.kg.kg_precompute import KGIndex
from raremind.normalize.hpo_ontology import HpoOntology

from .phenotype_score import phenotype_score, PhenotypeScoreConfig
from .phenotype_profile_score import phenotype_profile_score
from .specific_signal_score import specific_signal_score
from .incongruity_match_score import incongruity_match_score
from .cooccurrence_score import cooccurrence_score
from .semantic_similarity_score import semantic_similarity_score
from .gene_variant_score import genotype_score, GenotypeConfig
from .family_evidence_score import prepare_family_evidence, family_evidence_score
from .demographics_score import demographics_score

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

def _case_branch_concentration(present_hpos: List[Dict[str, Any]], hpo: HpoOntology) -> float:
    """Fraction of patient phenotype branch assignments in the dominant HPO branch."""
    branch_counts: Dict[str, int] = {}
    for item in present_hpos:
        hid = item.get("hpo_id")
        if not hid:
            continue
        branches = hpo.get_branches(str(hid))
        if not branches:
            branch_counts["NO_BRANCH"] = branch_counts.get("NO_BRANCH", 0) + 1
        for branch in branches:
            branch_counts[branch] = branch_counts.get(branch, 0) + 1
    total = sum(branch_counts.values())
    return float(max(branch_counts.values()) / total) if total else 0.0


def _case_low_ic_fraction(present_hpos: List[Dict[str, Any]], hpo: HpoOntology) -> float:
    ics = []
    for item in present_hpos:
        hid = item.get("hpo_id")
        if hid:
            ics.append(float(hpo.get_ic(str(hid))))
    return float(sum(ic < 2.0 for ic in ics) / len(ics)) if ics else 0.0


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
    weights = dict(base_weights)

    logger.info(f"Scoring weights: {weights}")

    ic_high = cfg.scoring.ic_high_threshold
    if ic_high is None:
        ic_high = hpo.ic_p75

    scoring_cfg = cfg_get(cfg, "scoring", {})
    use_specific_signal = bool(cfg_get(scoring_cfg, "use_specific_signal", True))
    use_incongruity_match = bool(cfg_get(scoring_cfg, "use_incongruity_match", True))
    use_cooccurrence_pairs = bool(cfg_get(scoring_cfg, "use_cooccurrence_pairs", True))
    use_semantic_similarity = bool(cfg_get(scoring_cfg, "use_semantic_similarity", False))
    use_phenotype_profile = bool(cfg_get(scoring_cfg, "use_phenotype_profile", False))
    use_generic_guard = bool(cfg_get(scoring_cfg, "use_generic_guard", False))
    use_anchor_cluster = bool(cfg_get(scoring_cfg, "use_anchor_cluster", False))
    semantic_max_disease_hpos = int(cfg_get(scoring_cfg, "semantic_similarity_max_disease_hpos", 80))
    phenotype_size_cap = int(cfg_get(scoring_cfg, "phenotype_disease_size_norm_cap", 60))
    gene_evidence_policy = str(
        cfg_get(scoring_cfg, "gene_evidence_policy", "negative_only")
    ).strip().lower()
    phenotype_cfg = PhenotypeScoreConfig(disease_size_norm_cap=phenotype_size_cap)
    genotype_cfg = GenotypeConfig(evidence_policy=gene_evidence_policy)

    if not use_specific_signal:
        weights["specific_signal"] = 0.0
    if not use_incongruity_match:
        weights["incongruity_match"] = 0.0
    if not use_cooccurrence_pairs:
        weights["cooccurrence_pairs"] = 0.0
    if not use_semantic_similarity:
        weights["semantic_similarity"] = 0.0
    if not use_phenotype_profile:
        weights["phenotype_profile"] = 0.0
    if not use_generic_guard:
        weights["generic_guard"] = 0.0
    if not use_anchor_cluster:
        weights["anchor_cluster"] = 0.0

    branch_concentration = _case_branch_concentration(present_hpos, hpo)
    low_ic_fraction = _case_low_ic_fraction(present_hpos, hpo)

    # Compute raw per-component scores for every candidate
    rows: List[Dict[str, Any]] = []
    for did in candidate_ids:
        name = kg_index.disease_name.get(did, did)
        group_id = kg_index.disease_group.get(did, "") or did
        group_name = kg_index.disease_name.get(group_id, group_id)

        pheno = phenotype_score(
            did,
            present_hpos,
            negated_hpos,
            kg_index,
            kg,
            hpo,
            total_candidates,
            hpo_normalizer=hpo_normalizer,
            cfg=phenotype_cfg,
        )
        specific = specific_signal_score(did, present_hpos, kg_index, kg, hpo, ic_high) if use_specific_signal else {"specific_signal_score": 0.0}
        incong = incongruity_match_score(did, incongruity, kg_index, hpo) if use_incongruity_match else {"incongruity_match_score": 0.0}
        cooc = cooccurrence_score(did, cooccurrence_candidates) if use_cooccurrence_pairs else {"cooccurrence_pairs_score": 0.0}
        semantic = (
            semantic_similarity_score(did, present_hpos, kg_index, hpo, max_disease_hpos=semantic_max_disease_hpos)
            if use_semantic_similarity
            else {"semantic_similarity_score": 0.0}
        )
        profile = (
            phenotype_profile_score(did, present_hpos, kg_index, kg, hpo)
            if use_phenotype_profile
            else {
                "phenotype_profile_score": 0.0,
                "profile_patient_recall": 0.0,
                "profile_unweighted_recall": 0.0,
                "profile_branch_recall": 0.0,
                "profile_match_coverage": 0.0,
                "profile_supportive_cluster": 0.0,
                "profile_generic_guard": 0.0,
                "profile_anchor_cluster": 0.0,
                "profile_high_ic_recall": 0.0,
                "profile_specific_hit_fraction": 0.0,
                "profile_match_density": 0.0,
                "profile_matched_count": 0,
                "profile_supportive_matches": 0,
                "profile_matched_branches": 0,
            }
        )
        geno = genotype_score(did, kg, gene_mentions, vcf_summary, cfg=genotype_cfg)
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
            "raw_semantic_similarity_score": semantic["semantic_similarity_score"],
            "raw_phenotype_profile_score": profile["phenotype_profile_score"],
            "raw_generic_guard_score": profile.get("profile_generic_guard", 0.0) if use_generic_guard else 0.0,
            "raw_anchor_cluster_score": profile.get("profile_anchor_cluster", 0.0) if use_anchor_cluster else 0.0,
            "profile_patient_recall": profile.get("profile_patient_recall", 0.0),
            "profile_unweighted_recall": profile.get("profile_unweighted_recall", 0.0),
            "profile_branch_recall": profile.get("profile_branch_recall", 0.0),
            "profile_match_coverage": profile.get("profile_match_coverage", 0.0),
            "profile_supportive_cluster": profile.get("profile_supportive_cluster", 0.0),
            "profile_high_ic_recall": profile.get("profile_high_ic_recall", 0.0),
            "profile_specific_hit_fraction": profile.get("profile_specific_hit_fraction", 0.0),
            "profile_match_density": profile.get("profile_match_density", 0.0),
            "profile_matched_count": profile.get("profile_matched_count", 0),
            "profile_supportive_matches": profile.get("profile_supportive_matches", 0),
            "profile_matched_branches": profile.get("profile_matched_branches", 0),
            "case_n_patient_hpos": len(present_hpos),
            "case_branch_concentration": branch_concentration,
            "case_low_ic_fraction": low_ic_fraction,
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
            "raw_cooccurrence_pairs_score", "raw_semantic_similarity_score",
            "raw_phenotype_profile_score", "raw_generic_guard_score", "raw_anchor_cluster_score",
            "profile_patient_recall", "profile_unweighted_recall",
            "profile_branch_recall", "profile_match_coverage", "profile_supportive_cluster",
            "profile_high_ic_recall", "profile_specific_hit_fraction", "profile_match_density",
            "profile_matched_count", "profile_supportive_matches", "profile_matched_branches",
            "matched_hpo_count", "matched_gene_count",
            "phenotype_score", "cases_score", "genotype_score", "inheritance_score",
            "family_evidence_score",
            "demographics_score", "specific_signal_score", "incongruity_match_score",
            "cooccurrence_pairs_score", "semantic_similarity_score",
            "phenotype_profile_score", "generic_guard_score", "anchor_cluster_score",
            "total_score", "rank",
        ]
        return pd.DataFrame(columns=empty_cols)

    # Rescale
    df["phenotype_score"] = _log_min_max_rescale(df["raw_phenotype_score"].values.astype(float), tie_floor=0.0)
    df["cases_score"] = _log_min_max_rescale(df["raw_cases_score"].values.astype(float), tie_floor=0.0)
    if gene_evidence_policy == "negative_only":
        raw_genotype = df["raw_genotype_score"].values.astype(float)
        neg = np.minimum(raw_genotype, 0.0)
        scale = abs(float(neg.min())) if len(neg) and neg.min() < 0 else 1.0
        df["genotype_score"] = neg / scale
    else:
        df["genotype_score"] = _min_max_rescale(
            df["raw_genotype_score"].values.astype(float), tie_floor=0.0
        )
    df["family_evidence_score"] = 0.0
    df["inheritance_score"] = 0.0
    df["demographics_score"] = _min_max_rescale(df["raw_demographics_score"].values.astype(float), tie_floor=0.0)
    df["specific_signal_score"] = _min_max_rescale(df["raw_specific_signal_score"].values.astype(float), tie_floor=0.0)
    df["incongruity_match_score"] = _min_max_rescale(df["raw_incongruity_match_score"].values.astype(float), tie_floor=0.0)
    df["cooccurrence_pairs_score"] = _min_max_rescale(df["raw_cooccurrence_pairs_score"].values.astype(float), tie_floor=0.0)
    df["semantic_similarity_score"] = _min_max_rescale(df["raw_semantic_similarity_score"].values.astype(float), tie_floor=0.0)
    df["phenotype_profile_score"] = _min_max_rescale(df["raw_phenotype_profile_score"].values.astype(float), tie_floor=0.0)
    df["generic_guard_score"] = _min_max_rescale(df["raw_generic_guard_score"].values.astype(float), tie_floor=0.0)
    df["anchor_cluster_score"] = _min_max_rescale(df["raw_anchor_cluster_score"].values.astype(float), tie_floor=0.0)

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
            + weights.get("semantic_similarity", 0.0) * row["semantic_similarity_score"]
            + weights.get("phenotype_profile", 0.0) * row["phenotype_profile_score"]
            + weights.get("generic_guard", 0.0) * row["generic_guard_score"]
            + weights.get("anchor_cluster", 0.0) * row["anchor_cluster_score"]
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
