"""Stage 9: Clinical evidence scorecard output."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


def build_scorecard(
    patient_state: Any,
    final_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
    reconciled: Dict[str, Any],
    frontier_output: Dict[str, Any],
    top_k: int = 10,
) -> Dict[str, Any]:
    audit_by_id = {a["disease_id"]: a for a in audit_results}

    top = final_df.head(top_k).copy()
    top_cards: List[Dict[str, Any]] = []

    for _, row in top.iterrows():
        did = row["disease_id"]
        audit = audit_by_id.get(did, {})

        composite_rank = int(row.get("rank", 0))
        adjusted_rank = int(row.get("adjusted_rank", composite_rank))
        reranked_rank = int(row.get("reranked_rank_subtype", adjusted_rank))
        reconciled_rank = int(row.get("reconciled_rank", reranked_rank))

        total_change = composite_rank - reconciled_rank if composite_rank and reconciled_rank else 0

        # Rank change driver attribution
        drivers = []
        if audit.get("source") == "frontier_underranked":
            drivers.append(f"Frontier consultation flagged as underranked (lens {audit.get('frontier_lens','')})")
        elif audit.get("source") == "frontier_overranked":
            drivers.append(f"Frontier consultation flagged as overranked (lens {audit.get('frontier_lens','')})")
        if audit.get("plausibility") == "strong" and audit.get("source") == "llm_audit":
            drivers.append("Local audit rated strong")
        if audit.get("plausibility") == "weak" and audit.get("source") == "llm_audit":
            drivers.append("Local audit rated weak")
        if float(row.get("llm_validation_score", 0.0)) > 0:
            drivers.append("Grounded LLM validation found phenotype support")
        if float(row.get("llm_validation_score", 0.0)) < 0:
            drivers.append("Grounded LLM validation found phenotype contradiction")
        if float(row.get("family_evidence_score", 0.0)) > 0:
            drivers.append("Family history supports candidate-specific phenotype/gene/disease context")

        top_cards.append({
            "disease_id": did,
            "disease_name": row["disease_name"],
            "group_id": row.get("group_id", ""),
            "group_name": row.get("group_name", ""),
            "composite_rank": composite_rank,
            "adjusted_rank": adjusted_rank,
            "reranked_rank": reranked_rank,
            "reconciled_rank": reconciled_rank,
            "total_rank_change": total_change,
            "audit_plausibility": audit.get("plausibility", "not_audited"),
            "supporting_evidence": audit.get("supporting_evidence", []),
            "contradicting_evidence": audit.get("contradicting_evidence", []),
            "missing_expected_evidence": audit.get("missing_expected_evidence", []),
            "key_distinguishing_note": audit.get("key_distinguishing_note", ""),
            "llm_validation_score": float(row.get("llm_validation_score", 0.0)),
            "family_evidence_score": float(row.get("family_evidence_score", 0.0)),
            "pedigree_mode_support": float(row.get("raw_pedigree_mode_score", 0.0)),
            "family_phenotype_support": float(row.get("family_phenotype_support", 0.0)),
            "family_gene_support": float(row.get("family_gene_support", 0.0)),
            "family_disease_support": float(row.get("family_disease_support", 0.0)),
            "family_system_support": float(row.get("family_system_support", 0.0)),
            "frontier_reasoning": audit.get("frontier_reasoning", ""),
            "frontier_lens": audit.get("frontier_lens", ""),
            "rank_change_drivers": drivers,
        })

    # Suggested workup from high-importance missing evidence in top-5
    workup: List[str] = []
    for card in top_cards[:5]:
        for m in card["missing_expected_evidence"]:
            if m.get("importance") == "high":
                workup.append(
                    f"{card['disease_name']}: check {m.get('cue','')}"
                )

    scorecard = {
        "top_candidates": top_cards,
        "suggested_workup": list(dict.fromkeys(workup)),  # dedupe preserving order
        "reconciled": {
            "top_group": reconciled.get("top_group"),
            "top_subtype": reconciled.get("top_subtype"),
            "disagreement": reconciled.get("disagreement", False),
            "tiebreaker": reconciled.get("tiebreaker"),
            "method": reconciled.get("method", "n/a"),
        },
        "frontier": {
            "triggered": frontier_output.get("triggered", False) if frontier_output else False,
            "trigger_reason": frontier_output.get("trigger_reason", "") if frontier_output else "",
            "underranked": frontier_output.get("underranked", []) if frontier_output else [],
            "overranked": frontier_output.get("overranked", []) if frontier_output else [],
        },
    }
    return scorecard


def format_scorecard_text(scorecard: Dict[str, Any]) -> str:
    """Human-readable version."""
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("RAREGRAPH CLINICAL SCORECARD")
    lines.append("=" * 70)
    lines.append("")

    rec = scorecard.get("reconciled", {})
    if rec.get("top_subtype"):
        ts = rec["top_subtype"]
        lines.append(f"TOP CANDIDATE (reconciled): {ts.get('disease_name')} ({ts.get('disease_id')})")
    if rec.get("top_group"):
        tg = rec["top_group"]
        lines.append(f"TOP GROUP: {tg.get('disease_name')} ({tg.get('disease_id')})")
    if rec.get("disagreement"):
        lines.append(f"  ⚠ Group / subtype disagreement. Tiebreaker: {rec.get('tiebreaker')}")
    lines.append("")

    fr = scorecard.get("frontier", {})
    if fr.get("triggered"):
        lines.append(f"FRONTIER CONSULTATION: triggered ({fr.get('trigger_reason','')})")
        if fr.get("underranked"):
            lines.append(f"  Underranked: {[u['disease_name'] for u in fr['underranked']]}")
        if fr.get("overranked"):
            lines.append(f"  Overranked:  {[u['disease_name'] for u in fr['overranked']]}")
    else:
        lines.append("FRONTIER CONSULTATION: not triggered")
    lines.append("")

    lines.append("TOP CANDIDATES")
    lines.append("-" * 70)
    for i, c in enumerate(scorecard.get("top_candidates", []), 1):
        lines.append(f"{i}. {c['disease_name']} ({c['disease_id']})")
        lines.append(
            f"   ranks: composite={c['composite_rank']} → audit={c['adjusted_rank']} "
            f"→ pairwise={c['reranked_rank']} → final={c['reconciled_rank']} "
            f"(change: {c['total_rank_change']:+d})"
        )
        lines.append(f"   audit: {c['audit_plausibility']}")
        if c.get("llm_validation_score"):
            lines.append(f"   llm validation score: {c['llm_validation_score']:+.2f}")
        if c.get("family_evidence_score"):
            lines.append(f"   family evidence score: {c['family_evidence_score']:.2f}")
        if c.get("frontier_reasoning"):
            lines.append(f"   frontier ({c.get('frontier_lens','')}): {c['frontier_reasoning']}")
        if c.get("key_distinguishing_note"):
            lines.append(f"   note: {c['key_distinguishing_note']}")
        if c["supporting_evidence"]:
            lines.append("   ✓ Supporting:")
            for s in c["supporting_evidence"][:4]:
                lines.append(f"     - {s.get('cue','')}")
        if c["contradicting_evidence"]:
            lines.append("   ✗ Contradicting:")
            for s in c["contradicting_evidence"][:3]:
                lines.append(f"     - {s.get('cue','')}")
        if c["missing_expected_evidence"]:
            highs = [m for m in c["missing_expected_evidence"] if m.get("importance") == "high"]
            if highs:
                lines.append("   ? Missing (high importance):")
                for m in highs[:3]:
                    lines.append(f"     - {m.get('cue','')}")
        if c.get("rank_change_drivers"):
            lines.append("   rank change drivers:")
            for d in c["rank_change_drivers"]:
                lines.append(f"     - {d}")
        lines.append("")

    if scorecard.get("suggested_workup"):
        lines.append("SUGGESTED ADDITIONAL WORKUP")
        lines.append("-" * 70)
        for w in scorecard["suggested_workup"]:
            lines.append(f"  - {w}")
        lines.append("")

    return "\n".join(lines)


def build_rank_trajectory(
    ranked_df: pd.DataFrame,
    audit_results: List[Dict[str, Any]],
    reranked_df: pd.DataFrame,
    reconciled: Dict[str, Any],
    frontier_output: Dict[str, Any],
) -> pd.DataFrame:
    """Build the per-candidate rank trajectory table for investigation."""
    df = ranked_df.copy()
    if "group_id" not in df.columns:
        df["group_id"] = df.get("mondo_group_id", "")
    if "group_name" not in df.columns:
        df["group_name"] = ""

    audit_by_id = {a["disease_id"]: a for a in audit_results}

    # Frontier flags
    frontier_flag: Dict[str, str] = {}
    frontier_reason: Dict[str, str] = {}
    for u in (frontier_output.get("underranked", []) if frontier_output else []):
        frontier_flag[u.get("disease_id", "")] = "underranked"
        frontier_reason[u.get("disease_id", "")] = u.get("reasoning", "")
    for o in (frontier_output.get("overranked", []) if frontier_output else []):
        frontier_flag[o.get("disease_id", "")] = "overranked"
        frontier_reason[o.get("disease_id", "")] = o.get("reasoning", "")

    df["frontier_flag"] = df["disease_id"].map(lambda d: frontier_flag.get(d, "none"))
    df["frontier_reasoning"] = df["disease_id"].map(lambda d: frontier_reason.get(d, ""))

    # Audit
    df["audit_plausibility"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("plausibility", "not_audited"))
    df["audit_multiplier"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("multiplier", 1.0))
    df["llm_validation_raw_score"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("validation_raw_score", 0.0))
    df["llm_validation_source"] = df["disease_id"].map(lambda d: audit_by_id.get(d, {}).get("validation_source", ""))

    # Merge reranked positions
    rerank_cols = [c for c in reranked_df.columns if c.startswith("reranked_") or c.startswith("final_")]
    merged = df.merge(
        reranked_df[["disease_id"] + rerank_cols].drop_duplicates("disease_id"),
        on="disease_id", how="left"
    )

    # Reconciled rank
    rec_df = reconciled.get("reconciled_df")
    if isinstance(rec_df, pd.DataFrame) and "reconciled_rank" in rec_df.columns:
        merged = merged.merge(
            rec_df[["disease_id", "reconciled_rank", "reconciled_score", "alpha"]].drop_duplicates("disease_id"),
            on="disease_id", how="left",
        )

    # Rank change
    if "reconciled_rank" in merged.columns:
        merged["rank_change_total"] = merged["rank"] - merged["reconciled_rank"]

    return merged
