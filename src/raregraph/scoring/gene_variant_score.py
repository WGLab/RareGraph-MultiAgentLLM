"""Genotype score (Bayesian log-LR over gene/variant matches).

Adapted from the original rare_dx_mcp/gene_variant_score.py with the same
likelihood-ratio design, but working against our new KGIndex rather than
the raw KG dict.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class GenotypeConfig:
    lr_variant_positive: float = 80.0
    lr_gene_positive: float = 50.0
    lr_variant_negative: float = 0.2
    lr_gene_negative: float = 0.5
    lr_gene_unknown: float = 3.0


def _norm_gene(g: Any) -> str:
    return str(g).strip().upper() if g else ""


def _norm_variant(v: Any) -> str:
    return str(v).strip().lower() if v else ""


def _variant_match(v: str, kg_variants: List[str]) -> bool:
    if not v:
        return False
    v_lc = v.lower()
    for kgv in kg_variants:
        kgv_lc = str(kgv).lower()
        if v_lc == kgv_lc or v_lc in kgv_lc or kgv_lc in v_lc:
            return True
    return False


def _collect_gene_entries(
    gene_mentions: List[Dict[str, Any]],
    vcf_summary: List[Dict[str, Any]],
    others: List[Dict[str, Any]] | None,
) -> Dict[str, Dict[str, Any]]:
    """Best-source-per-gene selection (VCF > text > other)."""
    bucket: Dict[str, Tuple[int, Dict[str, Any]]] = {}

    def _add(entries: List[Dict[str, Any]], priority: int):
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            g = _norm_gene(e.get("gene"))
            if not g:
                continue
            has_result = bool(e.get("result"))
            p = priority * 10 + (0 if has_result else 5)
            if g not in bucket or p < bucket[g][0]:
                bucket[g] = (p, e)

    _add(vcf_summary, 0)
    _add(gene_mentions, 1)
    _add(others or [], 2)
    return {g: e for g, (_, e) in bucket.items()}


def genotype_score(
    disease_id: str,
    kg: Dict[str, Dict[str, Any]],
    gene_mentions: List[Dict[str, Any]],
    vcf_summary: List[Dict[str, Any]],
    others: List[Dict[str, Any]] | None = None,
    cfg: GenotypeConfig | None = None,
) -> Dict[str, Any]:
    if cfg is None:
        cfg = GenotypeConfig()

    entry = kg.get(disease_id, {})
    kg_genes_raw = entry.get("genes", {}) or {}
    if isinstance(kg_genes_raw, dict):
        kg_gene_set = {_norm_gene(g) for g in kg_genes_raw.keys()}
        kg_variants = {
            _norm_gene(g): [
                v.get("name") if isinstance(v, dict) else v
                for v in (info.get("variants", []) if isinstance(info, dict) else [])
            ]
            for g, info in kg_genes_raw.items()
        }
    elif isinstance(kg_genes_raw, list):
        kg_gene_set = {_norm_gene(g) for g in kg_genes_raw}
        kg_variants = {}
    else:
        kg_gene_set = set()
        kg_variants = {}

    best = _collect_gene_entries(gene_mentions, vcf_summary, others)

    score = 0.0
    details: List[Dict[str, Any]] = []
    for g, e in best.items():
        if g not in kg_gene_set:
            continue
        variant = e.get("variant")
        result = str(e.get("result") or "").lower()
        variant_hit = _variant_match(_norm_variant(variant), kg_variants.get(g, []))

        if "pos" in result:
            lr = cfg.lr_variant_positive if variant_hit else cfg.lr_gene_positive
        elif "neg" in result:
            lr = cfg.lr_variant_negative if variant_hit else cfg.lr_gene_negative
        else:
            lr = cfg.lr_gene_unknown

        log_lr = math.log(lr)
        score += log_lr
        details.append({
            "gene": g,
            "variant": variant,
            "result": result,
            "variant_match": variant_hit,
            "LR": lr,
            "logLR": log_lr,
        })

    return {
        "genotype_score": score,
        "matched_gene_evidence": details,
        "gene_count": len(best),
    }
