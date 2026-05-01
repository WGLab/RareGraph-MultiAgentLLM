"""Gene-based candidate retrieval."""
from __future__ import annotations

from typing import Any, Dict, List, Set

from raregraph.kg.kg_precompute import KGIndex


def retrieve_by_gene(
    gene_mentions: List[Dict[str, Any]],
    vcf_summary: List[Dict[str, Any]],
    kg_index: KGIndex,
) -> Dict[str, Dict[str, Any]]:
    candidates: Dict[str, Dict[str, Any]] = {}

    genes: Set[str] = set()
    for g in gene_mentions or []:
        gn = g.get("gene")
        if gn:
            genes.add(gn.upper())
    for g in vcf_summary or []:
        gn = g.get("gene")
        if gn:
            genes.add(gn.upper())

    for g in genes:
        diseases = kg_index.gene_to_diseases.get(g, set())
        for did in diseases:
            info = candidates.setdefault(did, {"matched_genes": []})
            info["matched_genes"].append(g)

    return candidates
