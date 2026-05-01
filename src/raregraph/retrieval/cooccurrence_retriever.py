"""Co-occurrence pair retrieval.

For each pair of patient phenotypes (ancestor-expanded), look up in the
precomputed pair_to_diseases table. Rare pairs (few diseases) contribute
strongly; common pairs contribute weakly.

This is a RareGraph-specific retrieval channel that captures combination-level
signals beyond individual-phenotype matching.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Set

from raregraph.kg.kg_precompute import KGIndex
from raregraph.normalize.hpo_ontology import HpoOntology


def retrieve_by_cooccurrence(
    patient_hpos: List[Dict[str, Any]],
    kg_index: KGIndex,
    hpo: HpoOntology,
    ancestor_depth: int = 2,
    rare_threshold: int = 20,
) -> Dict[str, Dict[str, Any]]:
    """Return candidate_id → {matched_pairs: [...], pair_score: float}."""
    ids = [p.get("hpo_id") for p in patient_hpos if p.get("hpo_id")]
    if len(ids) < 2:
        return {}

    # Precompute expansions (ancestors) for each patient HPO
    expansions: Dict[str, Set[str]] = {}
    for h in ids:
        anc = _limited_ancestors(h, hpo, depth=ancestor_depth)
        expansions[h] = anc | {h}

    candidates: Dict[str, Dict[str, Any]] = {}

    for a, b in combinations(ids, 2):
        for ea in expansions[a]:
            for eb in expansions[b]:
                if ea == eb:
                    continue
                key = tuple(sorted([ea, eb]))
                freq = kg_index.pair_frequency.get(key, 0)
                if freq == 0 or freq > rare_threshold:
                    continue
                diseases = kg_index.pair_to_diseases.get(key, set())
                weight = 1.0 / (freq + 1)
                for did in diseases:
                    info = candidates.setdefault(
                        did, {"matched_pairs": [], "pair_score": 0.0}
                    )
                    info["matched_pairs"].append({
                        "patient_pair": (a, b),
                        "matched_pair": key,
                        "pair_frequency": freq,
                        "weight": weight,
                    })
                    info["pair_score"] += weight
    return candidates


def _limited_ancestors(hid: str, hpo: HpoOntology, depth: int) -> Set[str]:
    frontier = {hid}
    collected = set()
    for _ in range(depth):
        new_frontier = set()
        for node in frontier:
            if node not in hpo.directed:
                continue
            for p in hpo.directed.successors(node):
                if p not in collected:
                    collected.add(p)
                    new_frontier.add(p)
        if not new_frontier:
            break
        frontier = new_frontier
    return collected
