"""IC-gated HPO expansion.

Core insight (corrected from earlier design):
  - LOW-IC (non-specific) terms → expand DOWN (to children) + siblings.
    A vague patient term like "abnormal rib cage morphology" can match a disease
    annotated with a more specific term like "dorsal rib defect".
  - MEDIUM-IC terms → expand to children only.
  - HIGH-IC (specific) terms → NO expansion. Already precise.

Credit for matches:
  - Descendant match: credit = IC(patient) / IC(matched)  (< 1, rewards specificity gain)
  - Ancestor match:   credit = IC(matched) / IC(patient)  (< 1, penalty for generality)
  - Sibling match:    credit = 0.6 (flat, configurable)
  - Exact match:      credit = 1.0
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple

from .hpo_ontology import HpoOntology

logger = logging.getLogger(__name__)


@dataclass
class ExpandedTerm:
    hpo_id: str
    credit: float
    relation: str            # "exact" | "child" | "sibling" | "ancestor"
    original_hpo_id: str


def ic_gated_expand(
    hpo_id: str,
    hpo: HpoOntology,
    mode: str = "ic_gated",
    max_depth: int = 1,
    sibling_credit: float = 0.6,
) -> List[ExpandedTerm]:
    """Expand a patient HPO term into a list of (id, credit, relation) tuples.

    mode:
      - "ic_gated"     : use IC thresholds (recommended)
      - "fixed_degree" : legacy behavior (expand symmetrically)
      - "none"         : no expansion
    """
    if hpo_id not in hpo.id_to_name:
        return [ExpandedTerm(hpo_id, 1.0, "exact", hpo_id)]

    out: List[ExpandedTerm] = [ExpandedTerm(hpo_id, 1.0, "exact", hpo_id)]

    if mode == "none":
        return out

    patient_ic = hpo.get_ic(hpo_id)

    if mode == "fixed_degree":
        # Legacy path
        degrees = hpo.get_n_degree_nodes(hpo_id, n=max_depth)
        for d, nodes in degrees.items():
            if d == 0:
                continue
            for n in nodes:
                matched_ic = hpo.get_ic(n)
                credit = min(1.0, matched_ic / patient_ic) if patient_ic > 0 else 0.5
                out.append(ExpandedTerm(n, credit, "child" if d == 1 else "ancestor", hpo_id))
        return out

    # ---- ic_gated ----
    p25 = hpo.ic_p25
    median = hpo.ic_median

    # Children + siblings for LOW-IC
    if patient_ic < p25:
        children = _get_children_within_depth(hpo, hpo_id, depth=max_depth)
        siblings = hpo.get_siblings(hpo_id)
        for c in children:
            if c == hpo_id:
                continue
            child_ic = hpo.get_ic(c)
            credit = patient_ic / child_ic if child_ic > 0 else 0.5
            credit = min(1.0, max(0.2, credit))
            out.append(ExpandedTerm(c, credit, "child", hpo_id))
        for s in siblings:
            if s == hpo_id:
                continue
            out.append(ExpandedTerm(s, sibling_credit, "sibling", hpo_id))

    # Children only for MEDIUM-IC
    elif patient_ic < median:
        children = _get_children_within_depth(hpo, hpo_id, depth=max_depth)
        for c in children:
            if c == hpo_id:
                continue
            child_ic = hpo.get_ic(c)
            credit = patient_ic / child_ic if child_ic > 0 else 0.5
            credit = min(1.0, max(0.3, credit))
            out.append(ExpandedTerm(c, credit, "child", hpo_id))

    # HIGH-IC: no expansion
    return out


def _get_children_within_depth(
    hpo: HpoOntology, hpo_id: str, depth: int
) -> List[str]:
    """BFS down the ontology from hpo_id for at most `depth` levels."""
    if depth <= 0:
        return []
    frontier = {hpo_id}
    collected = set()
    for _ in range(depth):
        new_frontier = set()
        for node in frontier:
            for ch in hpo.get_children(node):
                if ch not in collected and ch != hpo_id:
                    collected.add(ch)
                    new_frontier.add(ch)
        if not new_frontier:
            break
        frontier = new_frontier
    return list(collected)


def expand_patient_hpo_set(
    patient_hpos: List[str],
    hpo: HpoOntology,
    mode: str = "ic_gated",
    max_depth: int = 1,
) -> Dict[str, List[ExpandedTerm]]:
    """Apply IC-gated expansion to a list of patient HPO IDs."""
    result: Dict[str, List[ExpandedTerm]] = {}
    for hid in patient_hpos:
        result[hid] = ic_gated_expand(hid, hpo, mode=mode, max_depth=max_depth)
    return result
