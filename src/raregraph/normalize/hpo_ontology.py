"""HPO ontology wrapper.

Provides:
  - Term name lookup
  - Information content (IC) computation from HPO annotation frequency
  - Top-level organ-system branch lookup per HPO term
  - Global IC statistics (median, p25, p75)

The IC calculation uses the HPO annotation file (phenotype_to_genes or
phenotype.hpoa) when available; otherwise falls back to a simpler depth-based
proxy.
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx
import obonet

logger = logging.getLogger(__name__)


# Top-level HPO branches (direct children of HP:0000118 "Phenotypic abnormality").
# Keep this aligned with the bundled HPO OBO; used for organ-system localization.
TOP_LEVEL_BRANCHES = {
    "HP:0033127": "Abnormality of the musculoskeletal system",
    "HP:0040064": "Abnormality of limbs",
    "HP:0000707": "Abnormality of the nervous system",
    "HP:0001939": "Abnormality of metabolism/homeostasis",
    "HP:0000119": "Abnormality of the genitourinary system",
    "HP:0001626": "Abnormality of the cardiovascular system",
    "HP:0000152": "Abnormality of head or neck",
    "HP:0002715": "Abnormality of the immune system",
    "HP:0000478": "Abnormality of the eye",
    "HP:0001574": "Abnormality of the integument",
    "HP:0001871": "Abnormality of blood and blood-forming tissues",
    "HP:0025031": "Abnormality of the digestive system",
    "HP:0002664": "Neoplasm",
    "HP:0002086": "Abnormality of the respiratory system",
    "HP:0000818": "Abnormality of the endocrine system",
    "HP:0025354": "Abnormal cellular phenotype",
    "HP:0000598": "Abnormality of the ear",
    "HP:0001197": "Abnormality of prenatal development or birth",
    "HP:0025142": "Constitutional symptom",
    "HP:0001507": "Growth abnormality",
    "HP:0000769": "Abnormality of the breast",
    "HP:0001608": "Abnormality of the voice",
    "HP:0045027": "Abnormality of the thoracic cavity",
}


class HpoOntology:
    def __init__(self, hpo_obo_path: str):
        path = Path(hpo_obo_path)
        if not path.exists():
            raise FileNotFoundError(
                f"HPO obo file not found at {hpo_obo_path}. "
                "Run scripts/00_download_hpo.py first."
            )
        logger.info(f"Loading HPO ontology from {path} ...")
        g = obonet.read_obo(str(path))
        # Keep only HP:* nodes
        nodes = [n for n in g.nodes if str(n).startswith("HP:")]
        g = g.subgraph(nodes).copy()

        self.directed = g               # directed (child -> parent) DAG as obonet returns
        self.undirected = g.to_undirected()
        self.id_to_name = {n: d.get("name", n) for n, d in g.nodes(data=True)}
        self.name_to_id = {v.lower(): k for k, v in self.id_to_name.items()}

        # Cache branch lookups
        self._branch_cache: Dict[str, Set[str]] = {}
        self._mica_ic_cache: Dict[tuple[str, str], float] = {}

        # IC placeholders; set via set_ic_from_counts(...) or compute_ic_from_kg(...)
        self.ic: Dict[str, float] = {}
        self.ic_median: float = 0.0
        self.ic_p25: float = 0.0
        self.ic_p75: float = 0.0

        logger.info(f"HPO loaded: {len(self.id_to_name)} terms")

    # -----------------------------------------------------------
    # Ontology walks
    # -----------------------------------------------------------
    def get_name(self, hpo_id: str) -> str:
        return self.id_to_name.get(hpo_id, hpo_id)

    def get_ancestors(self, hpo_id: str, include_self: bool = False) -> Set[str]:
        """All ancestors (transitive parents) of hpo_id."""
        if hpo_id not in self.directed:
            return set()
        anc = set(nx.descendants(self.directed, hpo_id))  # in obonet, 'descendants' on directed = parents
        # obonet returns edges child->parent, so nx.descendants goes up the DAG
        if include_self:
            anc.add(hpo_id)
        return anc

    def get_descendants(self, hpo_id: str, include_self: bool = False) -> Set[str]:
        """All descendants (transitive children)."""
        if hpo_id not in self.directed:
            return set()
        desc = set(nx.ancestors(self.directed, hpo_id))
        if include_self:
            desc.add(hpo_id)
        return desc

    def get_parents(self, hpo_id: str) -> Set[str]:
        """Direct parents of hpo_id."""
        if hpo_id not in self.directed:
            return set()
        return set(self.directed.successors(hpo_id))

    def get_siblings(self, hpo_id: str) -> Set[str]:
        """Terms sharing at least one parent with hpo_id."""
        if hpo_id not in self.directed:
            return set()
        sibs: Set[str] = set()
        for p in self.directed.successors(hpo_id):  # parents
            for c in self.directed.predecessors(p):  # siblings (including self)
                if c != hpo_id:
                    sibs.add(c)
        return sibs

    def get_children(self, hpo_id: str) -> Set[str]:
        if hpo_id not in self.directed:
            return set()
        return set(self.directed.predecessors(hpo_id))

    def get_mica_ic(self, hpo_a: str, hpo_b: str) -> float:
        """Return the IC of the most informative common ancestor."""
        key = tuple(sorted((hpo_a, hpo_b)))
        if key in self._mica_ic_cache:
            return self._mica_ic_cache[key]
        if hpo_a not in self.directed or hpo_b not in self.directed:
            self._mica_ic_cache[key] = 0.0
            return 0.0
        common = self.get_ancestors(hpo_a, include_self=True) & self.get_ancestors(hpo_b, include_self=True)
        mica_ic = max((self.get_ic(h) for h in common), default=0.0)
        self._mica_ic_cache[key] = mica_ic
        return mica_ic

    def get_branches(self, hpo_id: str) -> Set[str]:
        """Return the set of top-level organ-system branches this term belongs to."""
        if hpo_id in self._branch_cache:
            return self._branch_cache[hpo_id]
        anc = self.get_ancestors(hpo_id, include_self=True)
        branches = anc & set(TOP_LEVEL_BRANCHES.keys())
        self._branch_cache[hpo_id] = branches
        return branches

    def get_branch_names(self, hpo_id: str) -> List[str]:
        return [TOP_LEVEL_BRANCHES[b] for b in self.get_branches(hpo_id)]

    # -----------------------------------------------------------
    # Information content
    # -----------------------------------------------------------
    def compute_ic_from_kg(
        self,
        kg: Dict[str, Dict[str, Any]],
        phenotype_field: str = "phenotypes",
    ) -> None:
        """Compute IC for every HPO term from a KG's disease->phenotype annotations."""
        logger.info("Computing IC from KG annotation frequencies ...")

        # Count how many diseases each HPO (or its descendant) is annotated to
        total_diseases = len(kg)
        if total_diseases == 0:
            return

        # Direct annotation counts
        direct_counts: Dict[str, int] = {}
        for did, entry in kg.items():
            phens = entry.get(phenotype_field, {}) or {}
            if isinstance(phens, dict):
                hpo_ids = [
                    info.get("hpo") or info.get("hpo_id")
                    for info in phens.values()
                    if isinstance(info, dict)
                ]
            elif isinstance(phens, list):
                hpo_ids = [
                    (p.get("hpo") or p.get("hpo_id")) if isinstance(p, dict) else p
                    for p in phens
                ]
            else:
                continue
            for h in hpo_ids:
                if not h or not str(h).startswith("HP:"):
                    continue
                direct_counts[h] = direct_counts.get(h, 0) + 1

        # Propagate annotations up the DAG (true-path rule).
        propagated: Dict[str, int] = {}
        for h, c in direct_counts.items():
            if h not in self.directed:
                continue
            for anc in self.get_ancestors(h, include_self=True):
                propagated[anc] = propagated.get(anc, 0) + c

        # IC = -log(p); higher IC = rarer
        self.ic = {}
        for h, c in propagated.items():
            p = c / total_diseases
            if p > 0:
                self.ic[h] = -math.log(p)

        # Terms with zero count get max IC
        if self.ic:
            max_ic = max(self.ic.values())
            for h in self.id_to_name:
                if h not in self.ic:
                    self.ic[h] = max_ic

        # Stats
        import numpy as np
        values = np.array(list(self.ic.values()))
        self.ic_p25 = float(np.percentile(values, 25))
        self.ic_median = float(np.percentile(values, 50))
        self.ic_p75 = float(np.percentile(values, 75))

        logger.info(
            f"IC stats: p25={self.ic_p25:.2f}, median={self.ic_median:.2f}, p75={self.ic_p75:.2f}"
        )

    def get_ic(self, hpo_id: str) -> float:
        return self.ic.get(hpo_id, self.ic_median or 3.0)

    # -----------------------------------------------------------
    # N-degree expansion (legacy, for backward compat with fixed_degree mode)
    # -----------------------------------------------------------
    def get_n_degree_nodes(self, hpo_id: str, n: int) -> Dict[int, List[str]]:
        if hpo_id not in self.undirected:
            return {}

        lengths = nx.single_source_shortest_path_length(self.undirected, hpo_id, cutoff=n + 1)

        result: Dict[int, List[str]] = {}
        for node, dist in lengths.items():
            result.setdefault(dist, []).append(node)

        siblings = self.get_siblings(hpo_id)
        if 2 in result:
            result[2] = [x for x in result[2] if x not in siblings]
        result.setdefault(1, [])
        result[1].extend(list(siblings))

        return {k: result.get(k, []) for k in range(n + 1)}
