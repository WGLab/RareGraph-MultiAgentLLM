"""Patient-side HPO normalization pipeline.

Takes a list of phenotype mentions from the extractors, resolves each mention
to its best HPO ID via BioLORD, then attaches IC and onset. Result is the
normalized phenotype list consumed by downstream retrieval/scoring.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

from .biolord_embedder import BioLordEmbedder, OntologyIndex
from .hpo_ontology import HpoOntology

logger = logging.getLogger(__name__)


class HpoNormalizer:
    def __init__(
        self,
        hpo: HpoOntology,
        embedder: BioLordEmbedder,
        similarity_threshold: float = 0.75,
    ):
        self.hpo = hpo
        self.embedder = embedder
        self.threshold = similarity_threshold
        self._index: OntologyIndex | None = None
        self._id_to_index: Dict[str, int] = {}
        self._similarity_cache: Dict[tuple[str, str], float] = {}

    def build_index(self, force: bool = False) -> None:
        ids = list(self.hpo.id_to_name.keys())
        names = [self.hpo.id_to_name[i] for i in ids]
        self._index = OntologyIndex(self.embedder, ids, names, cache_name="hpo_norm")
        self._index.build(force=force)
        self._id_to_index = {hid: i for i, hid in enumerate(ids)}

    def normalize(
        self,
        mentions: List[Dict[str, Any]],
        include_negated: bool = True,
    ) -> List[Dict[str, Any]]:
        """Normalize a list of phenotype dicts.

        Each input item has:
          - mention (str)
          - attribution ('patient' | 'negated' | 'family' | 'references' | 'others' | 'uncertain')
          - onset (optional)
          - evidence (optional)
          - source (optional; 'text' | 'vision' | 'free_hpo')

        Returns normalized items with additional fields:
          - hpo_id, hpo_name, score, ic, present (bool)
        """
        if self._index is None:
            self.build_index()

        keep = []
        texts = []
        for m in mentions:
            if m['source'] != 'free_hpo':
                attr = m.get("attribution")
                if attr == "patient" or (include_negated and attr == "negated"):
                    keep.append(m)
                    texts.append(m.get("mention") or "")

        if texts:
            matches = self._index.match(texts, top_k=1)
        else:
            matches = []
        for m in mentions:
            if m['source'] == 'free_hpo':
                keep.append(m)
                matches.append([[
                    m.get("mention"),
                    self.hpo.get_name(m.get("mention")),
                    1
                    ]])
        if not matches:
            return []
        out = []
        for m, mres in zip(keep, matches):
            if not mres:
                continue
            hid, hname, score = mres[0]
            if score < self.threshold:
                continue
            item = dict(m)
            if m['source'] == 'free_hpo':
                item['mention'] = hname
            item["hpo_id"] = hid
            item["hpo_name"] = hname
            item["score"] = float(score)
            item["ic"] = self.hpo.get_ic(hid)
            item["present"] = (m.get("attribution") == "patient")
            item["source"] = m.get("source", "text")
            out.append(item)
        return out

    def similarity_by_hpo_id(self, hpo_a: str, hpo_b: str) -> float:
        if self._index is None:
            self.build_index()
        if self._index is None or self._index.embeddings is None:
            return 0.0
        if not self._id_to_index:
            self._id_to_index = {hid: i for i, hid in enumerate(self._index.ids)}

        key = tuple(sorted((str(hpo_a), str(hpo_b))))
        if key in self._similarity_cache:
            return self._similarity_cache[key]

        idx_a = self._id_to_index.get(str(hpo_a))
        idx_b = self._id_to_index.get(str(hpo_b))
        if idx_a is None or idx_b is None:
            return 0.0

        score = float(self._index.embeddings[idx_a] @ self._index.embeddings[idx_b])
        self._similarity_cache[key] = score
        return score
