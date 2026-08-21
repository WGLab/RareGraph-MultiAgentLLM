"""Mondo disease normalization using BioLORD embeddings.

Used to map free-text disease mentions (e.g., from family history) to Mondo IDs.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .biolord_embedder import BioLordEmbedder, OntologyIndex

logger = logging.getLogger(__name__)


class MondoNormalizer:
    def __init__(
        self,
        full_mondo_path: str,
        embedder: BioLordEmbedder,
        similarity_threshold: float = 0.75,
    ):
        self.full_mondo_path = Path(full_mondo_path)
        self.embedder = embedder
        self.threshold = similarity_threshold

        self.id_to_name: Dict[str, str] = {}
        self.name_to_id: Dict[str, str] = {}
        self.synonyms: Dict[str, List[str]] = {}

        self._index: Optional[OntologyIndex] = None

    def load(self) -> None:
        if not self.full_mondo_path.exists():
            logger.warning(f"Mondo file not found: {self.full_mondo_path}. MondoNormalizer will be empty.")
            return
        with open(self.full_mondo_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Flexible: support dict-of-dicts or list-of-dicts
        if isinstance(data, dict):
            entries = list(data.items())
        elif isinstance(data, list):
            entries = [
                (e.get("id") or e.get("mondo_id") or f"entry{i}", e)
                for i, e in enumerate(data)
            ]
        else:
            logger.warning(f"Unrecognized Mondo file format: {type(data)}")
            return

        for did, entry in entries:
            meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}
            name = (
                entry.get("preferred_title")
                or entry.get("name")
                or entry.get("label")
                or meta.get("preferred_title")
                or meta.get("name")
                or meta.get("label")
                or did
            )
            self.id_to_name[did] = name
            self.name_to_id[name.lower()] = did
            syns = []
            for block in (
                entry.get("alternative_titles"),
                entry.get("synonyms"),
                entry.get("synonym"),
                meta.get("synonyms"),
                meta.get("aliases"),
            ):
                if isinstance(block, str):
                    syns.extend([x.strip() for x in block.split("|") if x.strip()])
                elif isinstance(block, list):
                    syns.extend([str(x).strip() for x in block if str(x).strip()])
            seen = set()
            cleaned = []
            for syn in syns:
                key = syn.lower()
                if key not in seen:
                    seen.add(key)
                    cleaned.append(syn)
            self.synonyms[did] = cleaned

        logger.info(f"Mondo loaded: {len(self.id_to_name)} diseases")

    def build_index(self, force: bool = False) -> None:
        ids, names = [], []
        for did, name in self.id_to_name.items():
            ids.append(did)
            names.append(name)
            synonym_list = self.synonyms.get(did, [])
            if isinstance(synonym_list, float):
                pass
            else:
                for syn in synonym_list:
                    ids.append(did)
                    names.append(syn)
        self._index = OntologyIndex(self.embedder, ids, names, cache_name="mondo_norm")
        self._index.build(force=force)

    def match_one(self, text: str) -> Optional[Tuple[str, str, float]]:
        if not text:
            return None
        key = text.strip().lower()
        if key in self.name_to_id:
            did = self.name_to_id[key]
            return (did, self.id_to_name[did], 1.0)
        if self._index is None:
            self.build_index()
        return self._index.match_one(text, threshold=self.threshold)
