"""BioLORD embedder for HPO and Mondo normalization.

Provides:
  - Lazy model loading (only on first encode)
  - Disk caching of ontology embeddings under data/indexes/
  - Batched embed() with normalized vectors
  - match_one(text, top_k) returning (id, name, score)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class BioLordEmbedder:
    def __init__(
        self,
        model_name: str = "FremyCompany/BioLORD-2023",
        cache_dir: str = "data/cache",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = None
        self._device = device

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedder: {self.model_name}")
            self._model = SentenceTransformer(self.model_name, device=self._device)
        return self._model

    def encode(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        model = self._load_model()
        embs = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 256,
            convert_to_numpy=True,
        )
        return embs.astype(np.float32)
    def similarity(self, text1: str, text2: str) -> float:
            """
            Compute cosine similarity between two strings.
            Returns a score in [-1, 1], where 1 = most similar.
            """
            embs = self.encode([text1, text2])
            score = float(np.dot(embs[0], embs[1]))  # already normalized
            return score

class OntologyIndex:
    """Embedded ontology (IDs + names) with cached vectors."""

    def __init__(
        self,
        embedder: BioLordEmbedder,
        ids: List[str],
        names: List[str],
        cache_name: str,
    ):
        assert len(ids) == len(names)
        self.embedder = embedder
        self.ids = ids
        self.names = names
        self.cache_name = cache_name
        self.embeddings: Optional[np.ndarray] = None
        self._cache_path = embedder.cache_dir / f"{cache_name}.npz"

    def build(self, force: bool = False) -> None:
        if self._cache_path.exists() and not force:
            data = np.load(self._cache_path, allow_pickle=True)
            self.embeddings = data["embeddings"].astype(np.float32)
            # sanity check
            if self.embeddings.shape[0] == len(self.names):
                logger.info(f"Loaded cached embeddings: {self.cache_name} ({self.embeddings.shape})")
                return
            else:
                logger.info(f"Cache size mismatch; rebuilding {self.cache_name}")

        logger.info(f"Encoding {len(self.names)} {self.cache_name} terms ...")
        self.embeddings = self.embedder.encode(self.names)
        np.savez(self._cache_path, embeddings=self.embeddings)

    def match(self, texts: List[str], top_k: int = 3) -> List[List[Tuple[str, str, float]]]:
        if self.embeddings is None:
            self.build()
        qembs = self.embedder.encode(texts)
        sims = qembs @ self.embeddings.T  # (Q, N), normalized vectors → dot product = cosine

        out = []
        for i in range(len(texts)):
            row = sims[i]
            idx = np.argsort(row)[-top_k:][::-1]
            out.append([(self.ids[j], self.names[j], float(row[j])) for j in idx])
        return out

    def match_one(self, text: str, threshold: float = 0.75) -> Optional[Tuple[str, str, float]]:
        matches = self.match([text], top_k=1)[0]
        if matches and matches[0][2] >= threshold:
            return matches[0]
        return None
