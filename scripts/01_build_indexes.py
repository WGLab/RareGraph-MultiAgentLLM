#!/usr/bin/env python
"""Build BioLORD embedding caches and KG precomputed indexes.

Run this once after downloading HPO and placing the KG JSON file at the
path specified in your config.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from raregraph.core.config import load_config
from raregraph.core.logging import setup_logger

from raregraph.normalize.biolord_embedder import BioLordEmbedder
from raregraph.normalize.hpo_ontology import HpoOntology
from raregraph.normalize.normalizers import HpoNormalizer
from raregraph.normalize.mondo_normalizer import MondoNormalizer

from raregraph.kg.kg_loader import load_kg
from raregraph.kg.kg_precompute import precompute_kg_index


def main() -> int:
    parser = argparse.ArgumentParser(description="Build RareGraph indexes")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--force", action="store_true", help="Rebuild caches from scratch")
    parser.add_argument("--skip-kg", action="store_true", help="Skip KG precomputation")
    args = parser.parse_args()

    logger = setup_logger(level=logging.INFO)
    cfg = load_config(args.config)

    # 1. HPO ontology
    hpo = HpoOntology(cfg.paths.hpo_obo)

    # 2. Embedder
    embedder = BioLordEmbedder(
        model_name=cfg.normalization.embed_model,
        cache_dir=cfg.paths.cache_dir,
    )

    # 3. HPO embedding cache
    hpo_norm = HpoNormalizer(hpo, embedder, similarity_threshold=cfg.normalization.similarity_threshold)
    hpo_norm.build_index(force=args.force)

    # 4. Mondo embedding cache (if full mondo file provided)
    if cfg.paths.full_mondo:
        mondo = MondoNormalizer(cfg.paths.full_mondo, embedder,
                                similarity_threshold=cfg.normalization.similarity_threshold)
        mondo.load()
        if mondo.id_to_name:
            mondo.build_index(force=args.force)

    # 5. KG precomputation
    if not args.skip_kg and cfg.paths.kg_path and Path(cfg.paths.kg_path).exists():
        kg = load_kg(cfg.paths.kg_path)
        hpo.compute_ic_from_kg(kg)
        idx = precompute_kg_index(kg, hpo)
        logger.info(
            f"KG precompute done: {len(idx.disease_name)} diseases, "
            f"{len(idx.pair_frequency)} pairs, "
            f"{len(idx.pathognomonic_hpos)} pathognomonic HPOs"
        )
    else:
        logger.warning("KG path not found; skipping KG precompute.")

    logger.info("Indexes built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
