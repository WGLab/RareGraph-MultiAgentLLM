"""Harmonized KG loader.

Loads the Mondo-keyed JSON knowledge graph your upstream harmonization
(OMIM + Orphanet + Mondo + HPO + GeneReviews) produces.

Expected per-disease schema (flexible):
  {
    "MONDO:xxx": {
      "preferred_title": "<disease name>",
      "name": "<legacy disease name>",
      "phenotypes": {
          "<hpo_name>": {
            "hpo": "HP:yyy",
            "importance": "characteristic|supportive|incidental",
            "frequency": "very_common|common|more_than_half|occasional|rare",
            "polarity": "present|absent",
            "is_predefined": true|false,
            "evidence": "..."
          },
          ...
      },
      "genes": { "GENE": {...}, ... },
      "inheritance": ["autosomal dominant", ...],
      "demographics": {...},
      "differentials": [{"disease": "...", "rule": "..."}, ...],
      "definition": "...",  # optional free-text
      "narrative": "...",   # optional free-text
      "alternative_titles": [...],
      "group": "<Mondo group id or name>",
      "aliases": [...]
    }
  }

The loader is lenient — missing fields are substituted with safe defaults.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def load_kg(kg_path: str) -> Dict[str, Dict[str, Any]]:
    path = Path(kg_path)
    if not path.exists():
        raise FileNotFoundError(f"KG not found: {kg_path}")
    logger.info(f"Loading KG from {path}")
    with open(path, "r", encoding="utf-8") as f:
        kg = json.load(f)
    logger.info(f"KG loaded: {len(kg)} diseases")
    return kg


def load_hierarchy(hierarchy_path: str) -> Dict[str, Any]:
    """Load the group → subtype hierarchy file.

    Expected format (flexible):
      {
        "MONDO:groupA": {"name": "...", "children": ["MONDO:sub1", "MONDO:sub2", ...]},
        ...
      }
    """
    path = Path(hierarchy_path)
    if not path.exists():
        logger.warning(f"Hierarchy file not found: {hierarchy_path}")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        h = json.load(f)
    return h or {}


def disease_name(kg: Dict[str, Any], disease_id: str) -> str:
    entry = kg.get(disease_id, {})
    if not isinstance(entry, dict):
        return disease_id
    meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}
    return (
        entry.get("preferred_title")
        or entry.get("name")
        or entry.get("label")
        or meta.get("preferred_title")
        or meta.get("name")
        or meta.get("label")
        or disease_id
    )


def disease_phenotypes(kg: Dict[str, Any], disease_id: str) -> Dict[str, Dict[str, Any]]:
    entry = kg.get(disease_id, {})
    p = entry.get("phenotypes", {})
    return p if isinstance(p, dict) else {}


def disease_genes(kg: Dict[str, Any], disease_id: str) -> Dict[str, Any]:
    entry = kg.get(disease_id, {})
    g = entry.get("genes", {})
    return g if isinstance(g, dict) else {}


def disease_aliases(kg: Dict[str, Any], disease_id: str) -> list[str]:
    entry = kg.get(disease_id, {})
    if not isinstance(entry, dict):
        return []
    meta = entry.get("meta", {}) if isinstance(entry.get("meta"), dict) else {}

    aliases: list[str] = []
    for block in (
        entry.get("alternative_titles"),
        entry.get("aliases"),
        entry.get("synonyms"),
        meta.get("synonyms"),
        meta.get("aliases"),
    ):
        if isinstance(block, str):
            aliases.extend([x.strip() for x in block.split("|") if x.strip()])
        elif isinstance(block, list):
            aliases.extend([str(x).strip() for x in block if str(x).strip()])

    seen = set()
    out = []
    for alias in aliases:
        key = alias.lower()
        if key not in seen:
            seen.add(key)
            out.append(alias)
    return out
