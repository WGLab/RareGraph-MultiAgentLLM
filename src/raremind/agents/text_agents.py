"""Stage 1 text extraction agents.

All extractors share the same pattern:
  - read the prompt from configs/prompts/extraction/
  - prepend the context-flagged patient note
  - call VllmClient.chat_batch
  - parse output as JSON (list or dict depending on extractor)
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

from raregraph.core.json_utils import safe_json_load
from raregraph.core.utils import read_prompt
from raregraph.llm.vllm_client import VllmClient
from raregraph.agents.context_flags import add_context_flags

logger = logging.getLogger(__name__)

PHENOTYPE_ATTRIBUTIONS = {
    "patient",
    "negated",
    "family",
    "references",
    "others",
    "uncertain",
}


def _dedup_key(text: Any) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip().lower())


def _normalize_phenotype_items(items: Any) -> List[Dict[str, Any]]:
    """Keep minimally valid phenotype dicts and fill missing safe defaults."""
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        mention = str(item.get("mention") or "").strip()
        if not mention:
            continue
        attr = str(item.get("attribution") or "").strip().lower()
        if attr not in PHENOTYPE_ATTRIBUTIONS:
            attr = "uncertain"
        key = (_dedup_key(mention), attr)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "mention": mention,
            "attribution": attr,
            "onset": item.get("onset"),
            "evidence": item.get("evidence"),
        })
    return out


# ---------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------
def _run_extractor(
    llm: VllmClient,
    notes: List[str],
    prompt_path: Path,
    prefer: str = "any",
    apply_context_flags: bool = True,
    return_diagnostics: bool = False,
) -> List[Any]:
    prompt_tpl = read_prompt(prompt_path)

    users = []
    for note in notes:
        note_for_llm = add_context_flags(note) if apply_context_flags else note
        users.append(f"{prompt_tpl}\n\n{note_for_llm}")

    # Use "extraction" task so the cleaner tries harder to return structured JSON.
    raw_outs = llm.chat_batch(system="", users=users, task="extraction")

    parsed = []
    diagnostics = []
    for r in raw_outs:
        value, diag = _parse_extractor_output(r, prefer=prefer)
        parsed.append(value)
        diagnostics.append(diag)
    if return_diagnostics:
        return list(zip(parsed, diagnostics))
    return parsed


def _parse_extractor_output(raw: Any, prefer: str = "any") -> Tuple[Any, Dict[str, Any]]:
    """Parse one extractor response and record whether repair was needed.

    VllmClient.clean_output may already return {"raw": "..."} for extraction
    failures. Re-run safe_json_load on that raw text so its richer repair path
    is actually attempted before downstream code collapses the output to [].
    """
    diag: Dict[str, Any] = {
        "raw_type": type(raw).__name__,
        "parse_repaired": False,
        "parse_failed": False,
    }

    if isinstance(raw, dict) and "raw" in raw:
        repaired = safe_json_load(raw.get("raw", ""), prefer=prefer)
        diag["parse_repaired"] = not (
            isinstance(repaired, dict) and "raw" in repaired
        )
        diag["parse_failed"] = not diag["parse_repaired"]
        diag["raw_excerpt"] = str(raw.get("raw", ""))[:500]
        diag["parsed_type"] = type(repaired).__name__
        return repaired, diag

    if isinstance(raw, (list, dict)):
        diag["parsed_type"] = type(raw).__name__
        return raw, diag

    parsed = safe_json_load(raw, prefer=prefer)
    diag["parse_failed"] = isinstance(parsed, dict) and "raw" in parsed
    diag["parsed_type"] = type(parsed).__name__
    if diag["parse_failed"]:
        diag["raw_excerpt"] = str(raw)[:500]
    return parsed, diag


def run_stage1_text_extractors_batch(
    llm: VllmClient,
    notes: List[str],
    prompt_dir: str,
) -> List[Dict[str, Any]]:
    """Run all Stage 1 text extractors and return outputs plus diagnostics."""
    extractor_specs = [
        ("text_phenotypes", "text_phenotype_extractor.md", "array"),
        ("demographics", "text_demographics_extractor.md", "object"),
        ("family_history", "text_family_history_extractor.md", "array"),
        ("testing", "text_testing_extractor.md", "array"),
        ("gene_mentions", "text_gene_mentions_extractor.md", "array"),
    ]
    by_field: Dict[str, List[Any]] = {}
    by_field_diag: Dict[str, List[Dict[str, Any]]] = {}

    for field, filename, prefer in extractor_specs:
        path = Path(prompt_dir) / "extraction" / filename
        pairs = _run_extractor(
            llm,
            notes,
            path,
            prefer=prefer,
            return_diagnostics=True,
        )
        values, diags = zip(*pairs) if pairs else ([], [])
        by_field[field] = list(values)
        by_field_diag[field] = list(diags)

    out: List[Dict[str, Any]] = []
    for i in range(len(notes)):
        demo = by_field["demographics"][i]
        if not (isinstance(demo, dict) and "raw" not in demo):
            demo = {
                "age": {"value": None, "age_group": "unknown"},
                "sex": {"value": "unknown"},
                "ethnicity": {"value": None},
            }
        row = {
            "text_phenotypes": _normalize_phenotype_items(by_field["text_phenotypes"][i]),
            "demographics": demo,
            "family_history": by_field["family_history"][i]
            if isinstance(by_field["family_history"][i], list) else [],
            "testing": by_field["testing"][i]
            if isinstance(by_field["testing"][i], list) else [],
            "gene_mentions": by_field["gene_mentions"][i]
            if isinstance(by_field["gene_mentions"][i], list) else [],
            "extraction_diagnostics": {
                field: by_field_diag[field][i] for field, _, _ in extractor_specs
            },
        }
        out.append(row)
    return out


# ---------------------------------------------------------------
# Public extractor API (one call per modality)
# ---------------------------------------------------------------
def run_phenotype_extractor_batch(
    llm: VllmClient,
    notes: List[str],
    prompt_dir: str,
) -> List[List[Dict[str, Any]]]:
    path = Path(prompt_dir) / "extraction" / "text_phenotype_extractor.md"
    outs = _run_extractor(llm, notes, path, prefer="array")
    return [_normalize_phenotype_items(o) for o in outs]


def run_demographics_extractor_batch(
    llm: VllmClient,
    notes: List[str],
    prompt_dir: str,
) -> List[Dict[str, Any]]:
    path = Path(prompt_dir) / "extraction" / "text_demographics_extractor.md"
    outs = _run_extractor(llm, notes, path, prefer="object")
    fixed = []
    for o in outs:
        if isinstance(o, dict) and "raw" not in o:
            fixed.append(o)
        else:
            fixed.append({"age": {"value": None, "age_group": "unknown"},
                          "sex": {"value": "unknown"},
                          "ethnicity": {"value": None}})
    return fixed


def run_family_history_extractor_batch(
    llm: VllmClient,
    notes: List[str],
    prompt_dir: str,
) -> List[List[Dict[str, Any]]]:
    path = Path(prompt_dir) / "extraction" / "text_family_history_extractor.md"
    outs = _run_extractor(llm, notes, path, prefer="array")
    return [o if isinstance(o, list) else [] for o in outs]


def run_testing_extractor_batch(
    llm: VllmClient,
    notes: List[str],
    prompt_dir: str,
) -> List[List[Dict[str, Any]]]:
    path = Path(prompt_dir) / "extraction" / "text_testing_extractor.md"
    outs = _run_extractor(llm, notes, path, prefer="array")
    return [o if isinstance(o, list) else [] for o in outs]


def run_gene_mentions_extractor_batch(
    llm: VllmClient,
    notes: List[str],
    prompt_dir: str,
) -> List[List[Dict[str, Any]]]:
    path = Path(prompt_dir) / "extraction" / "text_gene_mentions_extractor.md"
    outs = _run_extractor(llm, notes, path, prefer="array")
    return [o if isinstance(o, list) else [] for o in outs]
