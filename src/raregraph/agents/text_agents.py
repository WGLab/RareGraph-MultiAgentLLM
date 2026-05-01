"""Stage 1 text extraction agents.

All extractors share the same pattern:
  - read the prompt from configs/prompts/extraction/
  - prepend the context-flagged patient note
  - call VllmClient.chat_batch
  - parse output as JSON (list or dict depending on extractor)
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

from raregraph.core.json_utils import safe_json_load
from raregraph.core.utils import read_prompt
from raregraph.llm.vllm_client import VllmClient
from raregraph.agents.context_flags import add_context_flags

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------
def _run_extractor(
    llm: VllmClient,
    notes: List[str],
    prompt_path: Path,
    prefer: str = "any",
    apply_context_flags: bool = True,
) -> List[Any]:
    prompt_tpl = read_prompt(prompt_path)

    users = []
    for note in notes:
        note_for_llm = add_context_flags(note) if apply_context_flags else note
        users.append(f"{prompt_tpl}\n\n{note_for_llm}")

    # Use "extraction" task so the cleaner tries harder to return structured JSON.
    raw_outs = llm.chat_batch(system="", users=users, task="extraction")

    parsed = []
    for r in raw_outs:
        if isinstance(r, (list, dict)):
            parsed.append(r)
        else:
            parsed.append(safe_json_load(r, prefer=prefer))
    return parsed


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
    return [o if isinstance(o, list) else [] for o in outs]


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
