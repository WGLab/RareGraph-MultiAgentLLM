"""Context flag preprocessor for clinical notes.

Annotates sentences with inline flags that help small LLMs correctly attribute
extracted phenotypes. Each sentence is prefixed with 0 or more of:
  [NEG], [FAMILY], [HYPOTHETICAL], [HISTORICAL]

This is a cheap regex pass that runs BEFORE the LLM extractor.
"""
from __future__ import annotations

import re
from typing import List

NEG_TRIGGERS = [
    r"\bno\b", r"\bnot\b", r"\bdenies\b", r"\bdenied\b", r"\bwithout\b",
    r"\bnegative for\b", r"\babsent\b", r"\bunremarkable\b",
    r"\bwithin normal limits\b", r"\bwnl\b", r"\bnormal\b",
    r"\bresolved\b", r"\brepaired\b", r"\bcorrected\b",
    r"\bno evidence of\b",
]
FAMILY_TRIGGERS = [
    r"\bmother\b", r"\bmothers?\b", r"\bmom\b",
    r"\bfather\b", r"\bdad\b", r"\bpater(nal)?\b", r"\bmater(nal)?\b",
    r"\bsister\b", r"\bbrother\b", r"\bsibling\b",
    r"\baunt\b", r"\buncle\b", r"\bcousin\b",
    r"\bgrand(mother|father|parent|ma|pa)\b",
    r"\bfamily history\b", r"\bfamilial\b",
    r"\brelative\b", r"\brelatives\b",
    r"\bson\b", r"\bdaughter\b",
]
HYPOTHETICAL_TRIGGERS = [
    r"\bif\b", r"\bpossible\b", r"\bpossibly\b",
    r"\bconcern for\b", r"\bconcerns? for\b",
    r"\bmay have\b", r"\bmight have\b",
    r"\bsuspected\b", r"\bsuspicion\b",
    r"\bcannot exclude\b", r"\brule out\b", r"\br/o\b",
    r"\bdifferential includes\b",
    r"\buncertain\b",
]
HISTORICAL_TRIGGERS = [
    r"\bhistory of\b", r"\bh/o\b",
    r"\bpreviously\b", r"\bin the past\b", r"\bpast medical\b",
    r"\bs/p\b", r"\bstatus post\b", r"\bprior\b",
]


def _compile_union(triggers: List[str]) -> re.Pattern:
    return re.compile("|".join(triggers), flags=re.IGNORECASE)


NEG_RE = _compile_union(NEG_TRIGGERS)
FAMILY_RE = _compile_union(FAMILY_TRIGGERS)
HYPOTHETICAL_RE = _compile_union(HYPOTHETICAL_TRIGGERS)
HISTORICAL_RE = _compile_union(HISTORICAL_TRIGGERS)

SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def flag_sentence(sent: str) -> str:
    flags = []
    if NEG_RE.search(sent):
        flags.append("[NEG]")
    if FAMILY_RE.search(sent):
        flags.append("[FAMILY]")
    if HYPOTHETICAL_RE.search(sent):
        flags.append("[HYPOTHETICAL]")
    if HISTORICAL_RE.search(sent):
        flags.append("[HISTORICAL]")
    if flags:
        return "".join(flags) + " " + sent.strip()
    return sent.strip()


def add_context_flags(text: str) -> str:
    """Return note text with per-sentence context flags prepended."""
    if not text or not isinstance(text, str):
        return text or ""

    # Naive sentence splitter; good enough for flagging.
    sentences = SENTENCE_SPLIT_RE.split(text.strip())
    flagged = [flag_sentence(s) for s in sentences if s.strip()]
    return " ".join(flagged)
