"""Best-effort JSON parsing for LLM outputs.

Small LLMs often emit:
- Thinking blocks wrapped in <think>...</think>
- JSON wrapped in markdown code fences (```json ... ```)
- Minor syntax errors (trailing commas, single quotes)
- Valid JSON with text prefix/suffix

These helpers extract usable JSON where possible, with a graceful fallback
that preserves the raw text under a "raw" key so debugging is easy.
"""
from __future__ import annotations

import json
import re
from typing import Any

import orjson


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", flags=re.DOTALL)


def strip_thinking(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_fenced(text: str) -> str:
    """If text has ```json ... ``` fences, return the inside; else text."""
    m = _FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # Handle common truncated output that starts a fence but never closes it.
    return re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE).strip()


def _first_balanced_json_prefix(s: str, prefer: str = "any") -> str | None:
    """Return the first balanced JSON object/array prefix, if present."""
    starts = []
    if prefer in ("any", "object"):
        idx = s.find("{")
        if idx != -1:
            starts.append((idx, "{", "}"))
    if prefer in ("any", "array"):
        idx = s.find("[")
        if idx != -1:
            starts.append((idx, "[", "]"))
    if not starts:
        return None
    start, opener, _ = min(starts, key=lambda x: x[0])

    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None
            stack.pop()
            if not stack:
                return s[start : i + 1]
    return None


def _repair_truncated_json(s: str, prefer: str = "any") -> str | None:
    """Best-effort repair for JSON truncated after a mostly-valid prefix."""
    starts = []
    if prefer in ("any", "object"):
        idx = s.find("{")
        if idx != -1:
            starts.append(idx)
    if prefer in ("any", "array"):
        idx = s.find("[")
        if idx != -1:
            starts.append(idx)
    if not starts:
        return None
    start = min(starts)

    pairs = {"{": "}", "[": "]"}
    stack: list[str] = []
    in_str = False
    escape = False
    out = []
    for ch in s[start:]:
        out.append(ch)
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch in pairs:
            stack.append(pairs[ch])
        elif ch in "}]":
            if stack and stack[-1] == ch:
                stack.pop()
            else:
                out.pop()
                break

    if in_str:
        # Drop a dangling escape before closing a truncated string.
        if out and out[-1] == "\\":
            out.pop()
        out.append('"')
    candidate = "".join(out)
    candidate = re.sub(r",(\s*)$", r"\1", candidate)
    while stack:
        candidate = re.sub(r",(\s*)$", r"\1", candidate)
        candidate += stack.pop()
    return candidate


def safe_json_load(s: str, prefer: str = "any") -> Any:
    """Best-effort JSON parse.

    Parameters
    ----------
    s : raw LLM output
    prefer : "any" | "object" | "array"
        If "object", prefers the largest { ... } block.
        If "array",  prefers the largest [ ... ] block.
    """
    if not isinstance(s, str):
        return s

    s = strip_thinking(s)
    s = extract_fenced(s)
    s = s.strip()

    # direct parse
    for loader in (orjson.loads, json.loads):
        try:
            return loader(s)
        except Exception:
            pass

    # parse first complete object/array before any repetitive tail
    prefix = _first_balanced_json_prefix(s, prefer=prefer)
    if prefix:
        for loader in (orjson.loads, json.loads):
            try:
                return loader(prefix)
            except Exception:
                pass

    # extract object substring
    if prefer in ("any", "object"):
        start = s.find("{")
        end = s.rfind("}") + 1
        if start != -1 and end > start:
            sub = s[start:end]
            for loader in (orjson.loads, json.loads):
                try:
                    return loader(sub)
                except Exception:
                    pass

    # extract array substring
    if prefer in ("any", "array"):
        start = s.find("[")
        end = s.rfind("]") + 1
        if start != -1 and end > start:
            sub = s[start:end]
            for loader in (orjson.loads, json.loads):
                try:
                    return loader(sub)
                except Exception:
                    pass

    # last-resort cleanup
    cleaned = re.sub(r",(\s*[}\]])", r"\1", s)  # remove trailing commas
    for loader in (orjson.loads, json.loads):
        try:
            return loader(cleaned)
        except Exception:
            pass

    repaired = _repair_truncated_json(s, prefer=prefer)
    if repaired:
        repaired = re.sub(r",(\s*[}\]])", r"\1", repaired)
        for loader in (orjson.loads, json.loads):
            try:
                return loader(repaired)
            except Exception:
                pass

    return {"raw": s}


def validate_quote(quote: str, source_text: str, case_insensitive: bool = True,
                   whitespace_tolerant: bool = True) -> bool:
    """Check whether a claimed verbatim quote actually appears in a source text.

    Used to filter hallucinated supporting/contradicting evidence in the audit.
    """
    if not quote or not source_text:
        return False

    q = quote.strip()
    src = source_text

    if whitespace_tolerant:
        q = re.sub(r"\s+", " ", q)
        src = re.sub(r"\s+", " ", src)

    if case_insensitive:
        return q.lower() in src.lower()
    return q in src
