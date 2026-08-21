"""Stage 10: normalized next-test and next-step recommendations.

The final disease ranking is collapsed to disease groups and restricted to
groups represented within the global Top-K.  Actions are then collected from
RareGraph ``testing`` and ``initial_evaluations`` fields, normalized first
with deterministic lexical rules and then clustered with BioLORD embeddings.

Recommendations are ranked by the number of distinct candidate disease groups
that support them, followed by the best supporting final disease rank.  This
is the production counterpart of the next-test concordance analysis; it does
not require or inspect a reference/ground-truth next-test list.
"""
from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


ACTION_REPLACEMENTS: tuple[tuple[str, str], ...] = (
    (r"\b6[ -]?minute walk(?:ing)? test\b", "six minute walk test"),
    (r"\b6mwt\b", "six minute walk test"),
    (r"\becho(?:cardiography)?\b", "echocardiogram"),
    (r"\bb[- ]?type natriuretic peptide\b", "bnp"),
    (r"\bbrain natriuretic peptide\b", "bnp"),
    (r"\bmagnetic resonance imaging\b", "mri"),
    (r"\bcomputed tomography\b", "ct"),
    (r"\bwhole[- ]exome sequencing\b", "exome sequencing"),
    (r"\bwhole[- ]genome sequencing\b", "genome sequencing"),
)

RAW_EVIDENCE_COLUMNS = [
    "group_id",
    "group_name",
    "representative_disease_id",
    "representative_disease_name",
    "group_final_rank",
    "node_id",
    "action",
    "action_norm",
    "field",
]

RECOMMENDATION_COLUMNS = [
    "action_rank",
    "action",
    "action_norm",
    "action_aliases",
    "support_groups",
    "best_supporting_group_rank",
    "fields",
    "supporting_group_ids",
    "supporting_group_names",
    "supporting_disease_ids",
    "supporting_disease_names",
]


def canonical_action(text: Any) -> str:
    """Return a conservative lexical canonical form for a clinical action."""
    value = str(text or "").lower().strip().replace("&", " and ")
    for pattern, replacement in ACTION_REPLACEMENTS:
        value = re.sub(pattern, replacement, value)
    value = re.sub(r"[^a-z0-9+/-]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def extract_action_names(node: Any, fields: Iterable[str]) -> list[dict[str, str]]:
    """Extract action labels from the flexible RareGraph node schema."""
    if not isinstance(node, dict):
        return []
    meta = node.get("meta", {}) if isinstance(node.get("meta"), dict) else {}
    actions: list[dict[str, str]] = []
    for field in fields:
        block = node.get(field, meta.get(field))
        if isinstance(block, dict):
            names = list(block.keys())
        elif isinstance(block, list):
            names = [
                item.get("name") or item.get("test") or item.get("evaluation")
                if isinstance(item, dict)
                else item
                for item in block
            ]
        elif isinstance(block, str):
            names = [part for part in re.split(r"[\n;•]+", block) if part.strip()]
        else:
            names = []
        for name in names:
            normalized = canonical_action(name)
            if name and normalized:
                actions.append({"action": str(name).strip(), "field": str(field)})
    return actions


def collapse_to_top_groups(final_df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """Select one best-ranked subtype for every group represented in Top-K."""
    if final_df is None or final_df.empty:
        return pd.DataFrame()
    if "disease_id" not in final_df.columns:
        raise ValueError("Stage 10 requires a disease_id column")

    groups = final_df.copy()
    if "final_rank" not in groups.columns:
        fallback_rank = next(
            (
                column for column in (
                    "reconciled_rank", "reranked_rank_subtype", "adjusted_rank", "rank"
                ) if column in groups.columns
            ),
            None,
        )
        if fallback_rank is None:
            raise ValueError("Stage 10 could not find a final or fallback rank column")
        groups["final_rank"] = groups[fallback_rank]
    groups["final_rank"] = pd.to_numeric(groups["final_rank"], errors="coerce")
    groups = groups[groups["final_rank"].notna() & groups["final_rank"].le(int(top_k))].copy()
    if groups.empty:
        return groups

    def clean_series(name: str, fallback: str) -> pd.Series:
        if name in groups.columns:
            values = groups[name].astype("string").fillna("").str.strip()
        else:
            values = pd.Series([""] * len(groups), index=groups.index, dtype="string")
        if fallback in groups.columns:
            fallback_values = groups[fallback].astype("string").fillna("").str.strip()
        else:
            fallback_values = groups["disease_id"].astype("string").fillna("").str.strip()
        return values.where(values.ne(""), fallback_values)

    groups["group_id"] = clean_series("group_id", "disease_id")
    groups["group_name"] = clean_series("group_name", "disease_name")
    groups = (
        groups.sort_values(["final_rank", "disease_id"], ascending=[True, True])
        .drop_duplicates("group_id", keep="first")
        .reset_index(drop=True)
    )
    return groups


def collect_group_actions(
    groups: pd.DataFrame,
    kg: dict[str, dict[str, Any]],
    fields: Iterable[str],
) -> pd.DataFrame:
    """Collect unique KG actions per selected disease group with provenance."""
    rows: list[dict[str, Any]] = []
    if groups is None or groups.empty:
        return pd.DataFrame(columns=RAW_EVIDENCE_COLUMNS)

    for group in groups.to_dict("records"):
        gid = str(group.get("group_id") or group.get("disease_id") or "").strip()
        did = str(group.get("disease_id") or "").strip()
        seen_for_group: set[str] = set()
        for node_id in dict.fromkeys([did, gid]):
            if not node_id:
                continue
            for action in extract_action_names(kg.get(node_id, {}), fields):
                key = canonical_action(action["action"])
                if not key or key in seen_for_group:
                    continue
                seen_for_group.add(key)
                rows.append({
                    "group_id": gid,
                    "group_name": str(group.get("group_name") or gid),
                    "representative_disease_id": did,
                    "representative_disease_name": str(group.get("disease_name") or did),
                    "group_final_rank": float(group.get("final_rank")),
                    "node_id": node_id,
                    "action": action["action"],
                    "action_norm": key,
                    "field": action["field"],
                })
    return pd.DataFrame(rows, columns=RAW_EVIDENCE_COLUMNS)


def _semantic_clusters(
    labels: Sequence[str],
    embedder: Any | None,
    threshold: float,
) -> tuple[list[list[str]], str]:
    """Cluster exact-normalized labels with BioLORD cosine similarity."""
    labels = list(labels)
    if not labels:
        return [], "lexical"
    if embedder is None or len(labels) == 1:
        return [[label] for label in labels], "lexical"

    try:
        matrix = np.asarray(embedder.encode(labels), dtype=np.float32)
        similarities = matrix @ matrix.T
    except Exception as exc:  # Keep the clinical pipeline usable if embedding fails.
        logger.warning("Stage 10 BioLORD clustering failed; using lexical normalization: %s", exc)
        return [[label] for label in labels], "lexical_fallback"

    parent = list(range(len(labels)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left in range(len(labels)):
        for right in np.flatnonzero(similarities[left, left + 1 :] >= float(threshold)):
            union(left, left + 1 + int(right))

    clusters: dict[int, list[str]] = defaultdict(list)
    for index, label in enumerate(labels):
        clusters[find(index)].append(label)
    return list(clusters.values()), "lexical+biolord"


def summarize_actions(
    raw: pd.DataFrame,
    embedder: Any | None,
    cluster_threshold: float = 0.90,
    top_n: int = 10,
) -> tuple[pd.DataFrame, str]:
    """Normalize, semantically cluster, and rank cross-diagnosis actions."""
    if raw is None or raw.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS), "lexical"

    support = (
        raw.groupby("action_norm", as_index=False)
        .agg(
            action=("action", "first"),
            support_groups=("group_id", "nunique"),
            best_supporting_group_rank=("group_final_rank", "min"),
            fields=("field", lambda values: "|".join(sorted(set(values)))),
            supporting_group_ids=("group_id", lambda values: "|".join(sorted(set(values)))),
            supporting_group_names=("group_name", lambda values: "|".join(sorted(set(values)))),
            supporting_disease_ids=(
                "representative_disease_id", lambda values: "|".join(sorted(set(values)))
            ),
            supporting_disease_names=(
                "representative_disease_name", lambda values: "|".join(sorted(set(values)))
            ),
        )
    )
    clusters, method = _semantic_clusters(
        support["action_norm"].tolist(), embedder, cluster_threshold
    )

    records: list[dict[str, Any]] = []
    for members in clusters:
        part = support[support["action_norm"].isin(members)].copy()
        representative = part.sort_values(
            ["support_groups", "best_supporting_group_rank", "action_norm"],
            ascending=[False, True, True],
        ).iloc[0]

        def union_pipe(column: str) -> str:
            values: set[str] = set()
            for cell in part[column].fillna(""):
                values.update(piece for piece in str(cell).split("|") if piece)
            return "|".join(sorted(values))

        group_ids = union_pipe("supporting_group_ids")
        records.append({
            "action": representative["action"],
            "action_norm": representative["action_norm"],
            "action_aliases": "|".join(sorted(set(members))),
            "support_groups": len([value for value in group_ids.split("|") if value]),
            "best_supporting_group_rank": float(part["best_supporting_group_rank"].min()),
            "fields": union_pipe("fields"),
            "supporting_group_ids": group_ids,
            "supporting_group_names": union_pipe("supporting_group_names"),
            "supporting_disease_ids": union_pipe("supporting_disease_ids"),
            "supporting_disease_names": union_pipe("supporting_disease_names"),
        })

    recommendations = (
        pd.DataFrame(records)
        .sort_values(
            ["support_groups", "best_supporting_group_rank", "action_norm"],
            ascending=[False, True, True],
        )
        .head(int(top_n))
        .reset_index(drop=True)
    )
    recommendations.insert(0, "action_rank", np.arange(1, len(recommendations) + 1, dtype=int))
    return recommendations[RECOMMENDATION_COLUMNS], method


def build_next_step_recommendations(
    final_df: pd.DataFrame,
    kg: dict[str, dict[str, Any]],
    embedder: Any | None,
    disease_top_k: int = 10,
    action_top_k: int = 10,
    fields: Iterable[str] = ("testing", "initial_evaluations"),
    cluster_threshold: float = 0.90,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build Stage 10 outputs from the final RareMind disease ranking."""
    if isinstance(fields, str):
        fields = (fields,)
    fields = tuple(str(field) for field in fields)
    groups = collapse_to_top_groups(final_df, disease_top_k)
    raw = collect_group_actions(groups, kg, fields)
    recommendations, method = summarize_actions(
        raw,
        embedder=embedder,
        cluster_threshold=cluster_threshold,
        top_n=action_top_k,
    )
    metadata = {
        "disease_top_k": int(disease_top_k),
        "action_top_k": int(action_top_k),
        "source_fields": list(fields),
        "selected_disease_groups": int(len(groups)),
        "raw_action_mentions": int(len(raw)),
        "normalization_method": method,
        "embedding_model": getattr(embedder, "model_name", None) if embedder is not None else None,
        "cluster_similarity_threshold": float(cluster_threshold),
    }
    return recommendations, raw, groups, metadata
