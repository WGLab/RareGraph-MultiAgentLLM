"""Stage 7: Rank centrality / PageRank aggregation.

Takes the list of pairwise verdicts and produces a final ranking.
Blends PageRank score with normalized weighted wins, then performs a local
swap refinement for close direct wins.
"""
from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


STRENGTH_WEIGHT = {"strong": 1.0, "moderate": 0.7, "weak": 0.4}


def build_win_loss_graph(
    pairwise_results: List[Dict[str, Any]],
    candidates: List[str],
) -> Dict[str, Dict[str, float]]:
    """Weighted edges: winner <- loser; weight = strength."""
    graph: Dict[str, Dict[str, float]] = {c: defaultdict(float) for c in candidates}
    for r in pairwise_results:
        a = r["disease_a_id"]
        b = r["disease_b_id"]
        w = r.get("winner", "tie")
        s = STRENGTH_WEIGHT.get(r.get("strength", "weak"), 0.4)
        if w == "A":
            graph[b][a] += s  # loser b -> winner a
        elif w == "B":
            graph[a][b] += s
        # ties contribute nothing
    return graph


def pagerank(
    graph: Dict[str, Dict[str, float]],
    damping: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-10,
    prior: Dict[str, float] | None = None,
) -> Dict[str, float]:
    # Defensive coercion: YAML may parse "1e-10" as str; force numeric types.
    damping = float(damping)
    max_iter = int(max_iter)
    tol = float(tol)

    nodes = list(graph.keys())
    n = len(nodes)
    if n == 0:
        return {}

    idx = {node: i for i, node in enumerate(nodes)}

    # Prior
    if prior:
        p = np.array([prior.get(node, 1.0 / n) for node in nodes])
    else:
        p = np.full(n, 1.0 / n)
    p = p / p.sum()

    # Transition matrix: column-stochastic, where col i is the outgoing distribution from i
    M = np.zeros((n, n), dtype=np.float64)
    for node, edges in graph.items():
        i = idx[node]
        total = sum(edges.values())
        if total <= 0:
            continue
        for target, w in edges.items():
            j = idx[target]
            M[j, i] = w / total  # probability mass from i -> j

    # Handle dangling (zero-sum columns): distribute to prior
    col_sums = M.sum(axis=0)
    for i in range(n):
        if col_sums[i] == 0:
            M[:, i] = p  # dangling node follows prior

    # Power iteration
    scores = p.copy()
    for _ in range(max_iter):
        new = damping * (M @ scores) + (1 - damping) * p
        new = new / new.sum()
        if np.linalg.norm(new - scores, ord=1) < tol:
            scores = new
            break
        scores = new

    return {node: float(scores[idx[node]]) for node in nodes}


def aggregate_rank(
    ranked_df: pd.DataFrame,
    pairwise_results: List[Dict[str, Any]],
    cfg: Any,
    track_name: str = "subtype",
) -> pd.DataFrame:
    """Produce a reranked DataFrame using rank centrality + weighted wins blend."""
    candidates = ranked_df.head(cfg.pairwise.top_n)["disease_id"].tolist()
    if not candidates or not pairwise_results:
        df = ranked_df.copy()
        df[f"reranked_rank_{track_name}"] = df.get("adjusted_rank", df.index + 1)
        df[f"reranked_score_{track_name}"] = df.get("adjusted_score", df.get("total_score", 0.0))
        return df

    # Build graph
    graph = build_win_loss_graph(pairwise_results, candidates)

    # Use original ranking as prior
    prior: Dict[str, float] = {}
    for _, row in ranked_df.iterrows():
        if row["disease_id"] in graph:
            # Inverse rank prior (higher composite = higher prior)
            prior[row["disease_id"]] = 1.0 / max(1.0, float(row.get("adjusted_rank", row.get("rank", 1))))

    pr_scores = pagerank(
        graph,
        damping=cfg.rank_aggregation.damping,
        max_iter=cfg.rank_aggregation.max_iter,
        tol=cfg.rank_aggregation.tol,
        prior=prior,
    )

    # Weighted wins
    weighted_wins: Dict[str, float] = defaultdict(float)
    for r in pairwise_results:
        w = r.get("winner", "tie")
        s = STRENGTH_WEIGHT.get(r.get("strength", "weak"), 0.4)
        if w == "A":
            weighted_wins[r["disease_a_id"]] += s
        elif w == "B":
            weighted_wins[r["disease_b_id"]] += s

    # Normalize weighted wins
    if weighted_wins:
        max_w = max(weighted_wins.values())
        if max_w > 0:
            for k in weighted_wins:
                weighted_wins[k] = weighted_wins[k] / max_w

    # Normalize pagerank
    if pr_scores:
        max_pr = max(pr_scores.values())
        if max_pr > 0:
            pr_scores = {k: v / max_pr for k, v in pr_scores.items()}

    # Blend (50/50)
    blended: Dict[str, float] = {}
    for c in candidates:
        blended[c] = 0.5 * pr_scores.get(c, 0.0) + 0.5 * weighted_wins.get(c, 0.0)

    # Apply to ranked_df (only to the subset that went through pairwise)
    df = ranked_df.copy()
    pw_score_col = f"reranked_score_{track_name}"
    df[pw_score_col] = df["disease_id"].map(blended).astype(float)

    # Compose final score: audit-adjusted score + boost from pairwise
    base_col = "adjusted_score" if "adjusted_score" in df.columns else "total_score"
    # We normalize base scores to [0,1]
    base_values = df[base_col].values.astype(float)
    if base_values.max() > 0:
        base_norm = base_values / base_values.max()
    else:
        base_norm = base_values
    df["_base_norm"] = base_norm

    # Only candidates in the pairwise set get the pairwise contribution
    df["_pw_contrib"] = df[pw_score_col].fillna(0.0)

    df["final_score_" + track_name] = 0.6 * df["_base_norm"] + 0.4 * df["_pw_contrib"]
    df = df.sort_values("final_score_" + track_name, ascending=False).reset_index(drop=True)
    df[f"reranked_rank_{track_name}"] = df.index + 1

    # Local swap refinement for close direct-win pairs
    df = _local_swap_refinement(df, pairwise_results, track_name, cfg)

    df.drop(columns=["_base_norm", "_pw_contrib"], inplace=True, errors="ignore")
    return df


def _local_swap_refinement(
    df: pd.DataFrame,
    pairwise_results: List[Dict[str, Any]],
    track_name: str,
    cfg: Any,
) -> pd.DataFrame:
    """For each adjacent pair within close_margin, honor direct pairwise verdicts."""
    score_col = "final_score_" + track_name
    margin = cfg.rank_aggregation.close_margin

    verdict_map: Dict[tuple, str] = {}
    for r in pairwise_results:
        key = tuple(sorted([r["disease_a_id"], r["disease_b_id"]]))
        verdict_map[key] = (r["disease_a_id"], r["disease_b_id"], r.get("winner"))

    df = df.reset_index(drop=True)
    swapped = True
    passes = 0
    while swapped and passes < 5:
        swapped = False
        for i in range(len(df) - 1):
            a = df.iloc[i]
            b = df.iloc[i + 1]
            if abs(a[score_col] - b[score_col]) > margin:
                continue
            key = tuple(sorted([a["disease_id"], b["disease_id"]]))
            v = verdict_map.get(key)
            if not v:
                continue
            a_id, b_id, winner = v
            # If the direct verdict says the lower-ranked one should win, swap.
            if winner == "A" and b["disease_id"] == a_id:
                df.iloc[[i, i + 1]] = df.iloc[[i + 1, i]].values
                swapped = True
            elif winner == "B" and b["disease_id"] == b_id:
                df.iloc[[i, i + 1]] = df.iloc[[i + 1, i]].values
                swapped = True
        passes += 1

    df[f"reranked_rank_{track_name}"] = range(1, len(df) + 1)
    return df
