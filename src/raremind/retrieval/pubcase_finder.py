"""PubCaseFinder HPO-based disease similarity.

Real implementation adapted from the original rare_dx_mcp.
Queries https://pubcasefinder.dbcls.jp/api/get_ranked_list in both
'disease' and 'gene' modes, then maps OMIM/Orphanet IDs back to Mondo
via DiseaseIdMapper so results can be consumed by the composite ranker.
"""
from __future__ import annotations

import logging
from io import StringIO
from typing import Any, List, Optional, Union

import pandas as pd
import requests

try:
    from fake_useragent import UserAgent
except Exception:  # pragma: no cover
    UserAgent = None

logger = logging.getLogger(__name__)


BASE_URL = "https://pubcasefinder.dbcls.jp/api/pcf_get_ranked_list"


def _resolve_user_agent() -> str:
    if UserAgent is not None:
        try:
            return UserAgent().random
        except Exception:
            pass
    return "Mozilla/5.0"


def _fetch_target(
    target: str,
    hpo_ids: str,
    headers: dict,
    max_results: int,
    timeout: int,
) -> pd.DataFrame:
    api_url = f"{BASE_URL}?target={target}&format=tsv&hpo_id={hpo_ids}"
    resp = requests.get(api_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    if not text.strip():
        return pd.DataFrame()
    df = pd.read_csv(StringIO(text), sep="\t")
    if df.empty:
        return df
    return df.head(max_results)


def search_PubCaseFinder(
    query: Union[str, List[str]],
    mode: str = "disease",
    max_results: int = 5,
    disease_mapper: Optional[Any] = None,
    timeout: int = 60,
) -> Optional[pd.DataFrame]:
    """Query PubCaseFinder ranked-list API.

    Parameters
    ----------
    query : HPO IDs as string or list (e.g. ['HP:0001250', 'HP:0000252']).
    mode : 'disease' or 'gene'.
    max_results : per-target cap.
    disease_mapper : optional DiseaseIdMapper to translate OMIM/ORPHA -> MONDO.
    timeout : seconds for HTTP timeout.

    Returns
    -------
    DataFrame with PubCaseFinder's ranked list. For 'disease' mode, columns
    include DISEASE_ID, Score, SOURCE, and MONDO_ID (if disease_mapper
    provided). For 'gene' mode, PubCaseFinder's native columns are returned.
    Returns None on network or parse failure (pipeline continues without it).
    """
    if isinstance(query, str):
        query = [query]

    processed_query: List[str] = []
    for q in query:
        q = str(q).strip()
        if q.startswith("HP:"):
            processed_query.append(q)
        elif q.isdigit():
            processed_query.append(f"HP:{q}")
        else:
            processed_query.append(q)

    hpo_ids = ",".join(processed_query)

    headers = {
        "User-Agent": _resolve_user_agent(),
        "Accept": "text/tab-separated-values,*/*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        mode = str(mode).lower().strip()

        if mode == "disease":
            omim_df = _fetch_target("omim", hpo_ids, headers, max_results, timeout)
            if not omim_df.empty:
                omim_df = omim_df.rename(columns={"OMIM_ID": "DISEASE_ID"})
                omim_df["SOURCE"] = "OMIM"

            orpha_df = _fetch_target("orphanet", hpo_ids, headers, max_results, timeout)
            if not orpha_df.empty:
                orpha_df = orpha_df.rename(columns={"ORPHA_ID": "DISEASE_ID"})
                orpha_df["SOURCE"] = "ORPHA"

            df = pd.concat([omim_df, orpha_df], axis=0, ignore_index=True)
            if df.empty:
                return df

            # Map OMIM / ORPHA -> MONDO
            if disease_mapper is not None and hasattr(disease_mapper, "to_mondo"):
                df["MONDO_ID"] = df["DISEASE_ID"].apply(disease_mapper.to_mondo)
                df = df.dropna(subset=["MONDO_ID"])
                if not df.empty:
                    # Deduplicate Mondo IDs, keeping the best score
                    df = df.sort_values("Score", ascending=False)
                    df = df.drop_duplicates(subset="MONDO_ID", keep="first")

            df = df.sort_values("Score", ascending=False).reset_index(drop=True)
            df["Rank"] = df.index + 1
            return df

        if mode == "gene":
            gene_df = _fetch_target("gene", hpo_ids, headers, max_results, timeout)
            return gene_df

        logger.warning(f"Unknown PubCaseFinder mode: {mode!r}")
        return None

    except requests.RequestException as e:
        #logger.warning(f"PubCaseFinder API request failed: {e}")
        return None
    except Exception as e:
        #logger.warning(f"PubCaseFinder processing error: {e}")
        return None


# Back-compat alias matching the stub name
def query_pubcase_finder_hpo(
    patient_hpos: List[str],
    max_results: int = 5,
    timeout: int = 30,
    disease_mapper: Optional[Any] = None,
) -> Optional[List[dict]]:
    """Thin wrapper returning a list of dicts for disease-mode results."""
    df = search_PubCaseFinder(
        patient_hpos, mode="disease",
        max_results=max_results, disease_mapper=disease_mapper,
        timeout=timeout,
    )
    if df is None or df.empty:
        return None
    return df.to_dict(orient="records")