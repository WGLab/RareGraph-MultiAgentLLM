from __future__ import annotations

import json
import logging, torch
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from raregraph.core.config import AttrDict, retrieval_initial_top_k
from raregraph.core.logging import setup_logger
from raregraph.core.state import (
    PatientCaseState, NormalizedPhenotype, TemporalView, IncongruityInfo,
)
from raregraph.core.utils import ensure_dir, write_json

from raregraph.llm.vllm_client import VllmClient
from raregraph.llm.vllm_vision_client import VllmVisionClient
from raregraph.llm.vision_api_client import ApiVisionClient

from raregraph.agents.text_agents import (
    run_phenotype_extractor_batch,
    run_demographics_extractor_batch,
    run_family_history_extractor_batch,
    run_testing_extractor_batch,
    run_gene_mentions_extractor_batch,
)
from raregraph.agents.vision_agents import (
    run_vision_extractor_batch,
    filter_vision_against_text,
)

from raregraph.normalize.biolord_embedder import BioLordEmbedder
from raregraph.normalize.hpo_ontology import HpoOntology
from raregraph.normalize.normalizers import HpoNormalizer
from raregraph.normalize.mondo_normalizer import MondoNormalizer
from raregraph.normalize.temporal_parser import build_temporal_view
from raregraph.normalize.inheritance_inference import infer_inheritance_prior
from raregraph.normalize.incongruity_detector import detect_incongruity
from raregraph.normalize.disease_id_mapper import DiseaseIdMapper

from raregraph.kg.kg_loader import load_kg, load_hierarchy
from raregraph.kg.kg_precompute import precompute_kg_index, KGIndex
from raregraph.genomics.adapters import discover_genomics_result, load_genomics_results
from raregraph.pipeline.vision_prefetch import release_vision_client

from raregraph.retrieval.hpo_retriever import retrieve_by_hpo
from raregraph.retrieval.gene_retriever import retrieve_by_gene
from raregraph.retrieval.cooccurrence_retriever import retrieve_by_cooccurrence
from raregraph.retrieval.pubcase_finder import search_PubCaseFinder

from raregraph.scoring.composite_ranker import score_candidates

from raregraph.frontier.client import FrontierClient
from raregraph.frontier.consultation import run_frontier_consultation

from raregraph.reasoning.audit import run_audit_batch, apply_audit_multipliers
from raregraph.reasoning.pairwise import run_pairwise_batch
from raregraph.reasoning.rank_centrality import aggregate_rank
from raregraph.reasoning.reconciliation import reconcile
from raregraph.reasoning.scorecard import (
    build_scorecard, format_scorecard_text, build_rank_trajectory,
)

logger = setup_logger("raregraph")


class RareGraphHost:
    def __init__(self, cfg: AttrDict):
        self.cfg = cfg
        self.kg: Dict[str, Dict[str, Any]] = {}
        self.kg_index: Optional[KGIndex] = None
        self.hpo: Optional[HpoOntology] = None
        self.hpo_normalizer: Optional[HpoNormalizer] = None
        self.mondo_normalizer: Optional[MondoNormalizer] = None
        self.embedder: Optional[BioLordEmbedder] = None
        self.text_llm: Optional[VllmClient] = None
        self.vision_llm: Optional[Any] = None
        self.frontier: Optional[FrontierClient] = None
        self.disease_mapper: Optional[DiseaseIdMapper] = None
        self.hierarchy: Dict[str, Any] = {}

        self.prompt_dir = str(Path("configs/prompts"))

    # ---------------------------------------------------------------
    # One-time setup
    # ---------------------------------------------------------------
    def load(self) -> None:
        """Load KG, ontology, build indexes, load LLMs."""
        logger.info("=== RareGraphHost.load() ===")

        # Embedder (lazy-loaded model)
        self.embedder = BioLordEmbedder(
            model_name=self.cfg.normalization.embed_model,
            cache_dir=self.cfg.paths.cache_dir,
        )

        # HPO ontology
        self.hpo = HpoOntology(self.cfg.paths.hpo_obo)

        # KG
        self.kg = load_kg(self.cfg.paths.kg_path)
        self.hpo.compute_ic_from_kg(self.kg)

        # Precompute indexes
        self.kg_index = precompute_kg_index(self.kg, self.hpo)

        # Hierarchy
        if self.cfg.paths.hierarchy:
            self.hierarchy = load_hierarchy(self.cfg.paths.hierarchy)
        self._apply_hierarchy_groups()

        # HPO normalizer
        self.hpo_normalizer = HpoNormalizer(
            self.hpo, self.embedder,
            similarity_threshold=self.cfg.normalization.similarity_threshold,
        )
        self.hpo_normalizer.build_index()

        # Mondo normalizer (optional)
        if self.cfg.paths.full_mondo:
            self.mondo_normalizer = MondoNormalizer(
                self.cfg.paths.full_mondo, self.embedder,
                similarity_threshold=self.cfg.normalization.similarity_threshold,
            )
            self.mondo_normalizer.load()

        # Disease ID mapper
        self.disease_mapper = DiseaseIdMapper(
            omim2mondo_path=self.cfg.paths.omim2path,
            orphanet2mondo_path=self.cfg.paths.orphanet2path,
        )

        # Text LLM (required for Stage 1, 5, 6, 8)
        tcfg = self.cfg.models.text_llm
        self.text_llm = VllmClient(
            model_path=tcfg.model_name,
            temperature=tcfg.temperature,
            max_tokens=tcfg.max_tokens,
            tensor_parallel_size=torch.cuda.device_count(),#tcfg.tensor_parallel_size,
            gpu_memory_utilization= tcfg.gpu_memory_utilization,
            max_batch_size=tcfg.max_batch_size,
            enable_thinking=getattr(tcfg, "thinking", False),
        )

        # Vision LLM (optional)
        # Defer loading until first use to save memory when no image is provided.

        # Frontier client
        if self.cfg.frontier.enabled:
            self.frontier = FrontierClient(
                provider=self.cfg.frontier.provider,
                model_name=self.cfg.frontier.model_name,
                api_key_env=self.cfg.frontier.api_key_env,
                api_base_url=self.cfg.frontier.api_base_url,
                temperature=self.cfg.frontier.temperature,
                max_tokens=self.cfg.frontier.max_tokens,
                timeout_seconds=self.cfg.frontier.timeout_seconds,
                local_llm=self.text_llm if self.cfg.frontier.provider == "local" else None,
            )
            logger.info(f"Frontier client ready: provider={self.cfg.frontier.provider}")

        logger.info("=== RareGraphHost.load() complete ===")

    def _apply_hierarchy_groups(self) -> None:
        """Apply hierarchy.json group assignments to the KG index.

        The hierarchy file is keyed by disease MONDO ID and stores the selected
        meaningful group/ancestor. Diseases missing from the file become
        singleton groups.
        """
        if not self.kg_index:
            return

        hierarchy = self.hierarchy or {}
        assigned = 0
        for did, disease_name in list(self.kg_index.disease_name.items()):
            record = hierarchy.get(did, {}) if isinstance(hierarchy, dict) else {}
            if isinstance(record, dict):
                group_id = str(record.get("group_id") or did).strip() or did
                group_name = str(record.get("group_name") or record.get("name") or "").strip()
            else:
                group_id = did
                group_name = ""

            if not group_name:
                group_name = self.kg_index.disease_name.get(
                    group_id,
                    disease_name if group_id == did else group_id,
                )

            self.kg_index.disease_group[did] = group_id
            self.kg_index.disease_name.setdefault(group_id, group_name)
            assigned += 1

        logger.info(f"Applied MONDO hierarchy groups for {assigned} diseases")

    def _ensure_group_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure ranking outputs expose group_id and group_name columns."""
        if df is None or not isinstance(df, pd.DataFrame) or df.empty or "disease_id" not in df.columns:
            return df
        if not self.kg_index:
            return df

        out = df.copy()
        if "group_id" not in out.columns:
            out["group_id"] = ""
        if "group_name" not in out.columns:
            out["group_name"] = ""

        legacy_group = out["mondo_group_id"] if "mondo_group_id" in out.columns else pd.Series([""] * len(out), index=out.index)

        def _clean(value: Any) -> str:
            text = str(value or "").strip()
            return "" if text.lower() in {"nan", "none"} else text

        def _group_id(row: pd.Series) -> str:
            did = _clean(row.get("disease_id", ""))
            existing = _clean(row.get("group_id", ""))
            legacy = _clean(legacy_group.loc[row.name])
            return existing or legacy or self.kg_index.disease_group.get(did, "") or did

        out["group_id"] = out.apply(_group_id, axis=1)

        def _group_name(row: pd.Series) -> str:
            existing = _clean(row.get("group_name", ""))
            if existing:
                return existing
            gid = _clean(row.get("group_id", ""))
            did = _clean(row.get("disease_id", ""))
            if gid == did:
                return _clean(row.get("disease_name", "")) or self.kg_index.disease_name.get(did, did)
            return self.kg_index.disease_name.get(gid, gid)

        out["group_name"] = out.apply(_group_name, axis=1)
        return out

    def _ensure_vision_llm(self) -> None:
        if self.vision_llm is None:
            vcfg = self.cfg.models.vision_llm
            vision_runtime = getattr(self.cfg, "vision", {})
            provider = str(getattr(vision_runtime, "provider", "local")).lower()
            if provider in {"local", "vllm"}:
                self.vision_llm = VllmVisionClient(
                    model_path=vcfg.model_name,
                    temperature=vcfg.temperature,
                    max_tokens=vcfg.max_tokens,
                    tensor_parallel_size=torch.cuda.device_count(),#vcfg.tensor_parallel_size,
                    gpu_memory_utilization=vcfg.gpu_memory_utilization,
                    max_batch_size=vcfg.max_batch_size,
                )
            elif provider in {"openrouter", "openai"}:
                self.vision_llm = ApiVisionClient(
                    provider=provider,
                    model_name=getattr(vision_runtime, "model_name", vcfg.model_name),
                    api_key_env=getattr(vision_runtime, "api_key_env", "OPENROUTER_API_KEY"),
                    api_base_url=getattr(vision_runtime, "api_base_url", "https://openrouter.ai/api/v1"),
                    temperature=vcfg.temperature,
                    max_tokens=vcfg.max_tokens,
                    timeout_seconds=getattr(vision_runtime, "timeout_seconds", 120),
                    max_batch_size=vcfg.max_batch_size,
                )
            else:
                raise ValueError(f"Unknown vision provider: {provider}")

    def _release_vision_llm(self) -> None:
        if self.vision_llm is not None:
            release_vision_client(self.vision_llm)
            self.vision_llm = None

    # ---------------------------------------------------------------
    # Per-patient pipeline
    # ---------------------------------------------------------------
    def run_patient(
        self,
        case_id: str,
        note_path: Optional[str] = None,
        image_path: Optional[str] = None,
        vcf_path: Optional[str] = None,
        free_hpo_path: Optional[str] = None,
        vision_json_path: Optional[str] = None,
        stage1_json_path: Optional[str] = None,
        genomics_result_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full 9-stage pipeline for a single patient."""
        out_dir = ensure_dir(output_dir or f"outputs/{case_id}")
        logger.info(f"=== Running case {case_id} ===")

        state = PatientCaseState(
            case_id=case_id,
            note_path=note_path,
            image_path=image_path,
            vcf_path=vcf_path,
            free_hpo_path=free_hpo_path,
        )

        # =================== STAGE 1 ===================
        logger.info("--- Stage 1: Multi-modal extraction ---")
        stage1_out = self._stage1_extraction(
            state,
            note_path,
            image_path,
            free_hpo_path,
            vision_json_path=vision_json_path,
            stage1_json_path=stage1_json_path,
            genomics_result_path=genomics_result_path,
        )
        write_json(stage1_out, out_dir / "stage1_extraction.json")

        # =================== STAGE 2 ===================
        logger.info("--- Stage 2: Normalization + post-processing ---")
        self._stage2_normalization(state)
        self._save_patient_evidence(state, out_dir)

        # =================== STAGE 3 ===================
        logger.info("--- Stage 3: Candidate retrieval + composite scoring ---")
        ranked_df, cooccurrence_candidates = self._stage3_retrieval_scoring(state)
        ranked_df = self._ensure_group_columns(ranked_df)
        state.ranked = ranked_df
        ranked_df.to_csv(out_dir / "stage3_composite_ranking.tsv", sep="\t", index=False)

        # =================== STAGE 4 ===================
        logger.info("--- Stage 4: Frontier consultation (conditional) ---")
        frontier_output, frontier_flags = self._stage4_frontier(state, ranked_df, out_dir)
        state.frontier_output = frontier_output

        # =================== STAGE 5 ===================
        logger.info("--- Stage 5: Audit ---")
        audit_results, ranking_after_audit = self._stage5_audit(
            state, ranked_df, frontier_flags
        )
        ranking_after_audit = self._ensure_group_columns(ranking_after_audit)
        state.audit_results = audit_results
        state.ranking_after_audit = ranking_after_audit
        write_json(audit_results, out_dir / "stage5_audit_results.json")
        ranking_after_audit.to_csv(out_dir / "stage5_ranking_after_audit.tsv", sep="\t", index=False)

        # =================== STAGE 6 ===================
        logger.info("--- Stage 6: Pairwise adjudication ---")
        pairwise_subtype = self._stage6_pairwise(
            state, ranking_after_audit, audit_results, frontier_flags, track="subtype"
        )
        state.pairwise_results_subtype = pairwise_subtype
        write_json(pairwise_subtype, out_dir / "stage6_pairwise_results_subtype.json")

        # Group-level pairwise (on group IDs)
        group_ranked_df = self._aggregate_to_group(ranking_after_audit, frontier_flags)
        pairwise_group = self._stage6_pairwise(
            state, group_ranked_df, audit_results, frontier_flags, track="group",
        ) if len(group_ranked_df) > 1 else []
        state.pairwise_results_group = pairwise_group
        write_json(pairwise_group, out_dir / "stage6_pairwise_results_group.json")

        # =================== STAGE 7 ===================
        logger.info("--- Stage 7: Rank aggregation ---")
        reranked_subtype = aggregate_rank(
            ranking_after_audit, pairwise_subtype, self.cfg, track_name="subtype"
        )
        reranked_subtype = self._ensure_group_columns(reranked_subtype)
        state.reranked_subtype = reranked_subtype
        reranked_subtype.to_csv(out_dir / "stage7_reranked_subtype.tsv", sep="\t", index=False)

        reranked_group = aggregate_rank(
            group_ranked_df, pairwise_group, self.cfg, track_name="group"
        ) if len(group_ranked_df) > 1 else group_ranked_df
        reranked_group = self._ensure_group_columns(reranked_group)
        state.reranked_group = reranked_group
        reranked_group.to_csv(out_dir / "stage7_reranked_group.tsv", sep="\t", index=False)

        # =================== STAGE 8 ===================
        logger.info("--- Stage 8: Reconciliation ---")
        reconciled = reconcile(
            reranked_subtype, reranked_group, state, self.kg_index, self.cfg,
            llm=self.text_llm, prompt_dir=self.prompt_dir,
        )
        state.reconciled = {
            "top_subtype": reconciled.get("top_subtype"),
            "top_group": reconciled.get("top_group"),
            "disagreement": reconciled.get("disagreement", False),
            "tiebreaker": reconciled.get("tiebreaker"),
            "method": reconciled.get("method", ""),
        }
        write_json(state.reconciled, out_dir / "stage8_reconciled.json")

        # =================== STAGE 9 ===================
        logger.info("--- Stage 9: Clinical scorecard ---")
        final_df_for_scorecard = (
            reconciled["reconciled_df"] if isinstance(reconciled.get("reconciled_df"), pd.DataFrame)
            else reranked_subtype
        )
        final_df_for_scorecard = self._ensure_group_columns(final_df_for_scorecard)
        if isinstance(reconciled.get("reconciled_df"), pd.DataFrame):
            reconciled["reconciled_df"] = final_df_for_scorecard
        scorecard = build_scorecard(
            state, final_df_for_scorecard, audit_results,
            reconciled, frontier_output,
            top_k=self.cfg.output.scorecard_top_k,
        )
        state.scorecard = scorecard

        if self.cfg.output.save_scorecard_json:
            write_json(scorecard, out_dir / "stage9_scorecard.json")
        if self.cfg.output.save_scorecard_txt:
            (out_dir / "stage9_scorecard.txt").write_text(
                format_scorecard_text(scorecard), encoding="utf-8"
            )

        # Rank trajectory
        if self.cfg.output.save_rank_trajectory:
            traj = build_rank_trajectory(
                ranked_df, audit_results, reranked_subtype, reconciled, frontier_output
            )
            traj.to_csv(out_dir / "rank_trajectory.tsv", sep="\t", index=False)

        logger.info(f"=== Case {case_id} done ===")

        return {
            "case_id": case_id,
            "top": scorecard.get("top_candidates", [])[:5],
            "output_dir": str(out_dir),
            "reconciled": state.reconciled,
        }

    # ===============================================================
    # Stage implementations
    # ===============================================================
    def _stage1_extraction(
        self,
        state: PatientCaseState,
        note_path: str,
        image_path: Optional[str],
        free_hpo_path: Optional[str],
        vision_json_path: Optional[str] = None,
        stage1_json_path: Optional[str] = None,
        genomics_result_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # Text
        loaded_stage1_cache = False
        cache_path = Path(stage1_json_path) if stage1_json_path else None
        if cache_path and cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached, dict):
                loaded_stage1_cache = True
                state.note_text = cached.get("note_text") or state.note_text
                state.phenotype_mentions_text = cached.get("text_phenotypes", []) or []
                state.demographics = cached.get("demographics", {}) or {}
                state.family_history = cached.get("family_history", []) or []
                state.testing = cached.get("testing", []) or []
                state.gene_mentions = cached.get("gene_mentions", []) or []
                out.update(cached)
                out["stage1_cache_path"] = str(cache_path)

        if note_path and not loaded_stage1_cache:
            # Read note
            note_text = Path(note_path).read_text(encoding="utf-8")
            phens = run_phenotype_extractor_batch(self.text_llm, [note_text], self.prompt_dir)[0]
            demo = run_demographics_extractor_batch(self.text_llm, [note_text], self.prompt_dir)[0]
            fam = run_family_history_extractor_batch(self.text_llm, [note_text], self.prompt_dir)[0]
            tests = run_testing_extractor_batch(self.text_llm, [note_text], self.prompt_dir)[0]
            genes = run_gene_mentions_extractor_batch(self.text_llm, [note_text], self.prompt_dir)[0]
            
            state.note_text = note_text
            state.phenotype_mentions_text = phens if isinstance(phens, list) else []
            state.demographics = demo if isinstance(demo, dict) else {}
            state.family_history = fam if isinstance(fam, list) else []
            state.testing = tests if isinstance(tests, list) else []
            state.gene_mentions = genes if isinstance(genes, list) else []

            out["text_phenotypes"] = state.phenotype_mentions_text
            out["demographics"] = state.demographics
            out["family_history"] = state.family_history
            out["testing"] = state.testing
            out["gene_mentions"] = state.gene_mentions

        if note_path and not state.note_text:
            state.note_text = Path(note_path).read_text(encoding="utf-8")
        # Vision
        if image_path and Path(image_path).exists():
            vphens: List[Dict[str, Any]] = []
            cache_path = Path(vision_json_path) if vision_json_path else None
            if cache_path and cache_path.exists():
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if isinstance(cached, dict):
                    vphens = cached.get("vision_phenotypes_raw", [])
                elif isinstance(cached, list):
                    vphens = cached
                out["vision_prefetch_path"] = str(cache_path)
            else:
                self._ensure_vision_llm()
                vphens = run_vision_extractor_batch(
                    self.vision_llm, [image_path], self.prompt_dir
                )[0]
                vision_runtime = getattr(self.cfg, "vision", {})
                if getattr(vision_runtime, "unload_after_extraction", True):
                    self._release_vision_llm()
            vfiltered = filter_vision_against_text(vphens, state.phenotype_mentions_text)
            state.phenotype_mentions_vision = vfiltered
            out["vision_phenotypes"] = vfiltered

        # Free HPO list (optional): lines of "HP:xxx" or free text
        if free_hpo_path and Path(free_hpo_path).exists():
            lines = [
                term
                for l in Path(free_hpo_path).read_text(encoding="utf-8").splitlines()
                if l.strip()
                for term in l.strip().split(";")
                if term  # remove empty from trailing ";"
            ]
            free_mentions: List[Dict[str, Any]] = []
            for l in lines:
                free_mentions.append({
                    "mention": l,
                    "attribution": "patient",
                    "onset": None,
                    "evidence": "free HPO list",
                    "source": "free_hpo",
                })
            state.phenotype_mentions_free_hpo = free_mentions
            out["free_hpo_mentions"] = free_mentions

        # VCF/genomics: prefer completed Exomiser/RankVar/RankVar-like results.
        if vcf_path := state.vcf_path:
            p = Path(vcf_path)
            if p.exists():
                genomics_cfg = getattr(self.cfg, "genomics", {})
                analyzer = getattr(genomics_cfg, "vcf_analyzer", "auto")
                results_dir = getattr(genomics_cfg, "results_dir", "")
                result_path = (
                    Path(genomics_result_path)
                    if genomics_result_path
                    else discover_genomics_result(
                        state.case_id,
                        p,
                        analyzer=analyzer,
                        results_dir=results_dir or None,
                    )
                )
                if result_path and Path(result_path).exists():
                    state.vcf_summary = load_genomics_results(result_path, analyzer=analyzer)
                    out["vcf_summary"] = state.vcf_summary
                    out["vcf_result_path"] = str(result_path)
                    out["vcf_note"] = (
                        f"Loaded {len(state.vcf_summary)} normalized genomics rows "
                        f"from {result_path}."
                    )
                else:
                    out["vcf_summary"] = []
                    out["vcf_note"] = (
                        f"VCF present at {p}; no completed Exomiser/RankVar result found. "
                        "Genomics evidence is pending and was not scored."
                    )

        return out

    def _stage1_text_extraction_batch(
        self,
        cases: List[Dict[str, Any]],
    ) -> Dict[str, Dict[str, Any]]:
        """Run Stage 1 text extractors for many cases in true vLLM batches."""
        note_cases = [
            case for case in cases
            if case.get("note_path") and Path(case["note_path"]).exists()
        ]
        if not note_cases:
            return {}

        notes = [Path(case["note_path"]).read_text(encoding="utf-8") for case in note_cases]
        phens = run_phenotype_extractor_batch(self.text_llm, notes, self.prompt_dir)
        demos = run_demographics_extractor_batch(self.text_llm, notes, self.prompt_dir)
        fams = run_family_history_extractor_batch(self.text_llm, notes, self.prompt_dir)
        tests = run_testing_extractor_batch(self.text_llm, notes, self.prompt_dir)
        genes = run_gene_mentions_extractor_batch(self.text_llm, notes, self.prompt_dir)

        out: Dict[str, Dict[str, Any]] = {}
        for case, note_text, phen, demo, fam, test, gene in zip(
            note_cases, notes, phens, demos, fams, tests, genes
        ):
            out[case["case_id"]] = {
                "note_text": note_text,
                "text_phenotypes": phen if isinstance(phen, list) else [],
                "demographics": demo if isinstance(demo, dict) else {},
                "family_history": fam if isinstance(fam, list) else [],
                "testing": test if isinstance(test, list) else [],
                "gene_mentions": gene if isinstance(gene, list) else [],
            }
        return out

    def precompute_stage1_extractions(
        self,
        cases: List[Dict[str, Any]],
        overwrite: bool = False,
    ) -> Dict[str, str]:
        """Batch Stage 1 text extraction and write per-case cache JSON files."""
        to_extract = []
        for case in cases:
            cache_path = Path(case["stage1_json_path"])
            if cache_path.exists() and not overwrite:
                continue
            if case.get("note_path"):
                to_extract.append(case)

        extracted = self._stage1_text_extraction_batch(to_extract)
        written: Dict[str, str] = {}
        for case in to_extract:
            case_id = case["case_id"]
            if case_id not in extracted:
                continue
            cache_path = Path(case["stage1_json_path"])
            write_json(extracted[case_id], cache_path)
            written[case_id] = str(cache_path)
        return written

    def _stage2_normalization(self, state: PatientCaseState) -> None:
        # Combine all phenotype sources
        all_mentions: List[Dict[str, Any]] = []
        for p in state.phenotype_mentions_text:
            q = dict(p)
            q.setdefault("source", "text")
            all_mentions.append(q)
        for p in state.phenotype_mentions_vision:
            q = dict(p)
            q.setdefault("source", "vision")
            all_mentions.append(q)
        for p in state.phenotype_mentions_free_hpo:
            q = dict(p)
            q.setdefault("source", "free_hpo")
            all_mentions.append(q)

        # Normalize
        normalized = self.hpo_normalizer.normalize(all_mentions, include_negated=True)
        # Convert to pydantic models
        state.normalized_hpo = []
        for n in normalized:
            state.normalized_hpo.append(NormalizedPhenotype(
                hpo_id=n["hpo_id"],
                hpo_name=n["hpo_name"],
                attribution=n.get("attribution", "patient"),
                score=float(n.get("score", 0.0)),
                source=n.get("source", "text"),
                mention=n.get("mention", ""),
                onset=n.get("onset"),
                evidence=n.get("evidence"),
                ic=float(n.get("ic", 0.0)),
            ))

        # Temporal view (over present phenotypes)
        present_list = [
            {"mention": n.mention, "hpo_id": n.hpo_id, "hpo_name": n.hpo_name, "onset": n.onset}
            for n in state.normalized_hpo if n.present
        ]
        tv = build_temporal_view(present_list)
        # age from demographics
        demo = state.demographics if isinstance(state.demographics, dict) else {}
        age_info = demo.get("age", {})
        if isinstance(age_info, dict):
            tv["age_at_presentation"] = age_info.get("value") or age_info.get("age_group")
        state.temporal_view = TemporalView(**tv)

        # Inheritance prior
        state.inheritance_prior = infer_inheritance_prior(state.family_history)

        # Incongruity
        state.incongruity = IncongruityInfo(**detect_incongruity(present_list, self.hpo))

    def _save_patient_evidence(self, state: PatientCaseState, out_dir: Path) -> None:
        pe = {
            "case_id": state.case_id,
            "demographics": state.demographics,
            "phenotypes": [n.model_dump() for n in state.normalized_hpo],
            "negated": [n.model_dump() for n in state.normalized_hpo if not n.present],
            "temporal_view": state.temporal_view.model_dump(),
            "inheritance_prior": state.inheritance_prior,
            "incongruity": state.incongruity.model_dump(),
            "gene_evidence": {
                "gene_mentions": state.gene_mentions,
                "vcf_summary": state.vcf_summary,
            },
            "family_history": state.family_history,
        }
        write_json(pe, out_dir / "stage2_patient_evidence.json")
        write_json(state.incongruity.model_dump(), out_dir / "stage2_incongruity.json")

    def _stage3_retrieval_scoring(
        self,
        state: PatientCaseState,
    ):
        present_hpos = [
            {"hpo_id": n.hpo_id, "mention": n.mention, "reliability": "high"}
            for n in state.normalized_hpo if n.present
        ]

        # Retrieval (union of 3 sources)
        hpo_candidates = retrieve_by_hpo(
            present_hpos, self.kg_index, self.hpo,
            expansion_mode=self.cfg.expansion.mode,
            max_depth=self.cfg.expansion.max_expansion_depth,
        )
        gene_candidates = retrieve_by_gene(
            state.gene_mentions, state.vcf_summary, self.kg_index
        )
        cooc_candidates = retrieve_by_cooccurrence(
            present_hpos, self.kg_index, self.hpo,
        )

        # -- PubCaseFinder (optional, network) --
        pubcase_scores: Dict[str, float] = {}
        pubcase_gene_df = None
        if (
            getattr(self.cfg, "external", None) is not None
            and getattr(self.cfg.external, "pubcase_finder", None) is not None
            and self.cfg.external.pubcase_finder.enabled
            and present_hpos
        ):
            hpo_id_list = [p["hpo_id"] for p in present_hpos if p.get("hpo_id")]
            if hpo_id_list:
                try:
                    disease_df = search_PubCaseFinder(
                        query=hpo_id_list,
                        mode="disease",
                        max_results=self.cfg.external.pubcase_finder.max_results,
                        disease_mapper=self.disease_mapper,
                    )
                    state.public_disease_cases = disease_df
                    if disease_df is not None and not disease_df.empty and "MONDO_ID" in disease_df.columns:
                        # Normalize scores to [0, 1] for use as the cases_score component
                        raw = disease_df["Score"].astype(float).values
                        if raw.max() > 0:
                            norm = raw / raw.max()
                        else:
                            norm = raw
                        for mondo_id, s in zip(disease_df["MONDO_ID"].values, norm):
                            if mondo_id:
                                pubcase_scores[str(mondo_id)] = float(s)

                    pubcase_gene_df = search_PubCaseFinder(
                        query=hpo_id_list,
                        mode="gene",
                        max_results=self.cfg.external.pubcase_finder.max_results,
                        disease_mapper=None,
                    )
                    state.public_gene_cases = pubcase_gene_df
                except Exception as e:
                    logger.warning(f"PubCaseFinder call failed: {e}")

        all_ids = (
            set(hpo_candidates.keys())
            | set(gene_candidates.keys())
            | set(cooc_candidates.keys())
            | set(pubcase_scores.keys())
        )

        if not all_ids:
            all_ids = set(list(self.kg_index.disease_name.keys())[: retrieval_initial_top_k(self.cfg)])

        # Cap by top_k
        all_ids = list(all_ids)
        
        # Score
        df = score_candidates(
            candidate_ids=all_ids,
            patient_state=state,
            kg=self.kg,
            kg_index=self.kg_index,
            hpo=self.hpo,
            cfg=self.cfg,
            hpo_normalizer=self.hpo_normalizer,
            cooccurrence_candidates=cooc_candidates,
            cases_scores=pubcase_scores,
        )

        # Keep full scored table here; downstream stages apply their own top-N cutoffs.
        df["rank"] = df.index + 1
        return df, cooc_candidates

    def _stage4_frontier(
        self,
        state: PatientCaseState,
        ranked_df: pd.DataFrame,
        out_dir: Path,
    ):
        if not self.cfg.frontier.enabled or self.frontier is None:
            logger.info("Frontier stage disabled in config")
            return {}, {}

        patient_evidence = {
            "demographics": state.demographics,
            "phenotypes": [n.model_dump() for n in state.normalized_hpo],
            "temporal_view": state.temporal_view.model_dump(),
            "gene_evidence": {
                "gene_mentions": state.gene_mentions,
                "vcf_summary": state.vcf_summary,
            },
            "family_history": state.family_history,
        }

        incongruity = state.incongruity.model_dump()

        out = run_frontier_consultation(
            self.frontier, patient_evidence, ranked_df, incongruity,
            self.cfg, self.prompt_dir,
        )

        # Build per-disease flags for downstream stages
        flags: Dict[str, Dict[str, Any]] = {}
        for e in out.get("underranked", []):
            if e.get("matched") and e.get("disease_id"):
                flags[e["disease_id"]] = {
                    "flag_type": "underranked",
                    "category": "strong",
                    "lens": e.get("lens", ""),
                    "reasoning": e.get("reasoning", ""),
                }
        for e in out.get("overranked", []):
            if e.get("matched") and e.get("disease_id"):
                flags[e["disease_id"]] = {
                    "flag_type": "overranked",
                    "category": "weak",
                    "lens": e.get("lens", ""),
                    "reasoning": e.get("reasoning", ""),
                }

        write_json(out, out_dir / "stage4_frontier_output.json")
        return out, flags

    def _stage5_audit(
        self,
        state: PatientCaseState,
        ranked_df: pd.DataFrame,
        frontier_flags: Dict[str, Dict[str, Any]],
    ):
        audit_results = run_audit_batch(
            self.text_llm, state, ranked_df, self.kg_index, self.cfg,
            self.prompt_dir, hpo=self.hpo, frontier_flags=frontier_flags,
        )
        ranking_after_audit = apply_audit_multipliers(ranked_df, audit_results)
        return audit_results, ranking_after_audit

    def _stage6_pairwise(
        self,
        state: PatientCaseState,
        ranked_df: pd.DataFrame,
        audit_results: List[Dict[str, Any]],
        frontier_flags: Dict[str, Dict[str, Any]],
        track: str = "subtype",
    ):
        return run_pairwise_batch(
            self.text_llm, state, ranked_df, audit_results,
            self.kg_index, self.kg, self.hpo, self.cfg, self.prompt_dir,
            frontier_flags=frontier_flags, track=track,
        )

    def _aggregate_to_group(
        self,
        subtype_df: pd.DataFrame,
        frontier_flags: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> pd.DataFrame:
        """Collapse subtype-level ranking to group-level ranking."""
        if not self.kg_index:
            return pd.DataFrame()

        frontier_flags = frontier_flags or {}
        rows: List[Dict[str, Any]] = []
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for _, r in subtype_df.iterrows():
            gid = r.get("group_id") or r.get("mondo_group_id") or r["disease_id"]
            grouped.setdefault(gid, []).append(r.to_dict())

        for gid, members in grouped.items():
            # best member's adjusted score is the group's initial score
            scores = [float(m.get("adjusted_score", m.get("total_score", 0.0))) for m in members]
            plausibility_priority = {
                "strong": 4,
                "moderate": 3,
                "weak": 2,
                "not_audited": 1,
                "implausible": 0,
            }
            best_plausibility = max(
                (str(m.get("audit_plausibility", "not_audited")) for m in members),
                key=lambda value: plausibility_priority.get(value, 1),
                default="not_audited",
            )
            member_sources = [str(m.get("audit_source", "")) for m in members]
            if "frontier_underranked" in member_sources:
                audit_source = "frontier_underranked"
            elif "frontier_overranked" in member_sources:
                audit_source = "frontier_overranked"
            else:
                audit_source = next((source for source in member_sources if source), "")
            member_ids = [str(m.get("disease_id", "")) for m in members]
            member_flag_types = {
                (frontier_flags.get(member_id) or {}).get("flag_type", "")
                for member_id in member_ids
            }
            if "underranked" in member_flag_types:
                audit_source = "frontier_underranked"
            elif "overranked" in member_flag_types and best_plausibility in {
                "weak",
                "implausible",
                "not_audited",
            }:
                audit_source = "frontier_overranked"
            group_name = self.kg_index.disease_name.get(gid, gid)
            rows.append({
                "disease_id": gid,
                "disease_name": group_name,
                "group_id": gid,
                "group_name": group_name,
                "total_score": max(scores) if scores else 0.0,
                "adjusted_score": max(scores) if scores else 0.0,
                "audit_plausibility": best_plausibility,
                "audit_multiplier": max(float(m.get("audit_multiplier", 1.0)) for m in members),
                "audit_source": audit_source,
                "n_members": len(members),
                "member_ids": member_ids,
            })

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows).sort_values("adjusted_score", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        df["adjusted_rank"] = df["rank"]
        return df
