#!/usr/bin/env python
"""Run the RareGraph pipeline on one or more patient cases.

Expected input layout:
  <input_dir>/<dataset>/text/<case_id>.txt         (required)
  <input_dir>/<dataset>/image/<case_id>.(jpg|png)  (optional)
  <input_dir>/<dataset>/vcf/<case_id>.vcf          (optional)
  <input_dir>/<dataset>/free_hpo/<case_id>.txt          (optional — free HPO list)

Output:
  <output_dir>/<dataset>/<case_id>/stage*.json|tsv
  <output_dir>/<dataset>/<case_id>/stage9_scorecard.{json,txt}
  <output_dir>/<dataset>/<case_id>/rank_trajectory.tsv
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from tqdm import tqdm

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from raregraph.core.config import load_config
from raregraph.core.logging import setup_logger
from raregraph.orchestration.host import RareGraphHost
from raregraph.genomics.adapters import discover_genomics_result
from raregraph.pipeline.vision_prefetch import prefetch_vision_for_cases


def _find_file(*candidates: Path) -> Path | None:
    for c in candidates:
        if c.exists():
            return c
    return None


def _discover_case_ids(input_root: Path) -> set:
    """Scan all modality folders and union the case IDs found."""
    case_ids = set()
    for subdir, globs in [
        ("text", ["*.txt"]),
        ("free_hpo", ["*.txt"]),
        ("vcf", ["*.vcf", "*.vcf.gz"]),
        ("image", ["*.jpg", "*.jpeg", "*.png", "*.tiff"]),
    ]:
        d = input_root / subdir
        if not d.exists():
            continue
        for pattern in globs:
            for p in d.glob(pattern):
                stem = p.stem
                # .vcf.gz has double suffix: strip the .vcf part from stem
                if stem.endswith(".vcf"):
                    stem = stem[:-4]
                case_ids.add(stem)
    return case_ids


def _cfg_get(obj, key: str, default=None):
    return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)


def _maybe_trigger_genomics(
    logger,
    command_template: str,
    case_id: str,
    vcf_path: Path,
    input_root: Path,
    output_root: Path,
    results_dir: str,
) -> None:
    if not command_template:
        return
    command = command_template.format(
        case_id=case_id,
        vcf_path=str(vcf_path),
        input_root=str(input_root),
        output_root=str(output_root),
        results_dir=results_dir,
    )
    subprocess.Popen(command, shell=True)
    logger.info(f"Triggered genomics job for {case_id}; current run will treat it as pending.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RareGraph pipeline")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", default=None, help="Dataset subfolder name (e.g. 'demo')")
    parser.add_argument("--input_dir", default="inputs")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--case_id", default=None, help="Process a single case by ID")
    parser.add_argument("--limit", type=int, default=None, help="Max cases to process")
    parser.add_argument(
        "--no_vision_prefetch",
        action="store_true",
        help="Disable vision-first extraction cache; run vision inline during Stage 1.",
    )
    parser.add_argument(
        "--overwrite_vision_cache",
        action="store_true",
        help="Re-run vision extraction even if stage1_vision_prefetch.json exists.",
    )
    parser.add_argument(
        "--no_stage1_prefetch",
        action="store_true",
        help="Disable batched Stage 1 text extraction cache; run extraction per case.",
    )
    parser.add_argument(
        "--overwrite_stage1_cache",
        action="store_true",
        help="Re-run Stage 1 text extraction even if stage1_text_prefetch.json exists.",
    )
    parser.add_argument(
        "--genomics_results_dir",
        default=None,
        help="Optional directory containing completed Exomiser/RankVar outputs.",
    )
    parser.add_argument(
        "--genomics_command",
        default=None,
        help=(
            "Optional async command template to run when VCF exists but no completed "
            "genomics output is found. Available fields: {case_id}, {vcf_path}, "
            "{input_root}, {output_root}, {results_dir}."
        ),
    )
    parser.add_argument(
        "--text_model",
        default=None,
        help=(
            "Override cfg.models.text_llm.model_name at runtime. "
            "If omitted, the value from the config file is used. "
            "Example: 'meta-llama/Llama-3.1-8B-Instruct'"
        ),
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logger = setup_logger(level=logging.DEBUG if args.verbose else logging.INFO)
    cfg = load_config(args.config)

    # --- Runtime overrides (applied after config load, before model is loaded) ---
    if args.text_model:
        cfg.models.text_llm.model_name = args.text_model
        logger.info(f"CLI override: cfg.models.text_llm.model_name → {args.text_model}")

    dataset = args.dataset or cfg.project.dataset
    input_root = Path(args.input_dir) / dataset
    output_root = Path(args.output_dir) / dataset
    output_root.mkdir(parents=True, exist_ok=True)

    if not input_root.exists():
        logger.error(f"Input root not found: {input_root}")
        return 1

    # Collect cases from ALL modality folders (text, hpo, vcf, image)
    if args.case_id:
        case_ids = [args.case_id]
    else:
        case_ids = sorted(_discover_case_ids(input_root))
        if args.limit:
            case_ids = case_ids[: args.limit]

    if not case_ids:
        logger.error(
            f"No patient files found in {input_root}. "
            f"Expected at least one file under text/, free_hpo/, vcf/, or image/."
        )
        return 1

    logger.info(f"Will process {len(case_ids)} cases")

    case_inputs = []
    for cid in case_ids:
        note_path = _find_file(input_root / "text" / f"{cid}.txt")
        image_path = _find_file(
            input_root / "image" / f"{cid}.jpg",
            input_root / "image" / f"{cid}.jpeg",
            input_root / "image" / f"{cid}.png",
            input_root / "image" / f"{cid}.tiff",
        )
        vcf_path = _find_file(
            input_root / "vcf" / f"{cid}.vcf",
            input_root / "vcf" / f"{cid}.vcf.gz",
        )
        hpo_path = _find_file(input_root / "free_hpo" / f"{cid}.txt")
        case_out = output_root / cid
        case_inputs.append({
            "case_id": cid,
            "note_path": str(note_path) if note_path else None,
            "image_path": str(image_path) if image_path else None,
            "vcf_path": str(vcf_path) if vcf_path else None,
            "free_hpo_path": str(hpo_path) if hpo_path else None,
            "output_dir": str(case_out),
            "stage1_json_path": str(case_out / "stage1_text_prefetch.json"),
            "vision_json_path": str(case_out / "stage1_vision_prefetch.json") if image_path else None,
        })

    # Vision prefetch runs before host.load(), so the local vision model can be
    # unloaded before the text model claims GPU memory.
    vision_runtime = _cfg_get(cfg, "vision", {})
    vision_mode = _cfg_get(vision_runtime, "extraction_mode", "preextract_unload")
    vision_cache_by_case = {}
    if args.no_vision_prefetch:
        vision_mode = "inline"
    if vision_mode == "preextract_unload":
        prefetch_cases = []
        for case in case_inputs:
            if case.get("image_path") and case.get("vision_json_path"):
                prefetch_cases.append((
                    case["case_id"], case["image_path"], case["vision_json_path"]
                ))
                vision_cache_by_case[case["case_id"]] = case["vision_json_path"]
        if prefetch_cases:
            logger.info(
                f"Prefetching vision extraction for {len(prefetch_cases)} image cases "
                "before loading the text model"
            )
            prefetch_vision_for_cases(
                cfg,
                prefetch_cases,
                prompt_dir="configs/prompts",
                overwrite=args.overwrite_vision_cache,
            )

    # Load host
    host = RareGraphHost(cfg)
    host.load()

    stage1_cache_by_case = {
        case["case_id"]: case["stage1_json_path"]
        for case in case_inputs
        if case.get("stage1_json_path")
    }
    if not args.no_stage1_prefetch:
        text_cases = [case for case in case_inputs if case.get("note_path")]
        if text_cases:
            logger.info(
                f"Prefetching Stage 1 text extraction for {len(text_cases)} note cases "
                "using batched vLLM calls"
            )
            host.precompute_stage1_extractions(
                text_cases,
                overwrite=args.overwrite_stage1_cache,
            )

    # Run
    summaries = []
    for case in tqdm(case_inputs, desc="Cases", unit="case"):
        cid = case["case_id"]
        vcf_path = Path(case["vcf_path"]) if case.get("vcf_path") else None
        genomics_results_dir = args.genomics_results_dir or getattr(cfg.genomics, "results_dir", "")
        genomics_result_path = discover_genomics_result(
            cid,
            vcf_path,
            analyzer=getattr(cfg.genomics, "vcf_analyzer", "auto"),
            results_dir=genomics_results_dir,
        ) if vcf_path else None
        if vcf_path and genomics_result_path is None and getattr(cfg.genomics, "trigger_if_missing", False):
            _maybe_trigger_genomics(
                logger,
                args.genomics_command or getattr(cfg.genomics, "trigger_command", ""),
                cid,
                vcf_path,
                input_root,
                output_root,
                genomics_results_dir,
            )

        result = host.run_patient(
            case_id=cid,
            note_path=case.get("note_path"),
            image_path=case.get("image_path"),
            vcf_path=case.get("vcf_path"),
            free_hpo_path=case.get("free_hpo_path"),
            vision_json_path=vision_cache_by_case.get(cid),
            stage1_json_path=None if args.no_stage1_prefetch else stage1_cache_by_case.get(cid),
            genomics_result_path=str(genomics_result_path) if genomics_result_path else None,
            output_dir=case.get("output_dir"),
        )
        summaries.append(result)

    # Aggregate summary
    import pandas as pd
    rows = []
    for s in summaries:
        top = s.get("top", [])
        if top:
            rows.append({
                "case_id": s["case_id"],
                "top1_disease": top[0].get("disease_name"),
                "top1_id": top[0].get("disease_id"),
                "top1_audit": top[0].get("audit_plausibility"),
                "top1_final_rank": top[0].get("reconciled_rank"),
            })
        else:
            rows.append({"case_id": s["case_id"]})
    if rows:
        pd.DataFrame(rows).to_csv(output_root / "summary.tsv", sep="\t", index=False)
        logger.info(f"Summary → {output_root / 'summary.tsv'}")

    logger.info("Pipeline done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())