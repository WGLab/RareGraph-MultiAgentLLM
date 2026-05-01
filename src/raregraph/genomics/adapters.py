"""Adapters from external genomics rankers into RareGraph vcf_summary rows."""
from __future__ import annotations

import csv
import gzip
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


EXOMISER_PATTERNS = (
    "{case_id}.variants.tsv",
    "{case_id}.genes.tsv",
    "variants.tsv",
    "genes.tsv",
    "results.tsv",
)
RANKVAR_PATTERNS = (
    "{case_id}.rank_var.tsv",
    "rank_var.tsv",
    "rankvar.tsv",
)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _read_tsv(path: Path) -> List[Dict[str, str]]:
    with _open_text(path) as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def _first(row: Dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _float(value: Any) -> Optional[float]:
    try:
        if value in (None, ""):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    try:
        if value in (None, ""):
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _variant_key(row: Dict[str, Any]) -> str:
    hgvs = _first(row, "HGVS", "HGVS_GENOMIC", "HGVS_CDNA", "HGVS_PROTEIN")
    if hgvs:
        return hgvs
    contig = _first(row, "CONTIG", "Chr", "CHROM")
    start = _first(row, "START", "Start", "POS")
    ref = _first(row, "REF", "Ref")
    alt = _first(row, "ALT", "Alt")
    if contig and start and ref and alt:
        return f"{contig}:{start}:{ref}>{alt}"
    return ""


def _result_from_exomiser(row: Dict[str, Any], score: Optional[float], rank: Optional[int]) -> str:
    acmg = _first(row, "EXOMISER_ACMG_CLASSIFICATION").lower()
    if "pathogenic" in acmg:
        return "positive"
    if score is not None and score >= 0.80:
        return "positive"
    if rank is not None and rank <= 10:
        return "candidate"
    if score is not None and score >= 0.30:
        return "candidate"
    return "unknown"


def _result_from_rankvar(row: Dict[str, Any], score: Optional[float], rank: Optional[int]) -> str:
    clinvar = _first(row, "ClinVar", "clinvar").lower()
    if "pathogenic" in clinvar:
        return "positive"
    if score is not None and score >= 0.75 and rank is not None and rank <= 5:
        return "positive"
    if rank is not None and rank <= 50:
        return "candidate"
    return "unknown"


def _normalize_exomiser_row(row: Dict[str, Any], source_path: Path) -> Optional[Dict[str, Any]]:
    gene = _first(row, "GENE_SYMBOL", "Gene.refGene", "gene")
    if not gene:
        return None
    rank = _int(_first(row, "#RANK", "RANK", "rank"))
    gene_score = _float(_first(row, "EXOMISER_GENE_COMBINED_SCORE"))
    phenotype_score = _float(_first(row, "EXOMISER_GENE_PHENO_SCORE"))
    variant_score = _float(_first(row, "EXOMISER_VARIANT_SCORE", "EXOMISER_GENE_VARIANT_SCORE"))
    score = gene_score if gene_score is not None else variant_score

    return {
        "source": "exomiser",
        "source_path": str(source_path),
        "gene": gene,
        "variant": _variant_key(row),
        "result": _result_from_exomiser(row, score, rank),
        "rank": rank,
        "score": score,
        "gene_score": gene_score,
        "phenotype_score": phenotype_score,
        "variant_score": variant_score,
        "moi": _first(row, "MOI"),
        "acmg_classification": _first(row, "EXOMISER_ACMG_CLASSIFICATION"),
        "acmg_evidence": _first(row, "EXOMISER_ACMG_EVIDENCE"),
        "consequence": _first(row, "FUNCTIONAL_CLASS", "ExonicFunc.refGene"),
        "genotype": _first(row, "GENOTYPE"),
        "clinvar": _first(row, "CLINVAR", "ClinVar"),
        "raw": row,
    }


def _rankvar_score(row: Dict[str, Any]) -> Optional[float]:
    explicit = _float(_first(row, "RankVar_score", "RankVar_prob", "score"))
    if explicit is not None:
        return explicit
    phen = _float(_first(row, "phen2gene_score", "Symptom_match"))
    path = _float(_first(row, "pathogenecity_score", "pathogenicity_score"))
    vals = [v for v in (phen, path) if v is not None]
    if vals:
        return sum(vals) / len(vals)
    return None


def _normalize_rankvar_row(row: Dict[str, Any], source_path: Path) -> Optional[Dict[str, Any]]:
    gene = _first(row, "Gene.refGene", "gene", "GENE_SYMBOL")
    if not gene:
        return None
    rank = _int(_first(row, "rank", "RANK", "#RANK", "HIPred_rank"))
    score = _rankvar_score(row)
    phenotype_score = _float(_first(row, "phen2gene_score", "Symptom_match"))
    variant_score = _float(_first(row, "pathogenecity_score", "pathogenicity_score"))

    return {
        "source": "rankvar",
        "source_path": str(source_path),
        "gene": gene,
        "variant": _variant_key(row),
        "result": _result_from_rankvar(row, score, rank),
        "rank": rank,
        "score": score,
        "phenotype_score": phenotype_score,
        "variant_score": variant_score,
        "consequence": _first(row, "ExonicFunc.refGene", "Func.refGene"),
        "gnomad_af": _float(_first(row, "gnomad41_exome_AF_grpmax", "gnomAD_AF")),
        "clinvar": _first(row, "ClinVar", "clinvar"),
        "inheritance_match": _first(row, "Inheritance_match"),
        "raw": row,
    }


def _dedupe_best_gene(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    priority = {"positive": 0, "candidate": 1, "unknown": 2, "negative": 3}
    best: Dict[str, Dict[str, Any]] = {}

    def key(row: Dict[str, Any]):
        rank = row.get("rank")
        score = row.get("score")
        return (
            priority.get(str(row.get("result", "")).lower(), 2),
            rank if isinstance(rank, int) else 10**9,
            -(score if isinstance(score, (int, float)) else -1.0),
            0 if row.get("variant") else 1,
        )

    for row in rows:
        gene = str(row.get("gene", "")).upper()
        if not gene:
            continue
        if gene not in best or key(row) < key(best[gene]):
            best[gene] = row
    return sorted(best.values(), key=key)


def load_genomics_results(path: str | Path, analyzer: str = "auto") -> List[Dict[str, Any]]:
    """Load a completed Exomiser/RankVar result file or directory."""
    root = Path(path)
    if not root.exists():
        return []

    files: List[Path]
    if root.is_file():
        files = [root]
    else:
        files = sorted(root.glob("*.tsv")) + sorted(root.glob("*.tsv.gz"))

    normalized: List[Dict[str, Any]] = []
    for file_path in files:
        rows = _read_tsv(file_path)
        if not rows:
            continue
        headers = set(rows[0].keys())
        detected = analyzer.lower()
        if detected == "auto":
            if "EXOMISER_GENE_COMBINED_SCORE" in headers or "GENE_SYMBOL" in headers:
                detected = "exomiser"
            elif "Gene.refGene" in headers or "phen2gene_score" in headers:
                detected = "rankvar"
        for row in rows:
            if detected == "exomiser":
                item = _normalize_exomiser_row(row, file_path)
            elif detected == "rankvar":
                item = _normalize_rankvar_row(row, file_path)
            else:
                item = None
            if item:
                normalized.append(item)
    return _dedupe_best_gene(normalized)


def discover_genomics_result(
    case_id: str,
    vcf_path: str | Path | None,
    analyzer: str = "auto",
    results_dir: str | Path | None = None,
) -> Optional[Path]:
    """Find a completed genomics result near the case inputs."""
    roots: List[Path] = []
    if results_dir:
        roots.append(Path(results_dir))
    if vcf_path:
        vcf = Path(vcf_path)
        input_root = vcf.parent.parent if vcf.parent.name.lower() == "vcf" else vcf.parent
        roots.extend([
            input_root / "genomics" / case_id,
            input_root / "genomics",
            input_root / "exomiser" / case_id,
            input_root / "exomiser",
            input_root / "rankvar" / case_id,
            input_root / "rankvar",
        ])

    patterns: List[str] = []
    if analyzer in ("auto", "exomiser"):
        patterns.extend(EXOMISER_PATTERNS)
    if analyzer in ("auto", "rankvar"):
        patterns.extend(RANKVAR_PATTERNS)

    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            return root
        for pattern in patterns:
            candidate = root / pattern.format(case_id=case_id)
            if candidate.exists():
                return candidate
    return None
