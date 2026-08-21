# 🧬 RareMind

### From multimodal patient evidence to ranked rare-disease hypotheses—and clinically actionable next steps

RareMind is a locally deployable, multi-agent rare-disease reasoning pipeline grounded in **RareGraph**, the structured knowledge graph developed for this project. It integrates clinical notes, phenotype terms, images, and genomic evidence; retrieves and adjudicates disease candidates; produces an auditable final ranking; and now converts its leading diagnostic hypotheses into normalized **next-test and next-step recommendations**.

**RareMind turns multimodal clinical evidence into an auditable rare-disease differential—and turns that differential into a normalized, traceable plan for what clinicians might evaluate next.**

> **Pipeline name:** RareMind  
> **Knowledge graph:** RareGraph  
> **Default final output:** ranked diagnoses **plus** ten normalized clinical actions

---

## ✨ What RareMind does

- 📝 Extracts phenotypes, demographics, family history, prior testing, and gene mentions from clinical notes
- 🖼️ Incorporates phenotype evidence from medical images when available
- 🧬 Accepts structured HPO terms and external genomic-ranking results
- 🔗 Grounds retrieval and reasoning in RareGraph, HPO, MONDO, OMIM, Orphanet, and GeneReviews-derived knowledge
- 🤖 Uses complementary agents for evidence auditing, pairwise adjudication, and group/subtype reconciliation
- 📊 Preserves every major rank transition in an auditable trajectory
- 🩺 Converts the final Top-10 disease groups into normalized next tests, referrals, and initial evaluations

RareMind is designed to move beyond a single Top-K accuracy number. The accompanying clinical-impact analyses evaluate:

1. ⏱️ **Diagnostic lead time and healthcare utilization**
2. 🤝 **Alignment with clinician differential diagnoses**
3. 🛟 **Recovery when the clinician differential misses the eventual diagnosis**
4. 🚫 **Deprioritization of diagnoses explicitly ruled out in the evaluated note**
5. 🧪 **Concordance between recommended and subsequently documented tests**

---

## 🧭 The 10-stage pipeline

| Stage | Purpose | Principal output |
|---:|---|---|
| 1 | Multimodal clinical extraction | Phenotypes, demographics, family history, testing, genes, and image evidence |
| 2 | Ontology normalization and post-processing | Normalized HPO evidence, temporal context, inheritance, incongruity |
| 3 | Candidate retrieval and composite scoring | Broad RareGraph-grounded disease ranking |
| 4 | Conditional frontier consultation | Targeted review of ambiguous, incongruous, or under-supported cases |
| 5 | Evidence audit | Supporting, contradicting, and missing expected evidence |
| 6 | Pairwise adjudication | Subtype- and group-level candidate comparisons |
| 7 | Rank aggregation | Aggregated subtype and disease-group rankings |
| 8 | Group/subtype reconciliation and final fusion | Calibrated global `final_rank` |
| 9 | Clinical scorecard | Human-readable evidence cards for the leading diagnoses |
| 10 | Next-test and next-step synthesis | Ten normalized, cross-diagnosis clinical actions with provenance |

Submitting a patient through `scripts/run_pipeline.py` runs the complete sequence from Stage 1 through Stage 10. Stage 10 is enabled by default.

---

## 🩺 Stage 10: from diagnosis to action

Stage 10 applies the same action-normalization strategy used in the retrospective next-test concordance analysis:

```text
Global final ranking
        ↓
Disease groups represented within final ranks 1–10
        ↓
RareGraph: testing + initial_evaluations
        ↓
Lexical canonicalization
        ↓
BioLORD semantic clustering (cosine similarity ≥ 0.90)
        ↓
Rank by number of supporting disease groups
        ↓
Top 10 next tests / evaluations / referrals
```

Examples of deterministic normalization include:

- `echo` / `echocardiography` → `echocardiogram`
- `6MWT` → `six minute walk test`
- `whole-exome sequencing` → `exome sequencing`
- `brain natriuretic peptide` → `BNP`

Semantically similar actions that remain lexically different are clustered with the already loaded BioLORD model. Each recommendation retains:

- its normalized label and observed aliases;
- the number of distinct disease groups supporting it;
- the best final rank among its supporting groups;
- the RareGraph field(s) that supplied it;
- the supporting group and representative disease IDs and names.

This stage generates recommendations from the model output alone. Reference future tests are used only for retrospective evaluation—not during patient inference.

### Stage 10 configuration

```yaml
next_steps:
  enabled: true
  disease_top_k: 10
  action_top_k: 10
  action_fields:
    - testing
    - initial_evaluations
  cluster_similarity_threshold: 0.90
```

---

## 🚀 Running RareMind

### 1. Prepare a dataset folder

At least one supported modality must be present. Clinical text is the usual starting point.

```text
inputs/
└── demo/
    ├── text/
    │   └── PATIENT_001.txt
    ├── free_hpo/
    │   └── PATIENT_001.txt          # optional
    ├── image/
    │   └── PATIENT_001.png          # optional
    └── vcf/
        └── PATIENT_001.vcf          # optional
```

The filename stem is the case identifier shared across modalities.

### 2. Configure local resources

Edit [`configs/default.yaml`](configs/default.yaml) to point to:

- the RareGraph JSON;
- the MONDO hierarchy and cross-mappings;
- the HPO ontology;
- local text and vision models;
- cache and output locations.

The default BioLORD model is `FremyCompany/BioLORD-2023`.

### 3. Run one patient

```bash
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --dataset demo \
  --case_id PATIENT_001
```

### 4. Run every discovered patient

```bash
python scripts/run_pipeline.py \
  --config configs/default.yaml \
  --dataset demo
```

On a SLURM cluster, the supplied wrapper submits the same complete Stage 1–10
workflow; no separate next-step command is required:

```bash
sbatch -p gpuq --gres=gpu:a100:1 \
  --wrap="bash scripts/run_agents.sh --dataset demo --input_dir inputs --output_dir outputs"
```

Useful overrides:

```bash
# Test a different local text model
python scripts/run_pipeline.py --dataset demo --text_model Qwen/Qwen3-8B

# Process only the first ten discovered cases
python scripts/run_pipeline.py --dataset demo --limit 10

# Recompute Stage 1 caches
python scripts/run_pipeline.py --dataset demo --overwrite_stage1_cache
```

> A Linux environment with an NVIDIA GPU is recommended for the local vLLM stages. BioLORD action normalization uses `sentence-transformers` and reuses the embedder already loaded for ontology normalization.

---

## 📦 Per-patient outputs

```text
outputs/<dataset>/<case_id>/
├── stage1_extraction.json
├── stage2_patient_evidence.json
├── stage3_composite_ranking.tsv
├── stage5_audit_results.json
├── stage5_ranking_after_audit.tsv
├── stage6_pairwise_results_subtype.json
├── stage6_pairwise_results_group.json
├── stage7_reranked_subtype.tsv
├── stage7_reranked_group.tsv
├── stage8_reconciled_ranking.tsv
├── stage8_final_fusion.tsv
├── stage8_reconciled.json
├── stage9_scorecard.json
├── stage9_scorecard.txt
├── stage10_next_steps.tsv
└── rank_trajectory.tsv
```

The dataset-level `summary.tsv` includes the leading diagnosis, its final rank, the highest-ranked next step, and the complete Top-10 action list.

### Stage 10 output

**`stage10_next_steps.tsv`** is the single canonical Stage 10 artifact. It
contains the normalized Top-10 actions, aliases, supporting disease groups and
diseases, source RareGraph fields, ranking support, normalization metadata, and
the selected source disease groups.

---

## 🔬 Clinical-impact analyses

The [`clinical_impact_analyses`](clinical_impact_analyses) workspace contains the retrospective evaluation notebooks and reusable functions for:

- earliest computationally recoverable diagnostic signal;
- utilization accumulated before formal genetic-testing recommendation;
- clinician-differential alignment and missed-diagnosis recovery;
- note-aligned ruled-out negative controls;
- next-test lexical and BioLORD semantic concordance;
- publication-ready figures and review tables.

These analyses are intentionally separated from patient inference. Cohort labels and future documented tests never enter the RareMind ranking or Stage 10 recommendation process.

---

## 🧱 Repository map

```text
src/raremind/                  Canonical RareMind implementation
├── agents/                    Extraction agents
├── normalize/                 HPO, MONDO, temporal, and BioLORD normalization
├── kg/                        RareGraph loading and indexing
├── retrieval/                 Candidate-generation channels
├── scoring/                   Deterministic composite scoring
├── reasoning/                 Audit, adjudication, fusion, scorecard, Stage 10
└── orchestration/             End-to-end patient runner

scripts/run_pipeline.py        Main 10-stage command-line entry point
configs/default.yaml           Runtime and Stage 10 configuration
clinical_impact_analyses/      Retrospective clinical-impact evaluations
```

New code should import the pipeline from `raremind`. A minimal historical
`raregraph` compatibility namespace is retained to avoid abruptly breaking
existing environments; **RareGraph** otherwise refers only to the knowledge
graph.

---

## 🛡️ Intended use

RareMind is a research decision-support system. Its rankings and suggested actions are not medical advice, are not a substitute for clinical judgment, and require review in the context of the complete patient record, prior testing, local practice, and test availability.

For protected health information, use only approved infrastructure and follow institutional privacy, security, and data-governance requirements.

---

## 🌟 Citation 📜

The full RareMind manuscript is currently in preparation. Until it becomes
available, if you use or reference RareMind, RareGraph, or this repository,
please cite our published work:

**Nguyen QM, Wang K.**  
RareGraph-AgenticAI: A Multimodal Knowledge Graph and Multi-agent LLM Framework for Rare Disease Evaluation and Gene Prioritization.  
In: *Artificial Intelligence in Medicine*. Springer Nature Switzerland; 2026:445–450.  
https://doi.org/10.1007/978-3-032-30813-9_82

### BibTeX

```bibtex
@inproceedings{nguyen2026raregraph,
  author    = {Nguyen, Quan M. and Wang, Kai},
  title     = {RareGraph-AgenticAI: A Multimodal Knowledge Graph and Multi-agent LLM Framework for Rare Disease Evaluation and Gene Prioritization},
  booktitle = {Artificial Intelligence in Medicine},
  year      = {2026},
  pages     = {445--450},
  publisher = {Springer Nature Switzerland},
  doi       = {10.1007/978-3-032-30813-9_82}
}