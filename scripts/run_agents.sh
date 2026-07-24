#!/usr/bin/env bash
# =============================================================================
# run_agents.sh  -  SLURM wrapper for run_pipeline.py
#
# Runs ALL cases found under <input_dir>/<dataset>/{free_hpo,text,vcf,image}/
# by default. Use --case_id only if you want to debug a single case.
#
# Submit example:
#
#   sbatch -p gpuq \
#          --gres=gpu:a100:1 \
#          --cpus-per-gpu=1 \
#          --mem-per-cpu=20G \
#          --time=5-00:00:00 \
#          --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                    --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                    --dataset HMS \
#                    --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs \
#                    --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                    --text_model Qwen/Qwen3-8B"
#
# Interactive test:
#   bash scripts/run_agents.sh --input_dir inputs --dataset HMS --case_id case001 --verbose
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Defaults - override via command-line arguments
# ---------------------------------------------------------------------------
CONFIG="configs/default.yaml"
DATASET=""             # REQUIRED: e.g. HMS
INPUT_DIR="inputs"     # parent of the dataset folder, e.g. .../inputs
OUTPUT_DIR="outputs"
CASE_ID=""             # leave empty to run ALL cases
LIMIT=""               # leave empty for no limit

GENOMICS_RESULTS_DIR=""
GENOMICS_COMMAND=""
TEXT_MODEL=""          # e.g. Qwen/Qwen3-8B

# Boolean flags
VERBOSE=""
NO_VISION_PREFETCH=""
OVERWRITE_VISION_CACHE=""
NO_STAGE1_PREFETCH=""
OVERWRITE_STAGE1_CACHE=""

# Continuing
SKIP_COMPLETED=""
COMPLETION_FILE=""
ARRAY_INDEX=""
ARRAY_COUNT=""

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
usage() {
    cat <<EOF
Usage: bash run_agents.sh [OPTIONS]

Required:
  --dataset NAME               Dataset folder name inside input_dir, e.g. HMS
  --input_dir PATH             Parent directory of the dataset folder.
                               Pipeline reads <input_dir>/<dataset>/free_hpo|text|vcf|image/

Optional:
  --output_dir PATH            Where to write results   (default: outputs)
  --config PATH                Config YAML path         (default: configs/default.yaml)
  --limit N                    Run only first N cases   (default: all)
  --case_id ID                 Debug: run ONE case only (default: run all)
  --genomics_results_dir PATH  Dir with Exomiser/RankVar outputs
  --genomics_command CMD       Async command template for missing VCF results
  --text_model NAME            Override cfg.models.text_llm.model_name
                               e.g. Qwen/Qwen3-8B
  --no_vision_prefetch         Skip vision prefetch; run inline instead
  --overwrite_vision_cache     Re-run vision even if cached JSON exists
  --no_stage1_prefetch         Skip batched Stage 1 text extraction
  --overwrite_stage1_cache     Re-run Stage 1 text even if cached JSON exists
  --verbose                    Enable debug-level logging
  -h, --help                   Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)                 CONFIG="$2";                shift 2 ;;
        --dataset)                DATASET="$2";               shift 2 ;;
        --input_dir)              INPUT_DIR="$2";             shift 2 ;;
        --output_dir)             OUTPUT_DIR="$2";            shift 2 ;;
        --case_id)                CASE_ID="$2";               shift 2 ;;
        --limit)                  LIMIT="$2";                 shift 2 ;;
        --genomics_results_dir)   GENOMICS_RESULTS_DIR="$2";  shift 2 ;;
        --genomics_command)       GENOMICS_COMMAND="$2";      shift 2 ;;
        --text_model)             TEXT_MODEL="$2";            shift 2 ;;
        --no_vision_prefetch)     NO_VISION_PREFETCH="1";     shift   ;;
        --overwrite_vision_cache) OVERWRITE_VISION_CACHE="1"; shift   ;;
        --no_stage1_prefetch)     NO_STAGE1_PREFETCH="1";     shift   ;;
        --overwrite_stage1_cache) OVERWRITE_STAGE1_CACHE="1"; shift   ;;
        --verbose)                VERBOSE="1";                shift   ;;
        --skip_completed) SKIP_COMPLETED="1"; shift ;;
        --completion_file) COMPLETION_FILE="$2"; shift 2 ;;
        --array_index) ARRAY_INDEX="$2"; shift 2 ;;
        --array_count) ARRAY_COUNT="$2"; shift 2 ;;
        -h|--help)                usage; exit 0               ;;
        *) echo "[ERROR] Unknown argument: $1"; usage; exit 1 ;;
    esac
done

if [[ -z "$DATASET" ]]; then
    echo "[ERROR] --dataset is required."
    usage
    exit 1
fi

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------
echo "======================================================="
echo " RareGraph Pipeline"
echo " Dataset : $DATASET"
echo " Input   : $INPUT_DIR/$DATASET"
echo " Output  : $OUTPUT_DIR/$DATASET"
echo " Repo    : $REPO_ROOT"
echo " Started : $(date)"
echo " Host    : $(hostname)"
echo " SLURM job ID: ${SLURM_JOB_ID:-<interactive>}"
echo "======================================================="

if command -v nvidia-smi &>/dev/null; then
    echo "--- GPU ---"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo "-----------"
fi

# ---------------------------------------------------------------------------
# Build the python argument list
# ---------------------------------------------------------------------------
PYTHON_ARGS=(
    --config     "$CONFIG"
    --input_dir  "$INPUT_DIR"
    --dataset    "$DATASET"
    --output_dir "$OUTPUT_DIR"
)

[[ -n "$CASE_ID"              ]] && PYTHON_ARGS+=(--case_id             "$CASE_ID")
[[ -n "$LIMIT"                ]] && PYTHON_ARGS+=(--limit               "$LIMIT")
[[ -n "$GENOMICS_RESULTS_DIR" ]] && PYTHON_ARGS+=(--genomics_results_dir "$GENOMICS_RESULTS_DIR")
[[ -n "$GENOMICS_COMMAND"     ]] && PYTHON_ARGS+=(--genomics_command    "$GENOMICS_COMMAND")
[[ -n "$TEXT_MODEL"           ]] && PYTHON_ARGS+=(--text_model          "$TEXT_MODEL")

[[ -n "$NO_VISION_PREFETCH"     ]] && PYTHON_ARGS+=(--no_vision_prefetch)
[[ -n "$OVERWRITE_VISION_CACHE" ]] && PYTHON_ARGS+=(--overwrite_vision_cache)
[[ -n "$NO_STAGE1_PREFETCH"     ]] && PYTHON_ARGS+=(--no_stage1_prefetch)
[[ -n "$OVERWRITE_STAGE1_CACHE" ]] && PYTHON_ARGS+=(--overwrite_stage1_cache)
[[ -n "$VERBOSE"                ]] && PYTHON_ARGS+=(--verbose)
[[ -n "$SKIP_COMPLETED" ]] && PYTHON_ARGS+=(--skip_completed)
[[ -n "$COMPLETION_FILE" ]] && PYTHON_ARGS+=(--completion_file "$COMPLETION_FILE")
[[ -n "$ARRAY_INDEX" ]] && PYTHON_ARGS+=(--array_index "$ARRAY_INDEX")
[[ -n "$ARRAY_COUNT" ]] && PYTHON_ARGS+=(--array_count "$ARRAY_COUNT")

echo ""
echo "Command: python scripts/run_pipeline.py ${PYTHON_ARGS[*]}"
echo ""

# ---------------------------------------------------------------------------
# Run the pipeline
# ---------------------------------------------------------------------------
python scripts/run_pipeline.py "${PYTHON_ARGS[@]}"
EXIT_CODE=$?

echo ""
echo "======================================================="
echo " Finished : $(date)"
echo " Exit code: $EXIT_CODE"
echo "======================================================="
exit $EXIT_CODE


# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset HMS \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma27b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-27b-it"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset MME \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma27b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-27b-it"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#        --array=1-2 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset LIRICAL \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma27b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-27b-it --skip_completed"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=9-00:00:00 \
#         --array=1-2 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset RAMEDIS \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma27b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-27b-it --skip_completed"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset HMS \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma4b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-4b-it"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset MME \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma4b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-4b-it"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset LIRICAL \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma4b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-4b-it"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset RAMEDIS \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs_medgemma4b \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-4b-it"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset GMDB_text_part1 \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model google/medgemma-27b-it"

# sbatch -p gpu-xe9680q \
#         --gres=gpu:h100:1 \
#         --cpus-per-gpu=1 \
#         --mem-per-cpu=20G \
#         --time=5-00:00:00 \
#         --wrap="bash /home/nguyenqm/projects/MultiAgentLLM/scripts/run_agents.sh \
#                 --input_dir /home/nguyenqm/projects/rare_dx_mcp/inputs \
#                 --dataset GMDB_text_part2 \
#                 --output_dir /home/nguyenqm/projects/MultiAgentLLM/outputs \
#                 --config /home/nguyenqm/projects/MultiAgentLLM/configs/default.yaml \
#                 --text_model Qwen/Qwen3-8B"