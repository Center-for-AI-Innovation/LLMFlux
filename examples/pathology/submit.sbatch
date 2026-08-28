#!/bin/bash
#SBATCH --job-name=pathology-embed
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Usage:
#   sbatch --account=<your_account> submit.sbatch <input_dir> <output_csv> <model> [text_prompts.json]
#
# <model> is "conch", "musk", "test", or a spec for a model with no bespoke
# adapter: "openclip:<hf-repo>" (e.g. openclip:wisdomik/QuiltNet-B-32),
# "openclip:<arch>@<hf-repo>/<filename>" (e.g.
# openclip:ViT-B-16@jamessyx/PathGen-CLIP/pathgenclip.pt), or
# "hfclip:<hf-repo>" (e.g. hfclip:openai/clip-vit-base-patch32). See
# models.py's resolve_adapter() / README.md.
#
# Standalone job — does not use SlurmRunner or any LLMFlux engine. Adjust the
# environment activation below to however you installed requirements.txt plus
# CONCH/MUSK (conda env, venv, or module load); this script does not assume one.
set -euo pipefail

INPUT_DIR="$1"
OUTPUT_CSV="$2"
MODEL="${3:-conch}"
TEXT_PROMPTS="${4:-}"

mkdir -p logs

# --- activate environment (edit for your setup) ---
# module load anaconda3_gpu
# conda activate pathology
# --- or ---
# source /path/to/venv/bin/activate

cd "$(dirname "$0")"

EXTRA_ARGS=()
if [ -n "$TEXT_PROMPTS" ]; then
    EXTRA_ARGS+=(--text-prompts "$TEXT_PROMPTS")
fi

python run_embeddings.py \
    --model "$MODEL" \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT_CSV" \
    "${EXTRA_ARGS[@]}"
