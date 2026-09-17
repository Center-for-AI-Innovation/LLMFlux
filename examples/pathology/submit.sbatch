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
# Verified working on Delta: `module load pytorch-conda/2.8` provides
# everything in requirements.txt except open_clip_torch -- one-time
# `pip install --user "open_clip_torch>=2.24"` after loading the module
# covers that. See README.md § Setup.
module load pytorch-conda/2.8
# --- or, without that module ---
# source /path/to/venv/bin/activate

# No `cd` needed: Slurm starts a batch job in the directory sbatch was
# invoked from ($SLURM_SUBMIT_DIR), not the directory containing this
# script -- $0 here is a spooled copy of this file
# (/var/spool/slurmd/job<id>/slurm_script), so `cd "$(dirname "$0")"` would
# silently land you in the wrong place instead of erroring.

EXTRA_ARGS=()
if [ -n "$TEXT_PROMPTS" ]; then
    EXTRA_ARGS+=(--text-prompts "$TEXT_PROMPTS")
fi

python run_embeddings.py \
    --model "$MODEL" \
    --input-dir "$INPUT_DIR" \
    --output "$OUTPUT_CSV" \
    "${EXTRA_ARGS[@]}"
