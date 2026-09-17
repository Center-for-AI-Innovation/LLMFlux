#!/bin/bash
#SBATCH --job-name=pathology-serve
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --time=47:55:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --signal=B:USR1@180
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Stages one CONCH/MUSK/CLIP server: teams hit it concurrently via client.py
# instead of each submitting their own batch job. One instance handles
# ~10-20 teams fine (serve.py micro-batches concurrent requests before each
# GPU forward pass) - for a much larger cohort, submit this script more than
# once (different --job-name) and split teams across the resulting endpoints.
#
# Auto-resubmit chain: Delta's gpuA100x4 partition caps a single job at 48h
# (see docs.ncsa.illinois.edu Delta user guide, "Running Jobs"), which is
# shorter than a multi-day event (e.g. Friday afternoon - Monday EOD). This
# script resubmits itself shortly before hitting its own walltime, so the
# service keeps running across many jobs without a human resubmitting it by
# hand. Each link in the chain:
#   - pins to the same compute node (--nodelist) and reuses the account and
#     --reservation of the job it replaces (read via `scontrol show job`,
#     not an assumed env var), so it lands back on the same reserved
#     hardware
#   - reuses the same API key (carried forward via the API_KEY env var), so
#     students don't need a new key every ~48h
#   - reuses the same port where possible (the new job starts only after the
#     old one has fully exited and freed it, via --dependency=afterany)
#   - writes connection info to a per-job path (audit trail) AND a stable
#     path that never moves for the life of the whole chain, namespaced by
#     --job-name so a second scaled-out instance doesn't collide with it:
#     ~/.pathology/serve/current-<job-name>/connection.json (default job
#     name is "pathology-serve") -- give teams THIS path (or the
#     host/port/key it contains), not the per-job one, since the per-job one
#     disappears when that link's job ends.
# This resubmit only happens for the walltime-approaching signal (SIGUSR1,
# see --signal above) -- a manual `scancel` sends SIGTERM/SIGINT instead,
# which cleans up without resubmitting. See "stop the chain for good" below.
#
# Usage:
#   sbatch --account=<your_account> [--reservation=<name>] submit_serve.sbatch [model]
#
# To stop the chain for good instead of letting it restart at the next
# walltime boundary:
#   touch ~/.pathology/serve/STOP_CHAIN-<job-name>
# (leave the current job running until the event ends, or `scancel` it
# immediately if you want it down right away -- either way, once that file
# exists no further link of THIS job-name's chain gets submitted; a second
# scaled-out instance with a different --job-name keeps running). Remove the
# file before the next event or it'll block that chain too.
#
# [model] is "conch", "musk", "test", or a spec for a model with no bespoke
# adapter: "openclip:<hf-repo>", "openclip:<arch>@<hf-repo>/<filename>", or
# "hfclip:<hf-repo>". See models.py's resolve_adapter() / README.md.
#
# Standalone job — does not use SlurmRunner or any LLMFlux engine. Adjust the
# environment activation below to however you installed requirements.txt plus
# CONCH/MUSK (conda env, venv, or module load); this script does not assume one.
set -euo pipefail

MODEL="${1:-conch}"
# Namespaced by job name, not just a global "current"/"STOP_CHAIN" -- if
# you're running a second instance for scaling (different --job-name, see
# README.md's "Scaling beyond one instance"), it must not collide with or
# stop the first one's chain.
STOP_FILE="$HOME/.pathology/serve/STOP_CHAIN-$SLURM_JOB_NAME"
CONNECTION_DIR="$HOME/.pathology/serve/$SLURM_JOB_ID"
CURRENT_DIR="$HOME/.pathology/serve/current-$SLURM_JOB_NAME"

mkdir -p logs
(umask 077 && mkdir -p "$CONNECTION_DIR" "$CURRENT_DIR")

# --- activate environment (edit for your setup) ---
# module load anaconda3_gpu
# conda activate pathology
# --- or ---
# source /path/to/venv/bin/activate

cd "$(dirname "$0")"

export PATHOLOGY_MODEL="$MODEL"
# Reuse the API key handed out at the start of the chain if this job is a
# resubmission (carried forward via the environment); otherwise mint one.
API_KEY="${API_KEY:-pathology-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')}"
export API_KEY

# Find a free port on this compute node.
find_free_port() {
    local port=$1
    while ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do
        port=$((port + 1))
    done
    echo "$port"
}
PORT=$(find_free_port 8000)

CONNECTION_FILE="$CONNECTION_DIR/connection.json"
CURRENT_FILE="$CURRENT_DIR/connection.json"

write_connection_file() {
    local target=$1
    (umask 077 && cat > "$target" <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "node": "$(hostname)",
  "port": $PORT,
  "model": "$MODEL",
  "api_key": "$API_KEY",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    )
    chmod 600 "$target"
}

# scontrol, not an env var, is the source of truth for account/reservation:
# SLURM_JOB_RESERVATION isn't guaranteed to be set in every Slurm version,
# and this only runs once per link so the extra call is cheap.
job_field() {
    scontrol show job "$SLURM_JOB_ID" | tr ' ' '\n' | sed -n "s/^$1=//p" | head -1
}

resubmit_chain() {
    if [ -f "$STOP_FILE" ]; then
        echo "STOP_CHAIN present at $STOP_FILE, not resubmitting."
        return
    fi
    local account reservation args
    account=$(job_field Account)
    reservation=$(job_field Reservation)
    args=(--job-name="$SLURM_JOB_NAME" --account="$account"
          --nodelist="$(hostname)" --dependency="afterany:$SLURM_JOB_ID"
          --export="ALL")
    if [ -n "$reservation" ] && [ "$reservation" != "(null)" ]; then
        args+=(--reservation="$reservation")
    fi
    echo "Walltime approaching, resubmitting chain: sbatch ${args[*]} $0 $MODEL"
    sbatch "${args[@]}" "$0" "$MODEL"
}

# Only the walltime-approaching signal (SIGUSR1) resubmits. A manual
# scancel/kill (SIGTERM/SIGINT) just cleans up -- that's the intentional
# "stop the chain for good" path.
trap 'resubmit_chain' USR1
trap 'rm -f "$CONNECTION_FILE" "$CURRENT_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT TERM INT

uvicorn serve:app --host 0.0.0.0 --port "$PORT" --workers 1 &
SERVER_PID=$!

echo "Waiting for server to be ready..."
for i in $(seq 1 60); do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "Server ready on port $PORT"
        break
    fi
    if ! ps -p $SERVER_PID > /dev/null; then
        echo "Server process died during startup"
        exit 1
    fi
    sleep 5
done

write_connection_file "$CONNECTION_FILE"
write_connection_file "$CURRENT_FILE"

echo ""
echo "Pathology server is up:"
echo "  Endpoint: http://$(hostname):$PORT"
echo "  Model:    $MODEL"
echo "  API key:  $API_KEY"
echo "  Connection file (this job only): $CONNECTION_FILE"
echo "  Connection file (stable across the whole chain): $CURRENT_FILE"
echo ""
echo "Give teams the STABLE path above (or its host/port/key), not the"
echo "per-job one -- it's what keeps working when this chain auto-resubmits"
echo "across a walltime boundary."
echo ""
echo "Example team usage:"
echo "  python client.py --endpoint http://$(hostname):$PORT --api-key $API_KEY tile.png"

wait $SERVER_PID
