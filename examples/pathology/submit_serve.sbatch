#!/bin/bash
#SBATCH --job-name=pathology-serve
#SBATCH --partition=gpuA100x4
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#
# Stages one CONCH or MUSK server for the whole event: teams hit it
# concurrently via client.py instead of each submitting their own batch job.
# One instance handles ~10-20 teams fine (serve.py micro-batches concurrent
# requests before each GPU forward pass) - for a much larger cohort, submit
# this script more than once (different --job-name) and split teams across
# the resulting endpoints.
#
# Usage: sbatch --account=<your_account> submit_serve.sbatch [model]
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
CONNECTION_DIR="$HOME/.pathology/serve/$SLURM_JOB_ID"

mkdir -p logs
(umask 077 && mkdir -p "$CONNECTION_DIR")

# --- activate environment (edit for your setup) ---
# module load anaconda3_gpu
# conda activate pathology
# --- or ---
# source /path/to/venv/bin/activate

cd "$(dirname "$0")"

export PATHOLOGY_MODEL="$MODEL"
export API_KEY="pathology-$(head -c 16 /dev/urandom | od -An -tx1 | tr -d ' \n')"

# Find a free port on this compute node.
find_free_port() {
    local port=$1
    while ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do
        port=$((port + 1))
    done
    echo "$port"
}
PORT=$(find_free_port 8000)

# Remove the connection file (contains the API key) if the job ends for any
# reason, including scancel (SIGTERM).
CONNECTION_FILE="$CONNECTION_DIR/connection.json"
trap 'rm -f "$CONNECTION_FILE"; kill $SERVER_PID 2>/dev/null || true' EXIT TERM INT

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

(umask 077 && cat > "$CONNECTION_FILE" <<EOF
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
chmod 600 "$CONNECTION_FILE"

echo ""
echo "Pathology server is up:"
echo "  Endpoint: http://$(hostname):$PORT"
echo "  Model:    $MODEL"
echo "  API key:  $API_KEY"
echo "  Connection file: $CONNECTION_FILE"
echo ""
echo "Share the endpoint + API key with teams (e.g. via a shared doc or the"
echo "channel used to announce the hackathon), or have a TA read $CONNECTION_FILE."
echo ""
echo "Example team usage:"
echo "  python client.py --endpoint http://$(hostname):$PORT --api-key $API_KEY tile.png"

wait $SERVER_PID
