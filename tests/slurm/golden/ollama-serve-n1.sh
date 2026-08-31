#!/bin/bash
#SBATCH --job-name=test-ollama-serve
#SBATCH --account=myaccount
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/logs/%j.out
#SBATCH --error=/logs/%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=someone@example.edu

# Create all necessary directories
mkdir -p $DATA_INPUT_DIR $DATA_OUTPUT_DIR $MODELS_DIR $LOGS_DIR $CONTAINERS_DIR $APPTAINER_TMPDIR $APPTAINER_CACHEDIR

# Start Ollama server
mkdir -p $OLLAMA_HOME $OLLAMA_MODELS

# Build container if needed (or if forced)
if [ "$LLMFLUX_FORCE_REBUILD" = "1" ] || [ ! -f "$CONTAINERS_DIR/llm_processor.sif" ]; then
    echo "Building Container in ${CONTAINERS_DIR}"
    export APPTAINER_DEBUG=1
    apptainer build --force $CONTAINERS_DIR/llm_processor.sif $CONTAINER_DEF
    echo Container built successfully: ${?}
fi

# Start server with clean environment
# All APPTAINERENV_* variables are automatically passed in (prefix removed)
# Find a consecutive free port on this compute node
find_free_port() {
    local port=$1
    while ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do
        port=$((port + 1))
    done
    echo $port
}
OLLAMA_PORT=$(find_free_port ${OLLAMA_PORT:-8000})
export APPTAINERENV_OLLAMA_PORT=$OLLAMA_PORT
export APPTAINERENV_OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"
export APPTAINERENV_OLLAMA_API_KEY="$LLMFLUX_API_KEY"
echo "Using port: $OLLAMA_PORT"

# Install cleanup trap early so the connection file (contains the API key)
# and server are removed even if the job is cancelled with scancel (SIGTERM).
CONNECTION_FILE="$HOME/.llmflux/serve/$SLURM_JOB_ID/connection.json"
trap 'rm -f "$CONNECTION_FILE"; [ -n "${OLLAMA_PID:-}" ] && kill "$OLLAMA_PID" 2>/dev/null; pkill -f "ollama serve" || true' EXIT TERM INT

# Pass SLURM-allocated GPU(s) into the container (--cleanenv strips
# CUDA_VISIBLE_DEVICES). Without this the container receives the list
# synthesised from the *requested* GPU count rather than the devices
# actually granted, and Ollama warns: "user overrode visible devices".
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
OLLAMA_DEBUG=1 apptainer exec --nv --cleanenv \
    --bind $DATA_INPUT_DIR:/app/data/input,$DATA_OUTPUT_DIR:/app/data/output,$MODELS_DIR:/app/models,$LOGS_DIR:/app/logs,$OLLAMA_HOME:$OLLAMA_HOME \
    $CONTAINERS_DIR/llm_processor.sif \
    ollama serve &

OLLAMA_PID=$!

# Wait for server
for i in {1..60}; do
    if curl -s "http://localhost:$OLLAMA_PORT/api/version" &>/dev/null; then
        echo "Ollama server started"
        break
    fi
    if ! ps -p $OLLAMA_PID > /dev/null; then
        echo "Ollama server died"
        exit 1
    fi
    echo "Waiting... ($i/60)"
    sleep 1
done

# Pull model if needed
MODEL_NAME="${OLLAMA_MODEL_NAME:-llama3.2:3b}"
echo "Checking if model ${MODEL_NAME} exists..."


# Check if model exists, try to pull if it doesn't
if ! curl -s "http://localhost:$OLLAMA_PORT/api/tags" | grep -q "\"name\":\"$MODEL_NAME\""; then
    echo "Model not found, pulling base model ${MODEL_NAME}..."
    curl -X POST "http://localhost:$OLLAMA_PORT/api/pull" -d '{"name": "'"$MODEL_NAME"'"}' -H "Content-Type: application/json"
    if [ $? -ne 0 ]; then
        echo "Failed to pull model ${MODEL_NAME}"
        echo "Available models:"
        curl -s "http://localhost:$OLLAMA_PORT/api/tags" | grep -o '"name":"[^"]*"' || echo "None found"
        exit 1
    else
        echo "Successfully pulled model ${MODEL_NAME}"
    fi
else
    echo "Model ${MODEL_NAME} already exists"
fi

# Write connection file for llmflux connect (restrict permissions — contains API key)
# CONNECTION_FILE was defined earlier alongside the cleanup trap.
(umask 077 && mkdir -p "$(dirname $CONNECTION_FILE)")
chmod 700 "$HOME/.llmflux" "$HOME/.llmflux/serve" "$(dirname $CONNECTION_FILE)"
LLMFLUX_ENDPOINT_HOST="$(hostname -s)"
(umask 077 && cat > "$CONNECTION_FILE" <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "node": "$LLMFLUX_ENDPOINT_HOST",
  "port": $OLLAMA_PORT,
  "model": "$OLLAMA_MODEL_NAME",
  "api_key": "$LLMFLUX_API_KEY",
  "engine": "ollama",
  "started_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
)
chmod 600 "$CONNECTION_FILE"
echo "Connection info written to $CONNECTION_FILE"

# Send ready notification email with connection details
LLMFLUX_EMAIL=someone@example.edu
mail -s "LLMFlux serve job $SLURM_JOB_ID is ready" -- "$LLMFLUX_EMAIL" <<MAIL_EOF
Your LLMFlux serve job has finished loading and is ready to use.

  Job ID:   $SLURM_JOB_ID
  Endpoint: http://$LLMFLUX_ENDPOINT_HOST:$OLLAMA_PORT/v1
  API Key:  $LLMFLUX_API_KEY
  Model:    $OLLAMA_MODEL_NAME
  Engine:   ollama

Example usage:

  from openai import OpenAI
  client = OpenAI(base_url="http://$LLMFLUX_ENDPOINT_HOST:$OLLAMA_PORT/v1", api_key="$LLMFLUX_API_KEY")
  response = client.chat.completions.create(
      model="$OLLAMA_MODEL_NAME",
      messages=[{"role": "user", "content": "Hello!"}]
  )

To get connection details again:
  llmflux connect $SLURM_JOB_ID
MAIL_EOF

# Keep alive until wall time or scancel
wait $OLLAMA_PID
LLMFLUX_SERVE_RC=$?
# scancel and Ctrl-C are how a serve job normally ends; those are not
# failures.
case "$LLMFLUX_SERVE_RC" in 130|143) LLMFLUX_SERVE_RC=0 ;; esac

# Cleanup
# Only remove temporary directories that we created
if [ -d "$APPTAINER_TMPDIR" ] && [ -w "$APPTAINER_TMPDIR" ]; then
    rm -rf "$APPTAINER_TMPDIR"
fi
if [ -d "$APPTAINER_CACHEDIR" ] && [ -w "$APPTAINER_CACHEDIR" ]; then
    rm -rf "$APPTAINER_CACHEDIR"
fi
# Exit with the server's status. Without this the script exits with
# whatever cleanup returned, so a serve job whose server died still
# reports success and the job looks complete.
exit ${LLMFLUX_SERVE_RC:-0}
