#!/bin/bash
#SBATCH --job-name=test-ollama-batch
#SBATCH --account=myaccount
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/logs/%j.out
#SBATCH --error=/logs/%j.err

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

# Run processor
/opt/llmflux/bin/python3 -c "
import os
from llmflux.core.config import Config
from llmflux.processors import BatchProcessor

# Ensure OLLAMA environment variables are available in Python
ollama_port = os.environ.get('OLLAMA_PORT')
if ollama_port:
    os.environ['OLLAMA_HOST'] = f'http://localhost:{ollama_port}'

# Load model configuration
config = Config()
model_identifier = os.environ.get('APPTAINERENV_MODEL_IDENTIFIER', 'Llama-3.2-3B-Instruct')
engine = os.environ.get('APPTAINERENV_ENGINE', 'ollama')

try:
    model_config = config.load_model_config(model_identifier)
    model_config.engine = engine
except Exception as e:
    print(f'Error loading model config for {model_identifier}: {e}')
    model_config = config.load_model_config('Llama-3.2-3B-Instruct')
    model_config.engine = engine

# Create batch processor with JSONL input
batch_processor = BatchProcessor(
    model_config=model_config,
    batch_size=int(os.environ.get('BATCH_SIZE', '4')),
    save_frequency=int(os.environ.get('SAVE_FREQUENCY', '50')),
    max_retries=int(os.environ.get('MAX_RETRIES', '3')),
    retry_delay=float(os.environ.get('RETRY_DELAY', '1.0'))
)

# Prepare run kwargs
run_kwargs = {}

# Add any other kwargs from environment variables
for key in ['max_tokens', 'temperature', 'top_p', 'top_k']:
    if key.upper() in os.environ:
        run_kwargs[key] = os.environ[key.upper()]

batch_processor.run('/data/in.jsonl', '/data/out.json', 'ollama', **run_kwargs)
"
# Capture the processor's status before cleanup overwrites $?.
BATCH_RC=$?

# Cleanup
pkill -f "ollama serve" || true
# Only remove temporary directories that we created
if [ -d "$APPTAINER_TMPDIR" ] && [ -w "$APPTAINER_TMPDIR" ]; then
    rm -rf "$APPTAINER_TMPDIR"
fi
if [ -d "$APPTAINER_CACHEDIR" ] && [ -w "$APPTAINER_CACHEDIR" ]; then
    rm -rf "$APPTAINER_CACHEDIR"
fi
# Exit with the processor's status. Without this the script exits with
# whatever cleanup returned, so a run whose every item failed still
# reports success and the job looks complete.
exit ${BATCH_RC:-0}
