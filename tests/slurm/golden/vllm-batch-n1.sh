#!/bin/bash
#SBATCH --job-name=golden-vllm-batch
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
mkdir -p $DATA_INPUT_DIR $DATA_OUTPUT_DIR $MODELS_DIR $LOGS_DIR $CONTAINERS_DIR $APPTAINER_TMPDIR $APPTAINER_CACHEDIR $HF_HOME $VLLM_HOME $VLLM_MODELS $XDG_CACHE_HOME $FLASHINFER_WORKSPACE_BASE 

# Start vLLM server

echo hf_home: $HF_HOME
echo vllm home: $VLLM_HOME
echo vllm models: $VLLM_MODELS

# Build container if needed (or if forced)
if [ "$LLMFLUX_FORCE_REBUILD" = "1" ] || [ ! -f "$CONTAINERS_DIR/llm_processor.sif" ]; then
    echo "Building Container in ${CONTAINERS_DIR}"
    apptainer build --force $CONTAINERS_DIR/llm_processor.sif $CONTAINER_DEF
    echo Container built successfully: ${?}
fi

# start serverwith clean environment
# All APPTAINERENV_* variables are automatically passed in (prefix removed)
# VLLM_MODEL_NAME contains the HuggingFace model name
echo "Using HuggingFace model: $VLLM_MODEL_NAME"
echo "Port: $VLLM_PORT"

APPTAINER_BIND_PATHS="$DATA_INPUT_DIR:/app/data/input,$DATA_OUTPUT_DIR:/app/data/output,$MODELS_DIR:/app/models,$LOGS_DIR:/app/logs,$VLLM_HOME:$VLLM_HOME,$HF_HOME:$HF_HOME,$XDG_CACHE_HOME:$XDG_CACHE_HOME,$FLASHINFER_WORKSPACE_BASE:$FLASHINFER_WORKSPACE_BASE"
if [ -d "$VLLM_MODEL_NAME" ]; then
    echo "Bind-mounting local model path: $VLLM_MODEL_NAME"
    APPTAINER_BIND_PATHS="$APPTAINER_BIND_PATHS,$VLLM_MODEL_NAME:$VLLM_MODEL_NAME"
fi

# Set HF_TOKEN for vLLM if available (for gated models)
if [ -n "$HF_TOKEN" ]; then
    echo "HuggingFace token detected - will use for model authentication"
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi

# Pass SLURM-allocated GPU(s) into the container (--cleanenv strips CUDA_VISIBLE_DEVICES)
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
VERBOSE=1 apptainer exec --nv --cleanenv \
    --bind "$APPTAINER_BIND_PATHS" \
    ${CONTAINERS_DIR}/llm_processor.sif \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$VLLM_MODEL_NAME" \
        --host "$VLLM_HOST" \
        --port "$VLLM_PORT" \
        $VLLM_ENGINE_ARGS &
VLLM_PID=$!
echo "        PID: $VLLM_PID"
echo ""
# Wait for server
echo "[3/4] Waiting for server to be ready...":
SERVER_READY=false
for i in {1..300}; do
    if curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
        echo "        ✓ Server ready!"
        SERVER_READY=true
        break
    fi
    if ! ps -p $VLLM_PID > /dev/null; then
        echo "VLLM server died"
        exit 1
    fi
    [ $((i %15)) -eq 0 ] && echo "        Still loading... ($i/300s)"
    sleep 1
done
if [[ $SERVER_READY == "false" ]]; then
    echo "Server failed to load!"
    exit 1
else
    echo "Server has started!"
fi
# Check if model exists, try to pull if it doesn't
echo ""
# Test inference
echo TEST
echo Time to ask questions!

# Run processor
python3 -c "
import sys
import os
sys.path.append('$PROJECT_ROOT')
from llmflux.core.config import Config
from llmflux.processors import BatchProcessor

# Ensure VLLM environment variables are available in Python
vllm_port = os.environ.get('VLLM_PORT')
if vllm_port:
    os.environ['VLLM_HOST'] = f'http://localhost:{vllm_port}'

# Load model configuration
config = Config()
model_identifier = os.environ.get('APPTAINERENV_MODEL_IDENTIFIER', 'Llama-3.2-3B-Instruct')
engine = os.environ.get('APPTAINERENV_ENGINE', 'vllm')
custom_config_path = os.environ.get('APPTAINERENV_CUSTOM_CONFIG_PATH') or None

try:
    model_config = config.load_model_config(model_identifier, custom_config_path=custom_config_path)
    model_config.engine = engine
except Exception as e:
    print(f'Error loading model config for {model_identifier}: {e}')
    model_config = config.load_model_config('Llama-3.2-3B-Instruct')
    model_config.engine = engine

print('Create batch processor with JSONL input')
batch_processor = BatchProcessor(
    model_config=model_config,
    batch_size=int(os.environ.get('BATCH_SIZE', '4')),
    save_frequency=int(os.environ.get('SAVE_FREQUENCY', '50')),
    max_retries=int(os.environ.get('MAX_RETRIES', '3')),
    retry_delay=float(os.environ.get('RETRY_DELAY', '1.0'))
)
print(f'batch processor: {batch_processor}')

# Prepare run kwargs
run_kwargs = {}

# Add any other kwargs from environment variables
for key in ['max_tokens', 'temperature', 'top_p', 'top_k']:
    if key.upper() in os.environ:
        run_kwargs[key] = os.environ[key.upper()]

batch_processor.run('/data/in.jsonl', '/data/out.json', 'vllm', **run_kwargs)
"
# Capture the processor's status before cleanup overwrites $?.
LLMFLUX_PROC_RC=$?

# Cleanup
pkill -f "vllm serve" || true
# Only remove temporary directories that we created
if [ -d "$APPTAINER_TMPDIR" ] && [ -w "$APPTAINER_TMPDIR" ]; then
    rm -rf "$APPTAINER_TMPDIR"
fi
if [ -d "$APPTAINER_CACHEDIR" ] && [ -w "$APPTAINER_CACHEDIR" ]; then
    rm -rf "$APPTAINER_CACHEDIR"
fi
echo "Cleaning up..."
kill $VLLM_PID 2>/dev/null || true
sleep 2
kill -9 $VLLM_PID 2>/dev/null || true

# Exit with the processor's status. Without this the script exits with
# whatever cleanup returned, so a run whose every item failed still
# reports success and the job looks complete.
exit ${LLMFLUX_PROC_RC:-0}
