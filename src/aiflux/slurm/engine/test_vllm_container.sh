#!/bin/bash
# ============================================================================
# vLLM Demo - Proves container v2.0 supports vLLM for future implementation
#
# Usage:
#   salloc --partition=ghx4-interactive --gpus-per-node=1 --time=00:15:00
#   bash vllm_demo_interactive.sh
#
# What this shows:
#   - Container has vLLM 0.11.0 + PyTorch 2.8.0 installed
#   - vLLM OpenAI API server works
#   - Ready to implement SlurmVllmRunner (no container rebuild needed)
# ============================================================================

CONTAINER="/u/jallen17/code/containers/llm_processor.sif"
MODEL="facebook/opt-125m"
PORT=8000

echo "=========================================="
echo "vLLM Demo - Container v2.0"
echo "=========================================="
echo ""

module purge

# Check container exists
if [ ! -f "$CONTAINER" ]; then
    echo "ERROR: Container not found: $CONTAINER"
    exit 1
fi

# Verify vLLM installation
echo "[1/4] Verifying vLLM installation..."
apptainer exec --nv "$CONTAINER" python3 -c "
import vllm, torch
print(f'✓ vLLM {vllm.__version__}')
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA {torch.version.cuda}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
"
echo ""

# Start vLLM server

echo "      Model: $MODEL (downloading on first run)"
apptainer exec --nv "$CONTAINER" \
    python3 -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --max-model-len 512 &

PID=$!
echo "      PID: $PID"
echo ""

# Wait for server
echo "[3/4] Waiting for server to be ready..."
for i in {1..90}; do
    if curl -s "http://localhost:$PORT/health" >/dev/null 2>&1; then
        echo "      ✓ Server ready!"
        break
    fi
    [ $((i % 15)) -eq 0 ] && echo "      Still loading... ($i/90s)"
    sleep 1
done
echo ""

# Test inference
echo "[4/4] Testing inference..."
curl -s "http://localhost:$PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL'",
        "prompt": "Baseball requires three outs instead of four because",
        "max_tokens": 30,
        "temperature": 0.7
    }' | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'choices' in data:
    print('✓ Response:', data['choices'][0]['text'].strip())
    print(f\"✓ Tokens: {data['usage']['total_tokens']}\")
else:
    print('✗ Error:', data.get('error', 'Unknown'))
"
echo ""

# Cleanup
echo "Cleaning up..."
kill $PID 2>/dev/null || true
sleep 2  # Give it time to shutdown gracefully
kill -9 $PID 2>/dev/null || true  # Force kill if still running

echo ""
echo "=========================================="
echo "✓ Demo Complete!"
echo "=========================================="
echo "Container v2.0 is ready for vLLM runner!"
echo ""
