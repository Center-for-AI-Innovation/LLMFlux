#!/bin/bash
#SBATCH --job-name=test-vllm-serve
#SBATCH --account=myaccount
#SBATCH --partition=gpu
#SBATCH --nodes=2
#SBATCH --gpus-per-node=2
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=/logs/%j.out
#SBATCH --error=/logs/%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=someone@example.edu
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

export APPTAINER_BIND_PATHS="$DATA_INPUT_DIR:/app/data/input,$DATA_OUTPUT_DIR:/app/data/output,$MODELS_DIR:/app/models,$LOGS_DIR:/app/logs,$VLLM_HOME:$VLLM_HOME,$HF_HOME:$HF_HOME,$XDG_CACHE_HOME:$XDG_CACHE_HOME,$FLASHINFER_WORKSPACE_BASE:$FLASHINFER_WORKSPACE_BASE"
if [ -d "$VLLM_MODEL_NAME" ]; then
    echo "Bind-mounting local model path: $VLLM_MODEL_NAME"
    export APPTAINER_BIND_PATHS="$APPTAINER_BIND_PATHS,$VLLM_MODEL_NAME:$VLLM_MODEL_NAME"
fi

# Set HF_TOKEN for vLLM if available (for gated models)
if [ -n "$HF_TOKEN" ]; then
    echo "HuggingFace token detected - will use for model authentication"
    export APPTAINERENV_HF_TOKEN="$HF_TOKEN"
fi

# Pass SLURM-allocated GPU(s) into the container (--cleanenv strips CUDA_VISIBLE_DEVICES)
export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# Find a consecutive free port on this compute node
find_free_port() {
    local port=$1
    while ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; do
        port=$((port + 1))
    done
    echo $port
}
VLLM_PORT=$(find_free_port ${VLLM_PORT:-8000})
export APPTAINERENV_VLLM_PORT=$VLLM_PORT
echo "Using port: $VLLM_PORT"

# Install cleanup trap early so the connection file (contains the API key)
# and server are removed even if the job is cancelled with scancel (SIGTERM).
CONNECTION_FILE="$HOME/.llmflux/serve/$SLURM_JOB_ID/connection.json"
trap 'rm -f "$CONNECTION_FILE"; pkill -f "vllm serve" || true' EXIT TERM INT


# ============ multi-node rendezvous ============
LLMFLUX_NNODES=2
# SLURM_RESTART_COUNT keeps a requeued job (same job ID) from seeing
# the previous attempt's rendezvous files and passing the barrier
# instantly with zero ranks actually up.
LLMFLUX_RUN_TAG="${SLURM_JOB_ID:-nojob}.${SLURM_RESTART_COUNT:-0}"
LLMFLUX_RUN_DIR="$LOGS_DIR/llmflux-$LLMFLUX_RUN_TAG"
export LLMFLUX_NNODES LLMFLUX_RUN_DIR
rm -rf "$LLMFLUX_RUN_DIR/rendezvous"
mkdir -p "$LLMFLUX_RUN_DIR/rendezvous" || { echo "LLMFLUX-ERROR: cannot create $LLMFLUX_RUN_DIR (must be on storage every node can see)" >&2; exit 1; }

cat > "$LLMFLUX_RUN_DIR/llmflux-lib.sh" <<'LLMFLUX_LIB_EOF'
llmflux_diagnostics() {
    echo "LLMFLUX-DIAG: host=$(hostname -s) job=${SLURM_JOB_ID:-?} nodes=${SLURM_JOB_NUM_NODES:-?} nodelist=${SLURM_JOB_NODELIST:-?}"
    echo "LLMFLUX-DIAG: run_dir=$LLMFLUX_RUN_DIR fstype=$(stat -f -c %T "$LLMFLUX_RUN_DIR" 2>/dev/null || echo unknown)"
    echo "LLMFLUX-DIAG: if run_dir is node-local, set LLMFLUX_LOGS_DIR to shared storage"
    echo "LLMFLUX-DIAG: ranks started=$(ls "$LLMFLUX_RUN_DIR/rendezvous" 2>/dev/null | wc -l)/${LLMFLUX_NNODES:-?}"
    ip -4 -o addr show 2>/dev/null | sed 's/^/LLMFLUX-DIAG: iface /'
}

llmflux_die() {
    echo "LLMFLUX-ERROR: $*" >&2
    llmflux_diagnostics >&2
    # Tear the srun step down rather than orphaning it. Without this a
    # barrier failure leaves ranks running with the allocation held for
    # the full walltime — the same waste this launcher exists to end.
    if [ -n "${VLLM_PID:-}" ]; then
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    exit 1
}

# True only for a dotted-quad with four octets each <= 255.
llmflux_is_ipv4() {
    case "$1" in
        *[!0-9.]*|'') return 1 ;;
    esac
    local IFS=. octet count=0
    for octet in $1; do
        [ -n "$octet" ] || return 1
        [ "$octet" -le 255 ] 2>/dev/null || return 1
        count=$((count + 1))
    done
    [ "$count" -eq 4 ]
}

# This node's routable IPv4 on the fabric. Never returns a hostname:
# an FQDN here resolves to IPv6 link-local only, and some node names
# round-robin over several A records.
llmflux_node_ip() {
    local want="${LLMFLUX_HSN_IFACE:-hsn}" ip
    # Exact 'hsn<digits>' — a bare 'hsn' prefix would also match the
    # VLAN children (hsn0.561) and pick a non-fabric address.
    ip=$(ip -4 -o addr show 2>/dev/null \
         | awk -v w="$want" '$2 ~ "^" w "[0-9]+$" { split($4, a, "/"); print a[1]; exit }')
    if llmflux_is_ipv4 "$ip"; then printf '%s\n' "$ip"; return 0; fi
    echo "LLMFLUX-WARN: no ${want}<n> interface; falling back to short-name lookup" >&2
    ip=$(getent ahostsv4 "$(hostname -s)" 2>/dev/null | awk '{print $1; exit}')
    case "$ip" in 127.*|169.254.*) ip="" ;; esac
    if llmflux_is_ipv4 "$ip"; then printf '%s\n' "$ip"; return 0; fi
    return 1
}

# Every fabric NIC, comma-separated, for NCCL_SOCKET_IFNAME. Derived
# rather than hardcoded: node types differ (measured 4 NICs on GH200,
# 2 on H200, 1 on A100), and a fixed list silently leaves fabric unused
# on the wider nodes. Exact hsn<digits> again, so VLAN children are out.
llmflux_fabric_ifaces() {
    local want="${LLMFLUX_HSN_IFACE:-hsn}"
    ip -4 -o addr show 2>/dev/null \
        | awk -v w="$want" '$2 ~ "^" w "[0-9]+$" { printf "%s%s", sep, $2; sep="," }'
}

# First free TCP port at or above $1. Fails loudly rather than
# returning a possibly-occupied port when ss is unavailable.
llmflux_free_port() {
    local port="$1" tries="${LLMFLUX_PORT_SCAN_MAX:-64}" n=0
    command -v ss >/dev/null 2>&1 || {
        echo "LLMFLUX-ERROR: ss not found; cannot verify a free port" >&2; return 1; }
    while [ "$n" -lt "$tries" ]; do
        if ! ss -tlnH 2>/dev/null | awk '{print $4}' | grep -q ":${port}$"; then
            printf '%s\n' "$port"; return 0
        fi
        port=$((port + 1)); n=$((n + 1))
    done
    echo "LLMFLUX-ERROR: no free port within $tries of $1" >&2
    return 1
}

# Seconds we may wait, clamped to 90% of the allocation's remaining
# time. Without this a readiness budget longer than the walltime is
# never reached: Slurm kills the job first, so the diagnostics and the
# non-zero exit never happen and the failure looks like a timeout.
llmflux_deadline_budget() {
    local want="$1" left cap
    [ -n "${SLURM_JOB_END_TIME:-}" ] || { printf '%s\n' "$want"; return 0; }
    left=$(( SLURM_JOB_END_TIME - $(date +%s) ))
    [ "$left" -gt 0 ] 2>/dev/null || { printf '%s\n' "$want"; return 0; }
    cap=$(( left * 9 / 10 ))
    if [ "$want" -gt "$cap" ]; then
        echo "LLMFLUX-WARN: budget ${want}s exceeds the allocation; clamping to ${cap}s" >&2
        want="$cap"
    fi
    printf '%s\n' "$want"
}
LLMFLUX_LIB_EOF
. "$LLMFLUX_RUN_DIR/llmflux-lib.sh"

# The allocation must match what the topology planned; --sbatch-arg can
# append a second --nodes and sbatch honours the last one.
if [ "${SLURM_JOB_NUM_NODES:-1}" != "$LLMFLUX_NNODES" ]; then
    llmflux_die "allocation has ${SLURM_JOB_NUM_NODES:-1} node(s), expected $LLMFLUX_NNODES"
fi

# Resolved ONCE here and propagated. Assignment-then-export: writing
# `export VAR=$(cmd)` would discard the command's exit status, since
# export itself always succeeds.
LLMFLUX_MASTER_ADDR=$(llmflux_node_ip) || \
    llmflux_die "no routable fabric IPv4 on the head node; set LLMFLUX_HSN_IFACE"
export LLMFLUX_MASTER_ADDR
LLMFLUX_MASTER_PORT=$(llmflux_free_port "${LLMFLUX_RDZV_PORT:-29500}") || \
    llmflux_die "no free rendezvous port on the head node"
export LLMFLUX_MASTER_PORT

echo "LLMFLUX-TOPOLOGY: nnodes=$LLMFLUX_NNODES tp=2 master=$LLMFLUX_MASTER_ADDR:$LLMFLUX_MASTER_PORT"
# Record the fabric the ranks will actually use. Nodes here carry more
# than one Slingshot NIC on separate subnets, so which interface each
# rank binds is not obvious from the master address alone.
ip -4 -o addr show 2>/dev/null | awk '$2 ~ /^hsn/ {print "LLMFLUX-TOPOLOGY: iface " $2 " " $4}'
echo "LLMFLUX-TOPOLOGY: nccl_socket_ifname=${LLMFLUX_NCCL_SOCKET_IFNAME:-$(llmflux_fabric_ifaces)}"

cat > "$LLMFLUX_RUN_DIR/rank-launch.sh" <<'LLMFLUX_RANK_EOF'
#!/bin/bash
# Runs on the HOST, outside the container: --cleanenv strips SLURM_*,
# so a rank cannot determine its own rank from inside.
set -u
. "$LLMFLUX_RUN_DIR/llmflux-lib.sh"

LLMFLUX_RANK="${SLURM_PROCID:-0}"
# Announce before doing anything slow, so the barrier separates "never
# placed" from "placed and still loading".
touch "$LLMFLUX_RUN_DIR/rendezvous/rank-${LLMFLUX_RANK}.started"
echo "LLMFLUX-STAGE-A: rank ${LLMFLUX_RANK} up on $(hostname -s)"

# --cleanenv drops these unless they are re-exported per rank.
APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export APPTAINERENV_CUDA_VISIBLE_DEVICES

# VLLM_PORT must NOT reach the container. vLLM seeds every internal ZMQ
# port from it and walks upward (utils/network_utils.py:_get_open_port),
# probing with bind() on THIS node only. Each node therefore walks the
# same sequence independently and two ranks on different nodes pick the
# same port for a shared endpoint:
#     zmq.error.ZMQError: Address already in use (tcp://...:53675)
# Unset, so vLLM falls back to ephemeral ports that are genuinely free
# per node. Rank 0's API port is still pinned, on the command line.
unset APPTAINERENV_VLLM_PORT

# vLLM's get_ip() route-probes 8.8.8.8 and returns the routed VLAN
# address, not the fabric one. Pin what this rank advertises.
LLMFLUX_RANK_IP=$(llmflux_node_ip) || llmflux_die "rank ${LLMFLUX_RANK}: no routable fabric IPv4"
APPTAINERENV_VLLM_HOST_IP="$LLMFLUX_RANK_IP"
export APPTAINERENV_VLLM_HOST_IP

# Derived per node: these node types carry 1, 2 or 4 fabric NICs.
LLMFLUX_IFACES="${LLMFLUX_NCCL_SOCKET_IFNAME:-$(llmflux_fabric_ifaces)}"
[ -n "$LLMFLUX_IFACES" ] || llmflux_die "rank ${LLMFLUX_RANK}: no fabric interfaces found"
APPTAINERENV_NCCL_SOCKET_IFNAME="$LLMFLUX_IFACES"
export APPTAINERENV_NCCL_SOCKET_IFNAME
# Symmetric-memory allreduce deadlocks cross-node during engine init.
APPTAINERENV_VLLM_ALLREDUCE_USE_SYMM_MEM="${LLMFLUX_SYMM_MEM:-0}"
export APPTAINERENV_VLLM_ALLREDUCE_USE_SYMM_MEM

# Rank 0 serves the API; the rest are headless workers. The headless
# branch lives in vLLM's `vllm serve` CLI, not in the api_server module
# entrypoint, so both ranks must use the console script.
if [ "$LLMFLUX_RANK" = "0" ]; then
    exec apptainer exec --nv --cleanenv \
        --bind "$APPTAINER_BIND_PATHS" \
        "${CONTAINERS_DIR}/llm_processor.sif" \
        vllm serve "$VLLM_MODEL_NAME" \
            --host "$VLLM_HOST" --port "$VLLM_PORT" \
            --nnodes "$LLMFLUX_NNODES" --node-rank 0 \
            --master-addr "$LLMFLUX_MASTER_ADDR" \
            --master-port "$LLMFLUX_MASTER_PORT" \
            $VLLM_ENGINE_ARGS
else
    exec apptainer exec --nv --cleanenv \
        --bind "$APPTAINER_BIND_PATHS" \
        "${CONTAINERS_DIR}/llm_processor.sif" \
        vllm serve "$VLLM_MODEL_NAME" --headless \
            --nnodes "$LLMFLUX_NNODES" \
            --node-rank "$LLMFLUX_RANK" \
            --master-addr "$LLMFLUX_MASTER_ADDR" \
            --master-port "$LLMFLUX_MASTER_PORT" \
            $VLLM_ENGINE_ARGS
fi
LLMFLUX_RANK_EOF
chmod +x "$LLMFLUX_RUN_DIR/rank-launch.sh"

# One SPMD step, one task per node. --overlap is deliberately absent:
# measured on a full-width 2-node allocation, the step starts in 0s
# because the batch step holds CPUs=0 and contends for nothing. The
# barrier below is the guard in case that is not true elsewhere.
# --kill-on-bad-exit=1 because Slurm defaults it off, and without it a
# dead remote rank leaves the srun client alive and the liveness check
# below inert.
srun --nodes="$LLMFLUX_NNODES" --ntasks="$LLMFLUX_NNODES" \
     --ntasks-per-node=1 --kill-on-bad-exit=1 \
     "$LLMFLUX_RUN_DIR/rank-launch.sh" &
VLLM_PID=$!
echo "        PID: $VLLM_PID"

# Stage A: every rank reached exec. Bounded, and separates 'the step
# never placed' from 'the model is loading', which look identical from
# the readiness probe alone.
LLMFLUX_RANK_DEADLINE=$(llmflux_deadline_budget "${LLMFLUX_RANK_START_TIMEOUT:-300}")
LLMFLUX_WAITED=0
while [ "$(ls "$LLMFLUX_RUN_DIR/rendezvous" 2>/dev/null | wc -l)" -lt "$LLMFLUX_NNODES" ]; do
    if ! ps -p $VLLM_PID > /dev/null 2>&1; then
        llmflux_die "srun step exited before all ranks started"
    fi
    if [ "$LLMFLUX_WAITED" -ge "$LLMFLUX_RANK_DEADLINE" ]; then
        llmflux_die "only $(ls "$LLMFLUX_RUN_DIR/rendezvous" 2>/dev/null | wc -l) of $LLMFLUX_NNODES ranks started within ${LLMFLUX_RANK_DEADLINE}s"
    fi
    LLMFLUX_WAITED=$((LLMFLUX_WAITED + 1))
    sleep 1
done
echo "LLMFLUX-STAGE-A: all $LLMFLUX_NNODES ranks launched"
# ============ end multi-node rendezvous ============

echo ""
# Wait for server
echo "[3/4] Waiting for server to be ready...":
SERVER_READY=false
LLMFLUX_SERVER_TIMEOUT=${LLMFLUX_SERVER_TIMEOUT:-300}
for i in $(seq 1 $LLMFLUX_SERVER_TIMEOUT); do
    if curl -s "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
        echo "        ✓ Server ready!"
        SERVER_READY=true
        break
    fi
    if ! ps -p $VLLM_PID > /dev/null; then
        echo "VLLM server died"
        exit 1
    fi
    [ $((i %15)) -eq 0 ] && echo "        Still loading... ($i/${LLMFLUX_SERVER_TIMEOUT}s)"
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

# Write connection file for llmflux connect (restrict permissions — contains API key)
# CONNECTION_FILE was defined earlier alongside the cleanup trap.
(umask 077 && mkdir -p "$(dirname $CONNECTION_FILE)")
chmod 700 "$HOME/.llmflux" "$HOME/.llmflux/serve" "$(dirname $CONNECTION_FILE)"
(umask 077 && cat > "$CONNECTION_FILE" <<EOF
{
  "job_id": "$SLURM_JOB_ID",
  "node": "$(hostname)",
  "port": $VLLM_PORT,
  "model": "$VLLM_MODEL_NAME",
  "api_key": "$LLMFLUX_API_KEY",
  "engine": "vllm",
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
  Endpoint: http://$(hostname):$VLLM_PORT/v1
  API Key:  $LLMFLUX_API_KEY
  Model:    $VLLM_MODEL_NAME
  Engine:   vllm

Example usage:

  from openai import OpenAI
  client = OpenAI(base_url="http://$(hostname):$VLLM_PORT/v1", api_key="$LLMFLUX_API_KEY")
  response = client.chat.completions.create(
      model="$VLLM_MODEL_NAME",
      messages=[{"role": "user", "content": "Hello!"}]
  )

To get connection details again:
  llmflux connect $SLURM_JOB_ID
MAIL_EOF

# Keep alive until wall time or scancel
wait $VLLM_PID

# Cleanup
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

