"""Multi-node launch for the vLLM engine.

Emits the bash that turns a `--nodes N` allocation into an actual N-node vLLM
deployment, instead of a server on the first node and N-1 idle nodes.

Shape: one SPMD `srun` step, one task per node. Rank 0 runs the API server;
ranks 1..N-1 run headless workers. Ranks find each other through vLLM's native
rendezvous flags (`--nnodes`, `--node-rank`, `--master-addr`, `--master-port`,
registered at `engine/arg_utils.py:776-779`), so no Ray cluster and no container
change are needed. Tensor parallelism stays within a node; pipeline parallelism
spans them.

The per-rank script runs on the HOST, outside the container, because
`apptainer --cleanenv` strips every `SLURM_*` variable — a rank cannot work out
which rank it is from inside.

Four hazards this code exists to avoid, each measured on Delta or DeltaAI rather
than assumed:

1. **A hostname is not a usable master address.** On these compute nodes an FQDN
   resolves to IPv6 link-local (`fe80::`) *only*, while the short name gives the
   routable IPv4 — and bare `hostname` returns the FQDN, so
   `getent hosts $(hostname)` silently yields garbage. Separately, some node
   names carry several A records and round-robin between calls, so two ranks
   resolving independently can disagree. The master address is therefore derived
   once, on the head node, from the fabric interface, and propagated.
2. **`hsn` as a prefix over-matches.** Nodes with two Slingshot NICs also expose
   VLAN children (`hsn0.561`), so a prefix match selects four interfaces where
   two are wanted. Interfaces are matched exactly as `hsn<digits>`.
3. **vLLM's own `get_ip()` route-probes 8.8.8.8** and returns the routed VLAN
   address rather than the fabric one, so `VLLM_HOST_IP` is pinned per rank.
4. **`VLLM_ALLREDUCE_USE_SYMM_MEM` defaults on** and deadlocks cross-node during
   engine init, so it is defaulted off (overridable).
"""

#: Marker prefixes. Greppable in job output, and asserted by tests.
STAGE_A = "LLMFLUX-STAGE-A"
ERROR = "LLMFLUX-ERROR"

#: Heredoc delimiters. Registered with the shell-safety tests as SHELL bodies so
#: their contents get their own `bash -n` — the outer parse cannot see them.
LIB_DELIM = "LLMFLUX_LIB_EOF"
RANK_DELIM = "LLMFLUX_RANK_EOF"


def _library() -> list:
    """Helper functions, written to a file and sourced by both sides.

    One definition, used by the batch body and by every rank, so the head node
    and the workers cannot disagree about how an address is derived.
    """
    return [
        "llmflux_diagnostics() {",
        '    echo "LLMFLUX-DIAG: host=$(hostname -s) job=${SLURM_JOB_ID:-?} '
        'nodes=${SLURM_JOB_NUM_NODES:-?} nodelist=${SLURM_JOB_NODELIST:-?}"',
        '    echo "LLMFLUX-DIAG: run_dir=$LLMFLUX_RUN_DIR fstype=$(stat -f -c %T '
        '"$LLMFLUX_RUN_DIR" 2>/dev/null || echo unknown)"',
        '    echo "LLMFLUX-DIAG: if run_dir is node-local, set LLMFLUX_LOGS_DIR to shared storage"',
        '    echo "LLMFLUX-DIAG: ranks started=$(ls "$LLMFLUX_RUN_DIR/rendezvous" '
        '2>/dev/null | wc -l)/${LLMFLUX_NNODES:-?}"',
        "    ip -4 -o addr show 2>/dev/null | sed 's/^/LLMFLUX-DIAG: iface /'",
        "}",
        "",
        "llmflux_die() {",
        '    echo "' + ERROR + ': $*" >&2',
        "    llmflux_diagnostics >&2",
        "    # Tear the srun step down rather than orphaning it. Without this a",
        "    # barrier failure leaves ranks running with the allocation held for",
        "    # the full walltime — the same waste this launcher exists to end.",
        '    if [ -n "${VLLM_PID:-}" ]; then',
        '        kill "$VLLM_PID" 2>/dev/null || true',
        "    fi",
        "    exit 1",
        "}",
        "",
        "# True only for a dotted-quad with four octets each <= 255.",
        "llmflux_is_ipv4() {",
        '    case "$1" in',
        "        *[!0-9.]*|'') return 1 ;;",
        "    esac",
        "    local IFS=. octet count=0",
        '    for octet in $1; do',
        '        [ -n "$octet" ] || return 1',
        '        [ "$octet" -le 255 ] 2>/dev/null || return 1',
        "        count=$((count + 1))",
        "    done",
        '    [ "$count" -eq 4 ]',
        "}",
        "",
        "# This node's routable IPv4 on the fabric. Never returns a hostname:",
        "# an FQDN here resolves to IPv6 link-local only, and some node names",
        "# round-robin over several A records.",
        "llmflux_node_ip() {",
        '    local want="${LLMFLUX_HSN_IFACE:-hsn}" ip',
        "    # Exact 'hsn<digits>' — a bare 'hsn' prefix would also match the",
        "    # VLAN children (hsn0.561) and pick a non-fabric address.",
        "    ip=$(ip -4 -o addr show 2>/dev/null \\",
        '         | awk -v w="$want" \'$2 ~ "^" w "[0-9]+$" { split($4, a, "/"); print a[1]; exit }\')',
        '    if llmflux_is_ipv4 "$ip"; then printf \'%s\\n\' "$ip"; return 0; fi',
        '    echo "LLMFLUX-WARN: no ${want}<n> interface; falling back to short-name lookup" >&2',
        '    ip=$(getent ahostsv4 "$(hostname -s)" 2>/dev/null | awk \'{print $1; exit}\')',
        '    case "$ip" in 127.*|169.254.*) ip="" ;; esac',
        '    if llmflux_is_ipv4 "$ip"; then printf \'%s\\n\' "$ip"; return 0; fi',
        "    return 1",
        "}",
        "",
        "# Every fabric NIC, comma-separated, for NCCL_SOCKET_IFNAME. Derived",
        "# rather than hardcoded: node types differ (measured 4 NICs on GH200,",
        "# 2 on H200, 1 on A100), and a fixed list silently leaves fabric unused",
        "# on the wider nodes. Exact hsn<digits> again, so VLAN children are out.",
        "llmflux_fabric_ifaces() {",
        '    local want="${LLMFLUX_HSN_IFACE:-hsn}"',
        "    ip -4 -o addr show 2>/dev/null \\",
        '        | awk -v w="$want" \'$2 ~ "^" w "[0-9]+$" '
        '{ printf "%s%s", sep, $2; sep="," }\'',
        "}",
        "",
        "# First free TCP port at or above $1. Fails loudly rather than",
        "# returning a possibly-occupied port when ss is unavailable.",
        "llmflux_free_port() {",
        '    local port="$1" tries="${LLMFLUX_PORT_SCAN_MAX:-64}" n=0',
        "    command -v ss >/dev/null 2>&1 || {",
        '        echo "' + ERROR + ': ss not found; cannot verify a free port" >&2; return 1; }',
        '    while [ "$n" -lt "$tries" ]; do',
        '        if ! ss -tlnH 2>/dev/null | awk \'{print $4}\' | grep -q ":${port}$"; then',
        "            printf '%s\\n' \"$port\"; return 0",
        "        fi",
        "        port=$((port + 1)); n=$((n + 1))",
        "    done",
        '    echo "' + ERROR + ': no free port within $tries of $1" >&2',
        "    return 1",
        "}",
        "",
        "# Seconds we may wait, clamped to 90% of the allocation's remaining",
        "# time. Without this a readiness budget longer than the walltime is",
        "# never reached: Slurm kills the job first, so the diagnostics and the",
        "# non-zero exit never happen and the failure looks like a timeout.",
        "llmflux_deadline_budget() {",
        '    local want="$1" left cap',
        '    [ -n "${SLURM_JOB_END_TIME:-}" ] || { printf \'%s\\n\' "$want"; return 0; }',
        "    left=$(( SLURM_JOB_END_TIME - $(date +%s) ))",
        '    [ "$left" -gt 0 ] 2>/dev/null || { printf \'%s\\n\' "$want"; return 0; }',
        "    cap=$(( left * 9 / 10 ))",
        '    if [ "$want" -gt "$cap" ]; then',
        '        echo "LLMFLUX-WARN: budget ${want}s exceeds the allocation; clamping to ${cap}s" >&2',
        '        want="$cap"',
        "    fi",
        "    printf '%s\\n' \"$want\"",
        "}",
    ]


def _rank_script(nnodes: str) -> list:
    """The per-rank launcher, run once per node inside the srun step."""
    return [
        "#!/bin/bash",
        "# Runs on the HOST, outside the container: --cleanenv strips SLURM_*,",
        "# so a rank cannot determine its own rank from inside.",
        "set -u",
        '. "$LLMFLUX_RUN_DIR/llmflux-lib.sh"',
        "",
        'LLMFLUX_RANK="${SLURM_PROCID:-0}"',
        '# Announce before doing anything slow, so the barrier separates "never',
        '# placed" from "placed and still loading".',
        'touch "$LLMFLUX_RUN_DIR/rendezvous/rank-${LLMFLUX_RANK}.started"',
        'echo "' + STAGE_A + ': rank ${LLMFLUX_RANK} up on $(hostname -s)"',
        "",
        "# --cleanenv drops these unless they are re-exported per rank.",
        'APPTAINERENV_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"',
        "export APPTAINERENV_CUDA_VISIBLE_DEVICES",
        "",
        "# VLLM_PORT must NOT reach the container. vLLM seeds every internal ZMQ",
        "# port from it and walks upward (utils/network_utils.py:_get_open_port),",
        "# probing with bind() on THIS node only. Each node therefore walks the",
        "# same sequence independently and two ranks on different nodes pick the",
        "# same port for a shared endpoint:",
        "#     zmq.error.ZMQError: Address already in use (tcp://...:53675)",
        "# Unset, so vLLM falls back to ephemeral ports that are genuinely free",
        "# per node. Rank 0's API port is still pinned, on the command line.",
        "unset APPTAINERENV_VLLM_PORT",
        "",
        "# vLLM's get_ip() route-probes 8.8.8.8 and returns the routed VLAN",
        "# address, not the fabric one. Pin what this rank advertises.",
        "LLMFLUX_RANK_IP=$(llmflux_node_ip) || llmflux_die \"rank ${LLMFLUX_RANK}: no routable fabric IPv4\"",
        'APPTAINERENV_VLLM_HOST_IP="$LLMFLUX_RANK_IP"',
        "export APPTAINERENV_VLLM_HOST_IP",
        "",
        "# Derived per node: these node types carry 1, 2 or 4 fabric NICs.",
        'LLMFLUX_IFACES="${LLMFLUX_NCCL_SOCKET_IFNAME:-$(llmflux_fabric_ifaces)}"',
        '[ -n "$LLMFLUX_IFACES" ] || llmflux_die "rank ${LLMFLUX_RANK}: no fabric interfaces found"',
        'APPTAINERENV_NCCL_SOCKET_IFNAME="$LLMFLUX_IFACES"',
        "export APPTAINERENV_NCCL_SOCKET_IFNAME",
        "# Symmetric-memory allreduce deadlocks cross-node during engine init.",
        'APPTAINERENV_VLLM_ALLREDUCE_USE_SYMM_MEM="${LLMFLUX_SYMM_MEM:-0}"',
        "export APPTAINERENV_VLLM_ALLREDUCE_USE_SYMM_MEM",
        "",
        "# Rank 0 serves the API; the rest are headless workers. The headless",
        "# branch lives in vLLM's `vllm serve` CLI, not in the api_server module",
        "# entrypoint, so both ranks must use the console script.",
        'if [ "$LLMFLUX_RANK" = "0" ]; then',
        "    exec apptainer exec --nv --cleanenv \\",
        '        --bind "$APPTAINER_BIND_PATHS" \\',
        '        "${CONTAINERS_DIR}/llm_processor.sif" \\',
        '        vllm serve "$VLLM_MODEL_NAME" \\',
        '            --host "$VLLM_HOST" --port "$VLLM_PORT" \\',
        '            --nnodes "$LLMFLUX_NNODES" --node-rank 0 \\',
        '            --master-addr "$LLMFLUX_MASTER_ADDR" \\',
        '            --master-port "$LLMFLUX_MASTER_PORT" \\',
        "            $VLLM_ENGINE_ARGS",
        "else",
        "    exec apptainer exec --nv --cleanenv \\",
        '        --bind "$APPTAINER_BIND_PATHS" \\',
        '        "${CONTAINERS_DIR}/llm_processor.sif" \\',
        '        vllm serve "$VLLM_MODEL_NAME" --headless \\',
        '            --nnodes "$LLMFLUX_NNODES" \\',
        '            --node-rank "$LLMFLUX_RANK" \\',
        '            --master-addr "$LLMFLUX_MASTER_ADDR" \\',
        '            --master-port "$LLMFLUX_MASTER_PORT" \\',
        "            $VLLM_ENGINE_ARGS",
        "fi",
    ]


def preamble(nodes: str, gpus_per_node: str) -> list:
    """Rendezvous setup, emitted before the launch. Head node only."""
    lines = [
        "",
        "# ============ multi-node rendezvous ============",
        f"LLMFLUX_NNODES={nodes}",
        "# SLURM_RESTART_COUNT keeps a requeued job (same job ID) from seeing",
        "# the previous attempt's rendezvous files and passing the barrier",
        "# instantly with zero ranks actually up.",
        'LLMFLUX_RUN_TAG="${SLURM_JOB_ID:-nojob}.${SLURM_RESTART_COUNT:-0}"',
        'LLMFLUX_RUN_DIR="$LOGS_DIR/llmflux-$LLMFLUX_RUN_TAG"',
        "export LLMFLUX_NNODES LLMFLUX_RUN_DIR",
        'rm -rf "$LLMFLUX_RUN_DIR/rendezvous"',
        'mkdir -p "$LLMFLUX_RUN_DIR/rendezvous" || { echo "' + ERROR + ': cannot create '
        '$LLMFLUX_RUN_DIR (must be on storage every node can see)" >&2; exit 1; }',
        "",
        f'cat > "$LLMFLUX_RUN_DIR/llmflux-lib.sh" <<\'{LIB_DELIM}\'',
        *_library(),
        LIB_DELIM,
        '. "$LLMFLUX_RUN_DIR/llmflux-lib.sh"',
        "",
        "# The allocation must match what the topology planned; --sbatch-arg can",
        "# append a second --nodes and sbatch honours the last one.",
        'if [ "${SLURM_JOB_NUM_NODES:-1}" != "$LLMFLUX_NNODES" ]; then',
        '    llmflux_die "allocation has ${SLURM_JOB_NUM_NODES:-1} node(s), '
        'expected $LLMFLUX_NNODES"',
        "fi",
        "",
        "# Resolved ONCE here and propagated. Assignment-then-export: writing",
        "# `export VAR=$(cmd)` would discard the command's exit status, since",
        "# export itself always succeeds.",
        "LLMFLUX_MASTER_ADDR=$(llmflux_node_ip) || \\",
        '    llmflux_die "no routable fabric IPv4 on the head node; set LLMFLUX_HSN_IFACE"',
        "export LLMFLUX_MASTER_ADDR",
        'LLMFLUX_MASTER_PORT=$(llmflux_free_port "${LLMFLUX_RDZV_PORT:-29500}") || \\',
        '    llmflux_die "no free rendezvous port on the head node"',
        "export LLMFLUX_MASTER_PORT",
        "",
        'echo "LLMFLUX-TOPOLOGY: nnodes=$LLMFLUX_NNODES tp=' + str(gpus_per_node) + " "
        'master=$LLMFLUX_MASTER_ADDR:$LLMFLUX_MASTER_PORT"',
        "# Record the fabric the ranks will actually use. Nodes here carry more",
        "# than one Slingshot NIC on separate subnets, so which interface each",
        "# rank binds is not obvious from the master address alone.",
        "ip -4 -o addr show 2>/dev/null | awk '$2 ~ /^hsn/ {print \"LLMFLUX-TOPOLOGY: iface \" $2 \" \" $4}'",
        'echo "LLMFLUX-TOPOLOGY: nccl_socket_ifname=${LLMFLUX_NCCL_SOCKET_IFNAME:-$(llmflux_fabric_ifaces)}"',
        "",
        f'cat > "$LLMFLUX_RUN_DIR/rank-launch.sh" <<\'{RANK_DELIM}\'',
        *_rank_script(nodes),
        RANK_DELIM,
        'chmod +x "$LLMFLUX_RUN_DIR/rank-launch.sh"',
    ]
    return lines


def launch() -> list:
    """The srun step plus the rank barrier. Replaces the single-node exec."""
    return [
        "",
        "# One SPMD step, one task per node. --overlap is deliberately absent:",
        "# measured on a full-width 2-node allocation, the step starts in 0s",
        "# because the batch step holds CPUs=0 and contends for nothing. The",
        "# barrier below is the guard in case that is not true elsewhere.",
        "# --kill-on-bad-exit=1 because Slurm defaults it off, and without it a",
        "# dead remote rank leaves the srun client alive and the liveness check",
        "# below inert.",
        'srun --nodes="$LLMFLUX_NNODES" --ntasks="$LLMFLUX_NNODES" \\',
        "     --ntasks-per-node=1 --kill-on-bad-exit=1 \\",
        '     "$LLMFLUX_RUN_DIR/rank-launch.sh" &',
        "VLLM_PID=$!",
        'echo "        PID: $VLLM_PID"',
        "",
        "# Stage A: every rank reached exec. Bounded, and separates 'the step",
        "# never placed' from 'the model is loading', which look identical from",
        "# the readiness probe alone.",
        'LLMFLUX_RANK_DEADLINE=$(llmflux_deadline_budget "${LLMFLUX_RANK_START_TIMEOUT:-300}")',
        "LLMFLUX_WAITED=0",
        'while [ "$(ls "$LLMFLUX_RUN_DIR/rendezvous" 2>/dev/null | wc -l)" -lt "$LLMFLUX_NNODES" ]; do',
        "    if ! ps -p $VLLM_PID > /dev/null 2>&1; then",
        '        llmflux_die "srun step exited before all ranks started"',
        "    fi",
        '    if [ "$LLMFLUX_WAITED" -ge "$LLMFLUX_RANK_DEADLINE" ]; then',
        '        llmflux_die "only $(ls "$LLMFLUX_RUN_DIR/rendezvous" 2>/dev/null | wc -l) '
        'of $LLMFLUX_NNODES ranks started within ${LLMFLUX_RANK_DEADLINE}s"',
        "    fi",
        "    LLMFLUX_WAITED=$((LLMFLUX_WAITED + 1))",
        "    sleep 1",
        "done",
        'echo "' + STAGE_A + ': all $LLMFLUX_NNODES ranks launched"',
        "",
        "# Clamp the readiness bound to the allocation. A large model sharded",
        "# across nodes legitimately takes far longer to load than the 300s",
        "# single-node default, so this must be raised — but a bound longer than",
        "# the walltime is never reached: Slurm kills the job first, so the",
        "# diagnostics and the non-zero exit never happen and a load failure",
        "# looks identical to a timeout.",
        'LLMFLUX_SERVER_TIMEOUT=$(llmflux_deadline_budget "${LLMFLUX_SERVER_TIMEOUT:-1800}")',
        "export LLMFLUX_SERVER_TIMEOUT",
        'echo "LLMFLUX-TOPOLOGY: readiness_budget=${LLMFLUX_SERVER_TIMEOUT}s"',
        "# ============ end multi-node rendezvous ============",
        "",
    ]
