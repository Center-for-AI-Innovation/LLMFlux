# Multi-Node Inference

A model that does not fit in one node's GPU memory can be served across several.
`--nodes N` deploys the model across all `N` nodes; before this existed, `N` was
forwarded to `#SBATCH --nodes=` and then ignored, so the server started on the
first node and the remaining `N-1` sat idle for the job's full walltime.

**vLLM only.** Ollama has no comparable distributed-inference story, so
`--nodes > 1 --engine ollama` is rejected at submit time.

## When you need it

Only when the model does not fit on one node. Multi-node adds a network hop to
every pipeline stage boundary; a model that fits in one node is faster there.

```
model weights (bytes) ≈ parameters × bytes-per-parameter
        FP16/BF16 → 2 bytes    FP8 → 1 byte
```

plus KV cache and activation headroom — budget roughly 1.2–1.4× the weights.
A 72B model at BF16 is ~136 GB, which fits comfortably in a node with 4 × 96 GB
but not in one with 4 × 40 GB. The second case is what `--nodes 2` is for.

## How the parallelism maps

| Axis | Spans | Why |
|---|---|---|
| Tensor parallel (`--tensor-parallel-size`) | GPUs **within** a node | needs an all-reduce per layer; belongs on the intra-node links |
| Pipeline parallel (`--pipeline-parallel-size`) | **across** nodes | exchanges only activations at stage boundaries |

So `--nodes 2 --gpus-per-node 4` runs `tensor-parallel-size=4
pipeline-parallel-size=2`, world size 8. You do not set these yourself — they
are derived from the allocation. See *Overriding the derived parallelism* below
if you must.

## Usage

```bash
llmflux run --model Qwen2.5-72B-Instruct --engine vllm \
    --input prompts.jsonl --output results.json \
    --account <acct> --partition <partition> \
    --nodes 2 --gpus-per-node 4 --cpus-per-task 16 --time 01:30:00
```

`llmflux serve` takes the same flags and produces a multi-node endpoint;
`llmflux connect <job_id>` returns its address and API key as usual.

Before submitting, make sure the model weights are already in `HF_HOME`.
Downloading inside the allocation leaves every GPU idle for the transfer.

## How it works

One SPMD `srun` step, one task per node. Rank 0 serves the API; ranks 1..N-1 run
`vllm serve --headless`. Ranks meet through vLLM's own rendezvous flags — no Ray,
and no change to the container.

The launcher prints what it decided, which is the first thing to read in a job
log:

```
LLMFLUX-TOPOLOGY: nnodes=2 tp=4 master=172.28.86.10:29500
LLMFLUX-TOPOLOGY: iface hsn0      172.28.86.10/21
LLMFLUX-TOPOLOGY: iface hsn0.561  141.142.254.10/21
LLMFLUX-TOPOLOGY: nccl_socket_ifname=hsn0
LLMFLUX-TOPOLOGY: readiness_budget=1800s
LLMFLUX-STAGE-A: rank 0 up on <node>
LLMFLUX-STAGE-A: rank 1 up on <node>
LLMFLUX-STAGE-A: all 2 ranks launched
```

Two bounded waits guard the startup, and both fail loudly with diagnostics
rather than hanging to walltime:

1. **Rank start** (`LLMFLUX_RANK_START_TIMEOUT`, default 300 s) — every rank must
   reach its launcher. This separates "SLURM never placed the step" from "the
   model is still loading", which look identical from a readiness probe.
2. **Server readiness** (`LLMFLUX_SERVER_TIMEOUT`, default 1800 s on multi-node)
   — clamped to 90% of the allocation's remaining time, because a budget longer
   than the walltime is never reached: SLURM kills the job first and the failure
   becomes indistinguishable from a timeout.

### Requirements

- `$LLMFLUX_LOGS_DIR` must be on storage **every node can see**. The rendezvous
  files live under it. A node-local logs directory fails the rank-start barrier;
  the diagnostics print the detected filesystem type to make that visible.
- `ss` must be available on the compute nodes (used to find a free rendezvous
  port).

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `LLMFLUX_SERVER_TIMEOUT` | 300 single-node / 1800 multi-node | readiness budget in seconds; clamped to 90% of remaining walltime on multi-node |
| `LLMFLUX_RANK_START_TIMEOUT` | 300 | seconds to wait for all ranks to reach their launcher |
| `LLMFLUX_HSN_IFACE` | `hsn` | fabric interface name prefix. Matched exactly as `<prefix><digits>`, so VLAN children such as `hsn0.561` are excluded |
| `LLMFLUX_NCCL_SOCKET_IFNAME` | derived | override the NCCL interface list instead of deriving it from the node |
| `LLMFLUX_RDZV_PORT` | 29500 | starting port for the rendezvous |
| `LLMFLUX_PORT_SCAN_MAX` | 64 | how far above `LLMFLUX_RDZV_PORT` to scan for a free port |
| `LLMFLUX_SYMM_MEM` | 0 | sets `VLLM_ALLREDUCE_USE_SYMM_MEM`; vLLM defaults it on, and it deadlocks cross-node during engine init |

## Overriding the derived parallelism

`--vllm-engine-args` values you supply always win over anything derived from the
allocation — a value you typed is never silently overridden.

On a multi-node job, `tensor-parallel-size × pipeline-parallel-size` must equal
`nodes × gpus-per-node`. An override that does not is rejected at submit time:

```
$ llmflux run --nodes 2 --gpus-per-node 4 \
      --vllm-engine-args '{"pipeline-parallel-size": 1}' ...
Error: --vllm-engine-args sets tensor-parallel-size=4 and pipeline-parallel-size=1,
which uses 4 GPU(s), but the allocation is --nodes 2 x --gpus-per-node 4 = 8 GPU(s).
The remaining 4 GPU(s) would be allocated and billed for the job's full walltime
while sitting idle.
```

Nothing downstream would have caught it: vLLM's own check is only
`world_size % nnodes == 0`, which `4 % 2` passes.

An override that *does* account for every GPU is allowed —
`tensor-parallel-size 8 pipeline-parallel-size 1` across two 4-GPU nodes puts
tensor parallelism across the network, which is usually slower, but it is your
call.

At `--nodes 1` there is no such check: asking for fewer GPUs than you allocated
is a legitimate thing to do.

## Diagnosing a failure

Every launcher message is prefixed, so `grep LLMFLUX- <jobid>.out` is the first
move.

| Message | Meaning |
|---|---|
| `LLMFLUX-ERROR: allocation has N node(s), expected M` | `--sbatch-arg` appended a second `--nodes`; SLURM honours the last one |
| `LLMFLUX-ERROR: only K of N ranks started within Ts` | SLURM could not place a rank, or the logs directory is not shared. Read the `LLMFLUX-DIAG:` lines that follow |
| `LLMFLUX-ERROR: srun step exited before all ranks started` | a rank died immediately — usually a bind-mount or cache path that does not exist inside the container |
| `LLMFLUX-ERROR: no routable fabric IPv4 on the head node` | no `hsn<n>` interface; set `LLMFLUX_HSN_IFACE` |
| `LLMFLUX-WARN: budget Ts exceeds the allocation; clamping to Cs` | informational — the readiness budget was larger than the walltime |

To confirm the allocation is actually being used, check that the `.0` step spans
every node:

```bash
sacct -j <jobid> --format=JobID,JobName,NNodes,AllocTRES%40
# 12345.0  rank-launch.sh  2  cpu=32,gres/gpu=8,node=2
```

A job with no `.0` step, or a `.0` step on one node, is not running multi-node.

### Caches under `$HOME`

Apptainer binds the path it is given, not what that path points at. A cache
relocated **by symlink** — `~/.cache/huggingface` or `~/.triton` pointed at
another filesystem to escape a home quota — dangles inside the container and the
engine dies with a `FileNotFoundError` naming a directory that plainly exists
from outside. `HF_HOME` is resolved before use and the Triton cache is pinned
into an already-bound directory, but if you relocate a cache yourself, point the
environment variable at the real path rather than symlinking.

## What is not supported

- **Ollama multi-node** — rejected at submit time.
- **More than one node with `--engine ollama`**, including `--nodes 2
  --gpus-per-node 1`.
- Model architectures without pipeline-parallel support in vLLM will fail inside
  the engine; the topology gate does not know about them yet.
