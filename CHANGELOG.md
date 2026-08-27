# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Breaking

- Batch jobs now exit with the batch processor's status. The generated script
  ran cleanup (`pkill`, `rm -rf`, `kill`) after the inline processor, each
  overwriting `$?`, so the job exited with whatever cleanup returned — always
  `0`. A run whose every item failed produced a complete-looking output file and
  a green SLURM job. Automation that treats a zero exit as "the batch succeeded"
  will now see failures it previously missed
  (see [#137](https://github.com/Center-for-AI-Innovation/LLMFlux/issues/137)).
- `llmflux serve` jobs likewise exit with the server's status rather than
  unconditionally `0`. `scancel` and Ctrl-C (130/143) are still reported as
  success, since that is how a serve job normally ends.
- `--nodes N` with `N > 1` is now validated at submit time. With the vLLM engine
  it is implemented (below); with the Ollama engine it is **rejected** with an
  actionable error instead of being silently accepted while `N-1` nodes sat idle
  and billed. A previously "working" Ollama multi-node command now fails
  immediately.
- `--gpus-per-node 0` is now rejected. It previously reached `sbatch`.
- `connection.json`'s `node` field is now a dialable address — the fabric IPv4
  at `nodes > 1`, the short hostname at `nodes == 1` — rather than `$(hostname)`,
  which returns an FQDN that can resolve to IPv6 link-local only. Anything
  parsing that field should expect either form.

### Added

- `LLMClient` and `BatchProcessor` accept an `api_key`, falling back to the
  `LLMFLUX_API_KEY` environment variable, and send it as a bearer token on
  every request. `llmflux serve` starts vLLM with `--api-key`, so until now
  LLMFlux's own client could not talk to an endpoint LLMFlux itself had
  started — every request came back HTTP 401, leaving the OpenAI SDK as the
  only usable client. A 401/403 with no key configured now logs where to get
  one instead of a bare request error. Blank and whitespace-only values are
  treated as unset, and surrounding whitespace is stripped, so a key quoted with
  a trailing space in `.env` no longer produces a `Bearer abc ` header that vLLM
  rejects. `.env.example` ships the entry commented out: `.env` is loaded with
  `override=True`, so a bare `LLMFLUX_API_KEY=` line would erase a key exported
  in the shell
  (see [#129](https://github.com/Center-for-AI-Innovation/LLMFlux/issues/129)).
- **Multi-node vLLM inference.** `--nodes N` now deploys the model across all
  `N` nodes instead of starting a server on the first and leaving the rest idle.
  Tensor parallelism stays within a node — it needs an all-reduce per layer —
  and pipeline parallelism spans nodes, exchanging only activations at stage
  boundaries, so `--nodes 2 --gpus-per-node 4` runs
  `tensor-parallel-size=4 pipeline-parallel-size=2`. Ranks meet through vLLM's
  own rendezvous flags under a single SPMD `srun` step, one task per node; rank 0
  serves the API and the rest run `--headless`. No Ray, and no container change.
  See `docs/MULTINODE.md`
  (see [#137](https://github.com/Center-for-AI-Innovation/LLMFlux/issues/137)).
- **Submit-time topology validation** (`llmflux.slurm.topology`). One place
  decides whether a requested node/GPU shape can be served and how it maps onto
  engine parallelism. Unservable shapes raise `TopologyError` before the job is
  queued — in milliseconds rather than after a queue wait — and the message names
  the flag to use instead and how many nodes would be billed while idle. That
  includes a `--vllm-engine-args` parallelism that does not account for every
  allocated GPU, which nothing downstream catches: vLLM asserts only
  `world_size % nnodes == 0`.
- New environment variables, all optional, documented in `.env.example` and
  `docs/MULTINODE.md`: `LLMFLUX_SERVER_TIMEOUT` (readiness budget, 300 s
  single-node / 1800 s multi-node, clamped to 90% of the allocation's remaining
  walltime), `LLMFLUX_RANK_START_TIMEOUT`, `LLMFLUX_HSN_IFACE`,
  `LLMFLUX_NCCL_SOCKET_IFNAME`, `LLMFLUX_RDZV_PORT`, `LLMFLUX_PORT_SCAN_MAX`,
  `LLMFLUX_SYMM_MEM`, and `LLMFLUX_CONNECT_TIMEOUT` / `LLMFLUX_READ_TIMEOUT` for
  the client request timeout.
- Test infrastructure for the generated SLURM scripts, which no test previously
  read end to end: golden characterization tests pinning every emitted script,
  heredoc-aware `bash -n` (bash treats heredoc bodies as data, so the per-rank
  launcher — the only code that runs on nodes 2..N — was invisible to a plain
  `bash -n`) plus a guard against `export VAR=$(cmd)`, which discards the
  substitution's exit status in every spelling including the quoted one, and a
  harness that *executes* the generated scripts under stubbed commands so exit
  status, elapsed time and termination can be asserted rather than inferred.

- Configurable workspace via `LLMFLUX_WORKSPACE` or `workspace="/path"` on
  `Config` / `ConfigManager.reset_config()`, resolved as code argument →
  environment variable → current working directory. Every workspace-derived
  path follows it: the `data`/`models`/`logs`/`containers` defaults, the
  `.ollama` / `.vllm` engine homes, and the Apptainer `tmp` / `tmp/cache`
  directories. Previously `Config.workspace` was hardcoded to the current
  working directory, and the `WORKSPACE` variable named in the docs was never
  read by any code. `workspace` is deliberately not accepted by
  `update_config()`, since changing it on a live config would leave already
  derived values stale — use `reset_config()`
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- Input and output directories are now configurable independently of
  `data_dir`, via `LLMFLUX_DATA_INPUT_DIR` / `LLMFLUX_DATA_OUTPUT_DIR` or the
  new `data_input_dir` / `data_output_dir` arguments on `Config`,
  `reset_config()` and `update_config()`, defaulting to `{data_dir}/input` and
  `{data_dir}/output`. This allows a read-only input location on one filesystem
  and results written to another
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).

### Changed

- `vision_to_jsonl()` no longer drops images without telling the caller, and
  its size limit is more realistic. `max_image_size` defaults to 25MB instead
  of 10MB: 12MP phone JPEGs are 2-6MB, but 48MP phones, DSLR JPEGs and
  full-screen PNG screenshots routinely pass 10MB, so legitimate inputs were
  being skipped. Oversized or failed images are now summarized in a single
  warning, and `return_report=True` returns a `(path, report)` tuple naming
  each one, so a short JSONL is explained rather than silent. Note the limit
  bounds the request path, not GPU cost — vision models tile images to a fixed
  resolution, so per-image GPU cost saturates; use `--limit-mm-per-prompt` to
  bound work per request
  (see [#134](https://github.com/Center-for-AI-Innovation/LLMFlux/issues/134)).
- The single-node readiness bound is now the `LLMFLUX_SERVER_TIMEOUT` variable
  rather than a literal `{1..300}`. The default is unchanged at 300 s.
- `APPTAINER_BIND_PATHS` is now `export`ed rather than assigned. It was a plain
  shell assignment, so an `srun`-launched child saw it empty and
  `apptainer exec --bind ""` exits 0 with no diagnostic.
- The vLLM and Ollama serve cleanup traps now kill the server by PID in addition
  to their existing `pkill`. `pkill -f "<engine> serve"` is not job-scoped, so it
  also reaches the same user's other jobs on a shared node; for vLLM at
  `nodes == 1` it matched nothing at all, since that path launches
  `python3 -m vllm.entrypoints.openai.api_server`.
- A serve job publishes its endpoint in three places — the connection file, the
  notification email, and the email's `OpenAI(base_url=...)` example. All three
  now read one resolved value instead of two different ones.
- `--nodes` and `--gpus-per-node` have help text.

- All directory environment variables are now `LLMFLUX_`-prefixed
  (`LLMFLUX_WORKSPACE`, `LLMFLUX_DATA_INPUT_DIR`, `LLMFLUX_DATA_OUTPUT_DIR`),
  replacing the unprefixed `WORKSPACE` / `DATA_INPUT_DIR` / `DATA_OUTPUT_DIR`
  names in `.env.example` and `docs/CONFIGURATION.md`
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- `SlurmRunner` resolves the input and output directories from the `Config`
  attributes it already uses for the SLURM job environment and the Apptainer
  bind mounts, instead of resolving them separately at use time
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).

### Fixed
- The generated batch stage now runs under the interpreter that generated it
  (`sys.executable`) instead of resolving a bare `python3` from whatever `PATH`
  the compute node inherits. The `llmflux` CLI has an absolute shebang and is
  therefore immune to `PATH` shadowing, so any other interpreter ahead on `PATH`
  — an activated venv, a second conda env, a notebook kernel — left the CLI
  working while the batch stage silently picked up a Python without `llmflux`.
  The job then failed with `ModuleNotFoundError: No module named 'llmflux'` only
  *after* the model had loaded and the server had passed its health check, so it
  read as a mid-run crash rather than an environment problem. Affects both the
  vLLM and Ollama batch paths
  (see [#142](https://github.com/Center-for-AI-Innovation/LLMFlux/issues/142)).
- Dropped `sys.path.append('$PROJECT_ROOT')` from the generated batch stage.
  `PROJECT_ROOT` is the user's working directory, and under the `src/` layout it
  can never contain the `llmflux` package, so the line was dead code that
  disguised the missing interpreter pin above.

- `vision_to_jsonl()` now finds images in a directory. Its default
  `file_pattern` was `"*.{jpg,jpeg,png,gif,webp,bmp}"`, but Python's `glob` has
  no brace expansion, so the pattern matched nothing and a directory of images
  produced an empty JSONL file with only an `INFO - Found 0 images` line.
  Directory discovery now defaults to every supported extension, matched
  case-insensitively so `IMG_1234.JPG` is included; an explicit `file_pattern`
  is still honored (brace groups are expanded, so callers that passed the old
  default keep working) and `"**/*.jpg"` still recurses. An empty result now
  logs a warning instead of an info line
  (see [#131](https://github.com/Center-for-AI-Innovation/LLMFlux/issues/131)).
- Ollama containers now receive SLURM's real `CUDA_VISIBLE_DEVICES`.
  `apptainer --cleanenv` strips it, so the container fell back to the list
  synthesised from the *requested* GPU count, which is correct only when the
  granted device indices happen to start at 0 and be contiguous. Ollama warns
  about it directly: `user overrode visible devices`. vLLM already did this; the
  two engine scripts had drifted.
- Every `LLMClient` request now has a timeout. `requests.Session` had none, so a
  read could block forever: if the engine died *after* the readiness check
  passed, the batch processor waited out the rest of the job's walltime, the
  allocation was billed in full, and nothing was reported anywhere.
- `HF_HOME` is resolved before being used as an Apptainer bind source. Apptainer
  binds the path it is given, not what that path points at, so a symlinked
  HuggingFace cache dangled inside the container and the engine died with
  `FileNotFoundError` naming a directory that plainly exists from outside.
- The Triton cache is pinned to `$XDG_CACHE_HOME/triton`, which is already
  created and bound, rather than defaulting to `~/.triton` — the same root cause
  as the `HF_HOME` fix.
- `llmflux connect` no longer prints a truncated SSH tunnel target. It derived
  the target by stripping everything after the first `.`, which turns a fabric
  IPv4 into its first octet.

- vLLM jobs no longer fail to start with `OSError: [Errno 30] Read-only file
  system: '{workspace}/.cache/flashinfer'`. `XDG_CACHE_HOME` and
  `FLASHINFER_WORKSPACE_BASE` were only exported with the `APPTAINERENV_`
  prefix, so the generated batch script expanded them to empty strings in its
  `mkdir -p` and `--bind` lines: the workspace cache directory was never
  created on the host and never bound into the container, leaving it read-only
  where FlashInfer writes its JIT kernel cache. Both are now host variables as
  well
  (see [#125](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/125)).
- `update_config()` no longer discards the directory overrides passed to it.
  After applying its arguments it rebuilt `DATA_INPUT_DIR`, `DATA_OUTPUT_DIR`,
  `MODELS_DIR`, `LOGS_DIR` and `CONTAINERS_DIR` from hardcoded
  `config.workspace / ...` paths, so a `data_dir` or `models_dir` passed in the
  same call was silently ignored by everything reading the derived paths. They
  are now derived from the configured directory attributes
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- `SlurmRunner.run()` resolved the input-file fallback and default output path
  via `Config.get_path()`, while the Apptainer bind mounts used
  `self.data_input_dir` / `self.data_output_dir` — two different reads of the
  same config, so a `data_input_dir` / `data_output_dir` override could reach
  one and not the other. Both now use `self.data_input_dir` /
  `self.data_output_dir`
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- `--mem` CLI flag and programmatic memory settings now actually reach the
  generated `#SBATCH --mem` line. Previously `SlurmConfig` had two fields for
  the same setting (`mem` and `memory`); the CLI and examples set `mem`, but
  the job scripts were built from `memory`, so user-specified memory was
  silently ignored and jobs always got the `SLURM_MEM` env default (see [#119](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/119)).
- HuggingFace token for gated models is now also read from `HF_TOKEN` when
  `HUGGINGFACE_TOKEN` is not set (see [#119](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/119)).

### Removed

- **Breaking:** `Config.get_environment()`, which built a full environment
  dictionary by iterating `default_paths` and `default_settings`. It had no
  callers; `SlurmRunner` builds the job environment itself
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- **Breaking:** `Config.get_path()`, `Config.get_setting()`, `Config.default_paths`
  and `Config.default_settings`, the layer `get_environment()` was built on.
  Directories are read directly from the `Config` attributes
  (`data_input_dir`, `data_output_dir`, `models_dir`, `logs_dir`,
  `containers_dir`); SLURM settings from `get_slurm_config()`.
  As a consequence the unprefixed `DATA_INPUT_DIR` / `DATA_OUTPUT_DIR`
  environment variables are no longer read — use `LLMFLUX_DATA_INPUT_DIR` and
  `LLMFLUX_DATA_OUTPUT_DIR` (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- **Breaking:** the duplicate `SlurmConfig.mem` field. Use
  `slurm_config.memory` instead. Note that `SlurmConfig(mem=...)` is silently
  ignored by pydantic, while attribute assignment (`slurm_config.mem = ...`)
  raises a `ValueError` (see [#119](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/119)).

## [1.0.0] - 2026-07-06

- First stable release.