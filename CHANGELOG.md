# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

- vLLM jobs no longer fail to start with `OSError: [Errno 30] Read-only file
  system: '{workspace}/.cache/flashinfer'`. `XDG_CACHE_HOME` and
  `FLASHINFER_WORKSPACE_BASE` were only exported with the `APPTAINERENV_`
  prefix, so the generated batch script expanded them to empty strings in its
  `mkdir -p` and `--bind` lines: the workspace cache directory was never
  created on the host and never bound into the container, leaving it read-only
  where FlashInfer writes its JIT kernel cache. Both are now host variables as
  well.
- `update_config()` no longer discards the directory overrides passed to it.
  After applying its arguments it rebuilt `DATA_INPUT_DIR`, `DATA_OUTPUT_DIR`,
  `MODELS_DIR`, `LOGS_DIR` and `CONTAINERS_DIR` from hardcoded
  `config.workspace / ...` paths, so a `data_dir` or `models_dir` passed in the
  same call was silently ignored by everything reading the derived paths. They
  are now derived from the configured directory attributes
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- Directory overrides now reach the Apptainer bind mounts. Path resolution and
  the bind mounts were computed from two different sources, so an override
  could change where the runner looked for input while the container was still
  bound to the default location
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