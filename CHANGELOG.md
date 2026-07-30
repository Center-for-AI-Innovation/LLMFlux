# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  (see [#121](https://github.com/Center-for-AI-Innovation/LLMFlux/pull/121)).
- `--mem` CLI flag and programmatic memory settings now actually reach the
  generated `#SBATCH --mem` line. Previously `SlurmConfig` had two fields for
  the same setting (`mem` and `memory`); the CLI and examples set `mem`, but
  the job scripts were built from `memory`, so user-specified memory was
  silently ignored and jobs always got the `SLURM_MEM` env default (#119).
- HuggingFace token for gated models is now also read from `HF_TOKEN` when
  `HUGGINGFACE_TOKEN` is not set (#119).

### Removed

- **Breaking:** the duplicate `SlurmConfig.mem` field. Use
  `slurm_config.memory` instead. Note that `SlurmConfig(mem=...)` is silently
  ignored by pydantic, while attribute assignment (`slurm_config.mem = ...`)
  raises a `ValueError` (#119).

## [1.0.0] - 2026-07-06

- First stable release.