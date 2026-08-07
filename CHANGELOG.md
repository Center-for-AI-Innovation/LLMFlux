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
  one instead of a bare request error
  (see [#129](https://github.com/Center-for-AI-Innovation/LLMFlux/issues/129)).

### Fixed

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