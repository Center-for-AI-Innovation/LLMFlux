# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added


### Changed


### Fixed

- `--mem` CLI flag and programmatic memory settings now actually reach the
  generated `#SBATCH --mem` line. Previously `SlurmConfig` had two fields for
  the same setting (`mem` and `memory`); the CLI and examples set `mem`, but
  the job scripts were built from `memory`, so user-specified memory was
  silently ignored and jobs always got the `SLURM_MEM` env default (#119).
- HuggingFace token for gated models is now also read from `HF_TOKEN` when
  `HUGGINGFACE_TOKEN` is not set (#119).

### Removed

- **Breaking:** `Config.get_path()`, `Config.get_setting()`, `Config.default_paths`
  and `Config.default_settings`. Directories are read directly from the
  `Config` attributes (`data_input_dir`, `data_output_dir`, `models_dir`,
  `logs_dir`, `containers_dir`); SLURM settings from `get_slurm_config()`.
  As a consequence the unprefixed `DATA_INPUT_DIR` / `DATA_OUTPUT_DIR`
  environment variables are no longer read — use `LLMFLUX_DATA_INPUT_DIR` and
  `LLMFLUX_DATA_OUTPUT_DIR`.
- **Breaking:** the duplicate `SlurmConfig.mem` field. Use
  `slurm_config.memory` instead. Note that `SlurmConfig(mem=...)` is silently
  ignored by pydantic, while attribute assignment (`slurm_config.mem = ...`)
  raises a `ValueError` (#119).

## [1.0.0] - 2026-07-06

- First stable release.