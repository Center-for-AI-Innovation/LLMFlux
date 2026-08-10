# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Fixed

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