#!/usr/bin/env python3
"""What `llmflux clean` and `llmflux remove` delete.

Both work from a fixed list of paths rather than wiping the workspace: on a
source checkout the workspace also holds the repo (src/, docs/, tests/,
pyproject.toml), and data/input and data/output hold the user's own files.
Nothing outside these lists is touched.
"""

import shutil
from pathlib import Path


def clean_paths(config) -> list:
    """Regenerable scratch: Slurm logs, Apptainer build space, leftover job files."""
    workspace = Path(config.workspace)
    return [
        Path(config.logs_dir),
        workspace / "tmp",
        workspace / "staged-input",
        workspace / "job.sh",
        Path.home() / ".llmflux" / "serve",
    ]


def remove_paths(config) -> list:
    """Everything `clean` deletes, plus model weights, engine caches and job history.

    `HF_HOME` is not followed: only the workspace's own `.cache` is cleared, so
    a cache relocated onto shared storage stays put.
    """
    workspace = Path(config.workspace)
    return clean_paths(config) + [
        Path(config.containers_dir),
        Path(config.models_dir),
        workspace / ".cache",
        workspace / ".ollama",
        workspace / ".vllm",
        Path.home() / ".llmflux",
    ]


def delete(paths) -> tuple:
    """Empty each directory in `paths` and unlink each file, keeping the directories.

    Returns `(deleted, errors)`. A path that is already gone is skipped, and a
    path that cannot be deleted is recorded in `errors` so one failure does not
    strand the rest.
    """
    deleted = []
    errors = []
    for path in paths:
        if not path.exists():
            continue
        try:
            # is_dir() follows symlinks, so a relocated cache — .cache pointing
            # at scratch, a common answer to a home quota — would have its
            # target emptied instead of the link removed. Unlink the link.
            if path.is_dir() and not path.is_symlink():
                for child in path.iterdir():
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            else:
                path.unlink()
            deleted.append(path)
        except OSError as exc:
            errors.append(f"{path}: {exc}")
    return deleted, errors
