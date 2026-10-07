#!/usr/bin/env python3
"""Paths deleted by `llmflux clean` and `llmflux remove`.

Read from env vars, not Config(), which validates dirs these commands never touch.
"""

import os
import shutil
from pathlib import Path


def _workspace() -> Path:
    return Path(os.getenv("LLMFLUX_WORKSPACE") or Path.cwd()).expanduser()


def _dir(env_var: str, default: Path) -> Path:
    return Path(os.getenv(env_var) or default).expanduser()


def clean_paths() -> list:
    """Logs, build scratch and leftover job files."""
    workspace = _workspace()
    return [
        _dir("LLMFLUX_LOGS_DIR", workspace / "logs"),
        workspace / "tmp",
        workspace / "staged-input",
        workspace / "job.sh",
        Path.home() / ".llmflux" / "serve",
    ]


def remove_paths() -> list:
    """Everything `clean` deletes, plus models, caches and job history."""
    workspace = _workspace()
    return clean_paths() + [
        _dir("LLMFLUX_CONTAINERS_DIR", workspace / "containers"),
        _dir("LLMFLUX_MODELS_DIR", workspace / "models"),
        workspace / ".cache",
        workspace / ".ollama",
        workspace / ".vllm",
        Path.home() / ".llmflux",
    ]


def delete(paths) -> tuple:
    """Empty each directory and unlink each file. Returns (deleted, errors)."""
    deleted = []
    errors = []
    for path in paths:
        if not path.exists():
            continue
        # Unlink symlinks; don't empty their targets.
        try:
            if path.is_dir() and not path.is_symlink():
                targets = list(path.iterdir())
            else:
                targets = [path]
        except OSError as exc:
            errors.append(f"{path}: {exc}")
            continue
        ok = True
        for target in targets:
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            except OSError as exc:
                errors.append(f"{target}: {exc}")
                ok = False
        if ok:
            deleted.append(path)
    return deleted, errors
