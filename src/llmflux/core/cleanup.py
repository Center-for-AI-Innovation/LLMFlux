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


def home_workspace_error(paths) -> str:
    """Why `clean`/`remove` must not run, or "" if the workspace is not home.

    The workspace defaults to the current directory, so running from ~ makes
    home the workspace, and `paths` then include ~/tmp, ~/.cache, ~/models and
    the like. Those are emptied in full, and in a home directory they hold far
    more than LLMFlux's files: pip's and other tools' caches, people's own
    scratch files and model weights (LLMFlux#155). An explicit
    LLMFLUX_WORKSPACE=~ is refused too: the risk is the same, and a dedicated
    workspace costs one env var.
    """
    home = Path.home().resolve()
    if _workspace().resolve() != home:
        return ""
    targets = []
    for path in paths:
        location = path.parent.resolve() / path.name  # as in _skip_reason: don't follow a final symlink
        if location.is_relative_to(home) and not location.is_relative_to(home / ".llmflux"):
            targets.append(f"~/{location.relative_to(home)}")
    return (
        f"the workspace is your home directory, so this would empty {', '.join(targets)} "
        "in full, including files that aren't LLMFlux's. Set LLMFLUX_WORKSPACE to your "
        "LLMFlux workspace, or run from inside it."
    )


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


def _skip_reason(path: Path) -> str:
    """Why `path` must not be touched, or "" if it is managed by these commands."""
    if path.name in ("", ".."):
        location = path.resolve()
    else:
        location = path.parent.resolve() / path.name  # don't follow a symlink at `path`
    workspace = _workspace().resolve()
    home = Path.home().resolve()
    if location in (workspace, home, Path(location.anchor)):
        return "is the workspace, home, or root"
    allowed = (workspace, home / ".llmflux", (home / ".llmflux").resolve())
    if any(location.is_relative_to(root) for root in allowed):
        return ""
    return "outside the workspace, not managed by this command"


def delete(paths) -> tuple:
    """Empty each managed directory and unlink each file. Returns (deleted, skipped, errors)."""
    deleted = []
    skipped = []
    errors = []
    for path in paths:
        if not path.exists() and not path.is_symlink():
            continue
        reason = _skip_reason(path)
        if reason:
            skipped.append(f"Skipping {path}: {reason}")
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
    return deleted, skipped, errors
