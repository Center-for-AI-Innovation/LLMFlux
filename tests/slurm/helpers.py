"""Shared builders for the SLURM script tests.

Not named test_* so pytest does not collect it.

Three test modules need the same thing: "give me the script the builders emit
for this engine and mode". Each had grown its own copy of the fixed-argument
dict, which meant four places to update when a builder's signature changed, and
subtly different arguments between modules — so a golden and an execution test
could disagree about what they were testing.
"""

from pathlib import Path
from unittest.mock import MagicMock

#: (engine, mode) pairs every script-level test sweeps.
CASES = [
    ("vllm", "batch"),
    ("vllm", "serve"),
    ("ollama", "batch"),
    ("ollama", "serve"),
]

#: Fixed builder arguments. Arbitrary but stable — changing a value churns every
#: golden file, so change it only deliberately.
DEFAULTS = dict(
    account="myaccount",
    partition="gpu",
    nodes="1",
    gpus_per_node="2",
    time="01:00:00",
    memory="64G",
    cpus_per_task="8",
    logs_dir=Path("/logs"),
    input_file=Path("/data/in.jsonl"),
    output_file=Path("/data/out.json"),
)

SERVE_EMAIL = "someone@example.edu"


def slurm_config(extra_sbatch_args=None):
    """A stand-in for SlurmConfig; the builders only read extra_sbatch_args."""
    cfg = MagicMock()
    cfg.extra_sbatch_args = extra_sbatch_args
    return cfg


def build_lines(engine, mode="batch", job_name=None, **overrides):
    """Return the generated script as the builder produces it: a list of lines."""
    if engine == "vllm":
        from llmflux.slurm.engine.vllm import create_vllm_batch_script as maker
    elif engine == "ollama":
        from llmflux.slurm.engine.ollama import create_ollama_batch_script as maker
    else:
        raise ValueError(f"unknown engine {engine!r}")

    kwargs = dict(
        DEFAULTS,
        job_name=job_name or f"test-{engine}-{mode}",
        slurm_config=overrides.pop("slurm_config", None) or slurm_config(),
        mode=mode,
    )
    if mode == "serve":
        kwargs["email"] = SERVE_EMAIL
    kwargs.update(overrides)
    return maker(**kwargs)


def build_text(engine, mode="batch", **overrides):
    """Return the generated script as text, with a trailing newline."""
    return "\n".join(build_lines(engine, mode, **overrides)) + "\n"
