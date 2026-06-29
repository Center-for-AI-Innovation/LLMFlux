#!/usr/bin/env python3
"""Command-line interface for LLMFlux.

Provides the `llmflux` executable with subcommands.
"""

import argparse
import json
import sys
from pathlib import Path
import time
import subprocess
import logging
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Optional, List, Dict
from importlib.metadata import PackageNotFoundError, version as pkg_version

from .slurm.runner import SlurmRunner
from .slurm.connection import connect, read_connection_info
from .slurm.commands import (
    ACTIVE_STATES,
    SlurmCommandError,
    cancel_job,
    get_active_job_details,
    get_job_details,
    get_list_of_jobs_details,
    get_job_log_paths,
    get_job_state,
    extract_state,
)
from .processors import BatchProcessor
from .core.config import Config, EngineConfig
from .core.registry import JobRegistry
from .benchmark_utils import create_test_prompts_file


def _get_llmflux_version() -> str:
    """Return installed llmflux version (best effort)."""
    try:
        return pkg_version("llmflux")
    except PackageNotFoundError:
        return "unknown"

def _parse_sbatch_args(sbatch_arg_list: Optional[List[str]]) -> Optional[Dict[str, str]]:
    """Parse --sbatch-arg arguments into a dictionary.

    Args:
        sbatch_arg_list: List of "key=value" strings from CLI

    Returns:
        Dictionary of extra SBATCH arguments or None
    """
    if not sbatch_arg_list:
        return None

    result = {}
    for arg in sbatch_arg_list:
        if '=' not in arg:
            print(f"Warning: Ignoring invalid --sbatch-arg format: {arg} (expected KEY=VALUE)")
            continue
        key, value = arg.split('=', 1)
        result[key.strip()] = value.strip()

    return result if result else None

def _parse_elapsed_to_seconds(elapsed: str) -> float:
    """Parse a sacct elapsed string (HH:MM:SS or D-HH:MM:SS) to total seconds."""
    try:
        day_seconds = 0
        rest = elapsed
        if "-" in elapsed:
            days, rest = elapsed.split("-", 1)
            day_seconds = int(days) * 86400
        h, m, s = rest.split(":")
        return day_seconds + int(h) * 3600 + int(m) * 60 + int(s)
    except Exception:
        return 0.0


def _wait_for_slurm_elapsed_seconds(job_id: str, poll_seconds: int = 30, timeout_seconds: int = 6 * 3600) -> str | None:
    """Return elapsed runtime (seconds) once SLURM job completes; None when unavailable."""
    start = time.monotonic()
    while time.monotonic() - start < timeout_seconds:
        result = subprocess.run(
            [
                "sacct",
                "-n",
                "-P",
                "-j",
                f"{job_id}.batch",
                "--format=State,Elapsed"
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            rows = [row for row in result.stdout.splitlines() if row.strip()]
            if rows:
                state, elapsed = rows[0].split("|", 1)
                state = state.upper()
                if state in {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}:
                    return str(elapsed)
        print(f"Job {job_id} is still running... Waiting for {poll_seconds} seconds")
        time.sleep(poll_seconds)
    return None

def _ensure_container(config: "Config") -> bool:
    """Build the Apptainer image on the current node if it does not exist.

    Returns True if the image is ready, False on failure.
    """
    import os
    sif_path = Path(config.containers_dir) / "llm_processor.sif"
    if sif_path.exists():
        return True

    container_def = Path(__file__).parent / "container" / "container.def"
    if not container_def.exists():
        print(f"Error: container.def not found at {container_def}", file=sys.stderr)
        return False

    containers_dir = Path(config.containers_dir)
    apptainer_tmp = Path(config.workspace) / "tmp"
    apptainer_cache = apptainer_tmp / "cache"
    apptainer_tmp.mkdir(parents=True, exist_ok=True)
    apptainer_cache.mkdir(parents=True, exist_ok=True)
    containers_dir.mkdir(parents=True, exist_ok=True)

    print("Container image not found — building now (this may take a few minutes)...")
    print(f"  Definition: {container_def}")
    print(f"  Output:     {sif_path}")

    result = subprocess.run(
        ["apptainer", "build", "--force", str(sif_path), str(container_def)],
        env={**os.environ, "APPTAINER_TMPDIR": str(apptainer_tmp), "APPTAINER_CACHEDIR": str(apptainer_cache)},
    )
    if result.returncode != 0:
        print(f"Error: apptainer build failed (exit {result.returncode}).", file=sys.stderr)
        return False

    print(f"Container built successfully: {sif_path}")
    return True


def _benchmark_command(args: argparse.Namespace) -> int:
    """Handle the `benchmark` subcommand.
    Args:
        args: Parsed CLI arguments
    Returns:
        Process exit code
    """
    # Generate or use provided dataset
    name = args.name or f"benchmark_{args.model}"
    num_prompts = getattr(args, "num_prompts", 50)

    if args.input:
        input_path = Path(args.input)
        # Read number of prompts from the input file
        num_prompts = sum(1 for _ in open(input_path))
        print(f"Number of prompts in the input file: {num_prompts}")
    else:
        # Generate synthetic prompts
        output_dir = Path("data/benchmarks")
        output_dir.mkdir(parents=True, exist_ok=True)
        temperature = 0.7 if args.temperature is None else args.temperature
        max_tokens = 500 if args.max_tokens is None else args.max_tokens

        input_path = create_test_prompts_file(num_prompts=num_prompts, temperature=temperature, max_tokens=max_tokens, model=args.model)
        
    # Set output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"results/benchmarks/{name}_results.json"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Collect SLURM config from CLI args (filter out None values)
    config = Config()
    if not getattr(args, "rebuild", False) and not _ensure_container(config):
        return 1
    slurm_overrides = {
        key: value for key, value in {
            "account": args.account,
            "partition": args.partition,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "time": args.time,
            "mem": args.mem,
            "cpus_per_task": args.cpus_per_task,
            "engine": args.engine,
        }.items() if value is not None
    }
    # Parse and add extra SBATCH args if provided
    extra_args = _parse_sbatch_args(getattr(args, 'sbatch_arg', None))
    if extra_args:
        slurm_overrides['extra_sbatch_args'] = extra_args
    # Merge CLI overrides with config from .env
    slurm_config = config.get_slurm_config(slurm_overrides)
    if args.engine == "ollama":
        engine_config = EngineConfig(
            engine="ollama",
            home=str(config.workspace / ".ollama")
        )
    else:
        engine_config = EngineConfig(
            engine="vllm",
            home=str(config.workspace / ".vllm")
        )
    runner = SlurmRunner(config=slurm_config, engine_config=engine_config)

    kwargs = {
        "model": args.model,
        "batch_size": getattr(args, "batch_size", 4),
    }

    if getattr(args, "rebuild", False):
        kwargs["rebuild"] = True
    if getattr(args, "debug", False):
        kwargs["debug"] = True
    if getattr(args, "vllm_engine_args", None) is not None:
        kwargs["vllm_engine_args"] = args.vllm_engine_args
    if getattr(args, "custom_config_path", None):
        kwargs["custom_config_path"] = args.custom_config_path

    print(f"Submitting benchmark job...")
    print(f"  Model: {args.model}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")

    job_id = runner.run(input_path=str(input_path), output_path=output_path, **kwargs)
    print(f"Job ID: {job_id}")

    summary = {
        "status": "submitted",
        "job_id": job_id,
        "benchmark_name": name,
        "model": args.model,
        "num_prompts": num_prompts,
        "input_path": str(input_path),
        "output_path": output_path,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = Path(f"results/benchmarks/{name}_submission.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))

    print("Benchmark job submitted (detached mode).")
    print(f"Submission details saved to {summary_path}")
    print(f"Benchmark output will be written to {output_path} once the job completes.")

    return 0


def _run_command(args: argparse.Namespace) -> int:
    """Handle the `run` subcommand.

    Args:
        args: Parsed CLI arguments

    Returns:
        Process exit code
    """
    input_path = args.input
    output_path = args.output
    model = args.model

    # Basic validation
    if not input_path:
        print("--input is required", file=sys.stderr)
        return 2

    # Ensure output directory exists if provided
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Local mode: run BatchProcessor directly (no SLURM)
    if getattr(args, "local", False):
        config = Config()
        model_type = model.split(":")[0] if ":" in model else model
        model_size = model.split(":")[1] if ":" in model else "3b"
        model_config = config.load_model_config(model_type, model_size)

        processor = BatchProcessor(
            model_config=model_config,
            batch_size=args.batch_size,
            save_frequency=args.save_frequency,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
        )

        # Additional kwargs that map to processor.run or underlying client params
        run_kwargs = {}
        if args.max_tokens is not None:
            run_kwargs["max_tokens"] = args.max_tokens
        if args.temperature is not None:
            run_kwargs["temperature"] = args.temperature
        if args.top_p is not None:
            run_kwargs["top_p"] = args.top_p
        if args.top_k is not None:
            run_kwargs["top_k"] = args.top_k

        processor.run(input_path=input_path, output_path=output_path or str(Path("results") / "output.json"), engine=args.engine, **run_kwargs)
        return 0

    # Initialize config - engine will be automatically detected from SLURM_ENGINE env var
    config = Config()
    if not getattr(args, "rebuild", False) and not _ensure_container(config):
        return 1

    # Override engine if provided via CLI
    if args.engine:
        engine_value = args.engine
        if engine_value == "ollama":
            config.engine = EngineConfig(
                engine="ollama",
                home=str(config.workspace / ".ollama")
            )
        else:
            config.engine = EngineConfig(
                engine="vllm",
                home=str(config.workspace / ".vllm")
            )

    # Collect Slurm config from args (excluding engine)
    logging.info(f"Engine set as = {config.engine}")
    # Collect Slurm config from args
    slurm_config = {
        key: value for key, value in {
            "account": args.account,
            "partition": args.partition,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "time": args.time,
            "mem": args.mem,
            "cpus_per_task": args.cpus_per_task,
        }.items() if value is not None
    }
    # Parse and add extra SBATCH args if provided
    extra_args = _parse_sbatch_args(getattr(args, 'sbatch_arg', None))
    if extra_args:
        slurm_config['extra_sbatch_args'] = extra_args
    # Update Slurm config with args
    slurm_config = config.get_slurm_config(slurm_config)
    # SLURM mode
    runner = SlurmRunner(config=slurm_config, engine_config=config.engine)
    # Collect kwargs accepted by SlurmRunner.run to set env for the job script
    kwargs = {
        "model": model,
        "batch_size": args.batch_size,
        "save_frequency": args.save_frequency,
        "max_retries": args.max_retries,
        "retry_delay": args.retry_delay,
    }
    if args.max_tokens is not None:
        kwargs["max_tokens"] = args.max_tokens
    if args.temperature is not None:
        kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    if args.top_k is not None:
        kwargs["top_k"] = args.top_k

    # Pass rebuild flag through to runner
    if getattr(args, "rebuild", False):
        kwargs["rebuild"] = True

    # Pass debug flag through to runner
    if getattr(args, "debug", False):
        kwargs["debug"] = True

    if getattr(args, "vllm_engine_args", None) is not None:
        kwargs["vllm_engine_args"] = args.vllm_engine_args
    if getattr(args, "custom_config_path", None):
        kwargs["custom_config_path"] = args.custom_config_path

    job_id = runner.run(input_path=input_path, output_path=output_path, **kwargs)
    print(f"Job ID: {job_id}")
    return 0


def _connect_command(args: argparse.Namespace) -> int:
    """Handle the `connect` subcommand."""
    job_id = str(args.job_id)
    if not job_id.isdigit():
        print(f"Error: Invalid job ID {job_id!r}: SLURM job IDs must be positive integers.", file=sys.stderr)
        return 1
    registry = JobRegistry()
    metadata = registry.get_job(job_id)

    if metadata is None:
        print(f"Job {job_id} is not tracked by LLMFlux registry.", file=sys.stderr)
        return 1

    if metadata.get("type") != "serve":
        print(f"Job {job_id} is a batch job, not a serve job.", file=sys.stderr)
        return 1

    try:
        state = get_job_state(job_id)
    except SlurmCommandError:
        state = None

    if state in {"FAILED", "CANCELLED", "TIMEOUT", "COMPLETED"}:
        print(f"Job {job_id} is no longer running (state: {state}).", file=sys.stderr)
        return 1

    if state == "PENDING":
        print(
            f"Job {job_id} is still queued (PENDING). "
            "Wait for the email notification before connecting.",
            file=sys.stderr,
        )
        return 1

    return connect(
        job_id=job_id,
        local_port=args.local_port,
        wait_timeout=args.wait_timeout,
    )


def _serve_command(args: argparse.Namespace) -> int:
    """Handle the `serve` subcommand."""
    config = Config()

    if args.engine == "ollama":
        engine_config = EngineConfig(engine="ollama", home=str(config.workspace / ".ollama"))
    else:
        engine_config = EngineConfig(engine="vllm", home=str(config.workspace / ".vllm"))

    slurm_overrides = {
        key: value for key, value in {
            "account": args.account,
            "partition": args.partition,
            "nodes": args.nodes,
            "gpus_per_node": args.gpus_per_node,
            "time": args.time,
            "mem": args.mem,
            "cpus_per_task": args.cpus_per_task,
        }.items() if value is not None
    }
    extra_args = _parse_sbatch_args(getattr(args, 'sbatch_arg', None))
    if extra_args:
        slurm_overrides['extra_sbatch_args'] = extra_args

    slurm_config = config.get_slurm_config(slurm_overrides)
    runner = SlurmRunner(config=slurm_config, engine_config=engine_config)

    kwargs = {"model": args.model}
    if getattr(args, "rebuild", False):
        kwargs["rebuild"] = True
    if getattr(args, "debug", False):
        kwargs["debug"] = True
    if getattr(args, "vllm_engine_args", None) is not None:
        kwargs["vllm_engine_args"] = args.vllm_engine_args

    try:
        job_id = runner.serve(email=args.email, **kwargs)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except subprocess.CalledProcessError as exc:
        print(f"Error: sbatch failed: {exc.stderr.strip()}", file=sys.stderr)
        return 1

    if not job_id:
        print("Error: serve job submission failed — no job ID returned.", file=sys.stderr)
        return 1

    print(f"Serve job submitted: {job_id}")
    print(f"  Model:  {args.model}")
    print(f"  Engine: {args.engine}")
    print(f"  Time:   {args.time}")
    print(f"  Email:  {args.email}")
    print(f"\nYou will receive an email at {args.email} when the service is ready.")
    print(f"Then run: llmflux connect {job_id}")
    return 0


def _show_models_command(args: argparse.Namespace) -> int:
    """Handle the `show-models` subcommand.

    Lists all model keys defined in models.yaml.
    """
    import yaml

    templates_dir = Path(__file__).parent / "templates"
    models_yaml = templates_dir / "models.yaml"

    if not models_yaml.exists():
        print(f"Error: models.yaml not found at {models_yaml}", file=sys.stderr)
        return 1

    with open(models_yaml) as f:
        data = yaml.safe_load(f)

    models = data.get("models", {})
    if not models:
        print("No models found in models.yaml.")
        return 0

    print(f"Available models ({len(models)} total):\n")
    for key in models:
        entry = models[key]
        name = entry.get("name", "")
        hf_name = entry.get("hf_name", "")
        engines = []
        if name and name != "NA":
            engines.append("ollama")
        if hf_name and hf_name != "NA":
            engines.append("vllm")
        engine_str = f"  [{', '.join(engines)}]" if engines else "  [no engine]"
        print(f"  {key}{engine_str}")

    return 0


def _render_table(rows: list[list[str]], headers: list[str]) -> None:
    """Helper function to render a table of rows and headers"""
    if not rows:
        print("No jobs found.")
        return
    widths = [len(header) for header in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    print(header_line)
    for row in rows:
        print("  ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))


def _pick(job: Dict[str, object], *keys: str) -> str:
    """
    Helper function to pick a value from a dictionary of keys
    If not found, returns "--"
    """
    for key in keys:
        value = job.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return "--"


def _format_timestamp(value: object) -> str:
    """Format any Slurm timestamp as local time in YYYY-MM-DD HH:MM:SS."""
    if value is None:
        return "--"
    if isinstance(value, (int, float)) and value > 0:
        return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")
    text = str(value).strip()
    if not text:
        return "--"
    try:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo:
            parsed = parsed.astimezone(tz=None)
        return parsed.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, OSError):
        return "--"


def _format_seconds_to_hhmmss(seconds: object) -> str:
    """Format an integer number of seconds as HH:MM:SS."""
    if not isinstance(seconds, (int, float)) or seconds <= 0:
        return "--"
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _format_time_limit(limit: object) -> str:
    """Unpack Slurm's set-object or plain integer time limit (minutes) into a string."""
    if isinstance(limit, dict):
        if not limit.get("set") or limit.get("infinite"):
            return "--"
        limit = limit.get("number", 0)
    if isinstance(limit, (int, float)) and limit > 0:
        minutes = int(limit)
        return f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    return "--"


def _pick_time_fields(details: Dict[str, object]) -> Dict[str, str]:
    """Extract start/elapsed/limit from sacct's nested 'time' block"""
    time_block = details.get("time") if isinstance(details.get("time"), dict) else {}
    return {
        "start": _format_timestamp(time_block.get("start")),
        "elapsed": _format_seconds_to_hhmmss(time_block.get("elapsed")),
        "limit": _format_time_limit(time_block.get("limit")),
    }


def _jobs_command(args: argparse.Namespace) -> int:
    """
    Handle the `jobs` command.
    Lists all jobs in a tabular format tracked by LLMFlux in the local registry.
    Defaults to showing only active jobs.
    """
    registry = JobRegistry()
    tracked_jobs = registry.get_all_jobs()
    if not tracked_jobs:
        print("No LLMFlux jobs found in local registry.")
        return 0

    state_filters = {state.upper() for state in (args.state or [])}
    rows: list[list[str]] = []

    if args.all:
        # Fetch all tracked jobs in a single sacct call, then iterate the registry.
        jobs = get_list_of_jobs_details(list(tracked_jobs.keys()))
        for job_id, metadata in tracked_jobs.items():
            slurm_data = jobs.get(str(job_id), {})
            state = extract_state(slurm_data) if slurm_data else "UNKNOWN"
            if state_filters and state not in state_filters:
                continue
            times = _pick_time_fields(slurm_data)
            rows.append([
                str(job_id),
                str(metadata.get("job_name", "--")),
                state,
                metadata.get("type", "batch"),
                str(metadata.get("model", "--")),
                str(metadata.get("engine", "--")),
                times["elapsed"],
                _format_timestamp(metadata.get("submitted_at")),
            ])
    else:
        # Default: one squeue call returns only active jobs; intersect with the registry.
        jobs = get_active_job_details()
        effective_filters = state_filters or set(ACTIVE_STATES)
        for job_id, slurm_data in jobs.items():
            if job_id not in tracked_jobs:
                continue
            metadata = tracked_jobs[job_id]
            state = extract_state(slurm_data)
            if effective_filters and state not in effective_filters:
                continue
            times = _pick_time_fields(jobs[job_id])
            rows.append([
                str(job_id),
                str(metadata.get("job_name", "--")),
                state,
                metadata.get("type", "batch"),
                str(metadata.get("model", "--")),
                str(metadata.get("engine", "--")),
                times["elapsed"],
                _format_timestamp(metadata.get("submitted_at")),
            ])

    rows.sort(key=lambda row: row[7], reverse=True)
    _render_table(
        rows,
        ["JOB ID", "NAME", "STATE", "TYPE", "MODEL", "ENGINE", "ELAPSED", "SUBMITTED"],
    )
    return 0


def _status_command(args: argparse.Namespace) -> int:

    """Handle the `status` command.
    Shows detailed status for a specific job.
    """
    job_id = str(args.job_id)
    registry = JobRegistry()
    metadata = registry.get_job(job_id)

    try:
        details = get_job_details(job_id)
    except SlurmCommandError as exc:
        print(f"Error querying Slurm for job {job_id}: {exc}", file=sys.stderr)
        return 1

    if not details and metadata is None:
        print(f"Job {job_id} was not found in Slurm or LLMFlux registry.", file=sys.stderr)
        return 1

    state = extract_state(details)
    stdout_path, stderr_path = get_job_log_paths(
        job_id=job_id,
        logs_dir=str(metadata.get("logs_dir")) if metadata else None,
    )
    times = _pick_time_fields(details)
    is_serve = metadata is not None and metadata.get("type") == "serve"

    view = OrderedDict(
        [
            ("Job ID", job_id),
            ("Job Name", metadata.get("job_name", _pick(details, "name", "job_name")) if metadata else _pick(details, "name", "job_name")),
            ("Type", metadata.get("type", "batch") if metadata else "batch"),
            ("State", state),
            ("Model", metadata.get("model", "--") if metadata else "--"),
            ("Engine", metadata.get("engine", "--") if metadata else "--"),
            ("Partition", _pick(details, "partition", "Partition")),
            ("Nodes", _pick(details, "nodes", "Nodes")),
            ("Submit Time", _format_timestamp(metadata.get("submitted_at") if metadata else None)),
            ("Start Time", times["start"]),
            ("Elapsed", times["elapsed"]),
            ("Time Limit", times["limit"]),
            ("Working Dir", metadata.get("workspace", _pick(details, "working_directory", "work_dir", "WorkDir")) if metadata else _pick(details, "working_directory", "work_dir", "WorkDir")),
        ]
    )

    if is_serve:
        conn = read_connection_info(job_id)
        if conn:
            engine = conn.get("engine", "vllm")
            node = conn.get("node", "--")
            port = conn.get("port", "--")
            endpoint = f"http://{node}:{port}/v1" if node != "--" and port != "--" else "--"
            view["Endpoint"] = endpoint
        else:
            view["Endpoint"] = "(not ready yet)"
        view["API Key"] = metadata.get("api_key", "--")
        view["Email"] = metadata.get("email", "--")
    else:
        view["Input"] = metadata.get("input", "--") if metadata else "--"
        view["Output"] = metadata.get("output", "--") if metadata else "--"

    view["Log (stdout)"] = stdout_path or "--"
    view["Log (stderr)"] = stderr_path or "--"

    pad = max(len(key) for key in view)
    for key, value in view.items():
        print(f"{key + ':':<{pad + 1}} {value}")

    if is_serve and state == "RUNNING":
        print(f"\nTip: Run `llmflux connect {job_id}` to get the full connection details.")
    else:
        print(f"\nTip: Run `llmflux logs {job_id}` to view job logs (stdout and stderr).")
    return 0


def _read_last_lines(path: Path, tail_count: int) -> list[str]:
    """Helper function to read the last N lines of a file."""
    if not path.exists():
        return [f"(missing: {path})"]
    if path.stat().st_size == 0:
        return ["(empty)"]
    tail_arg = str(tail_count) if tail_count > 0 else "+1"
    result = subprocess.run(
        ["tail", "-n", tail_arg, str(path)],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        error = result.stderr.strip() or result.stdout.strip() or "unknown tail error"
        return [f"(error reading {path}: {error})"]
    return result.stdout.splitlines()


def _logs_command(args: argparse.Namespace) -> int:
    """Handle the `logs` command.
    Shows the last N lines of stdout and stderr for a specific job.
    """
    job_id = str(args.job_id)
    registry = JobRegistry()
    metadata = registry.get_job(job_id)
    if metadata is None:
        print(f"Job {job_id} is not tracked by LLMFlux registry.", file=sys.stderr)
        return 1

    if args.stdout_only and args.stderr_only:
        print("Choose only one of --stdout-only or --stderr-only.", file=sys.stderr)
        return 2

    try:
        stdout_path, stderr_path = get_job_log_paths(job_id, str(metadata.get("logs_dir")))
    except SlurmCommandError as exc:
        print(f"Error resolving logs for job {job_id}: {exc}", file=sys.stderr)
        return 1

    selected_logs: list[tuple[str, Optional[str]]] = []
    if not args.stderr_only:
        selected_logs.append(("stdout", stdout_path))
    if not args.stdout_only:
        selected_logs.append(("stderr", stderr_path))

    resolved_files = [path for _, path in selected_logs if path]
    if not resolved_files:
        print(f"No log files resolved for job {job_id}.", file=sys.stderr)
        return 1

    missing_files = [
        (stream_name, file_path)
        for stream_name, file_path in selected_logs
        if file_path and not Path(file_path).exists()
    ]
    if missing_files:
        try:
            job_state = get_job_state(job_id)
        except SlurmCommandError:
            job_state = None

        if job_state == "PENDING":
            missing_streams = ", ".join(stream_name for stream_name, _ in missing_files)
            print(
                f"Job {job_id} is PENDING. {missing_streams} log file(s) are not created yet. "
                "Try again after the job starts running."
            )
            return 0

    if args.follow:
        command = ["tail", "-n", str(args.tail), "-f", *resolved_files]
        return subprocess.call(command)

    for stream_name, file_path in selected_logs:
        print(f"=== {stream_name} ({file_path or 'missing'}) ===")
        if not file_path:
            print("(unavailable)")
            print()
            continue
        for line in _read_last_lines(Path(file_path), args.tail):
            print(line)
        print()

    return 0


def _cancel_command(args: argparse.Namespace) -> int:
    """Handle the `cancel` command.
    Cancels a specific job.
    """
    job_id = str(args.job_id)
    registry = JobRegistry()
    if registry.get_job(job_id) is None:
        print(f"Job {job_id} is not tracked by LLMFlux registry.", file=sys.stderr)
        return 1

    try:
        cancel_job(job_id, force=bool(args.force))
    except SlurmCommandError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print(f"Job {job_id} cancelled successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llmflux", description="LLMFlux CLI")
    parser.add_argument(
        "--version",
        "-V",
        action="version",
        version=f"%(prog)s {_get_llmflux_version()}",
        help="Show llmflux version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_parser = subparsers.add_parser("run", help="Submit a batch processing job")
    run_parser.add_argument("--model", required=True, help="Model key from models.yaml, e.g., Llama-3.2-3B-Instruct, Qwen2.5-32B-Instruct")
    run_parser.add_argument("--input", required=True, help="Path to input JSONL file")
    run_parser.add_argument("--output", required=False, help="Path to output JSON file")

    # Common tuning options
    run_parser.add_argument("--batch-size", type=int, default=4)
    run_parser.add_argument("--save-frequency", type=int, default=50)
    run_parser.add_argument("--max-retries", type=int, default=3)
    run_parser.add_argument("--retry-delay", type=float, default=1.0)
    run_parser.add_argument("--max-tokens", type=int)
    run_parser.add_argument("--temperature", type=float)
    run_parser.add_argument("--top-p", type=float)
    run_parser.add_argument("--top-k", type=int)

    # SLURM configuration
    run_parser.add_argument("--account", type=str)
    run_parser.add_argument("--partition", type=str)
    run_parser.add_argument("--nodes", type=int)
    run_parser.add_argument("--gpus-per-node", type=int)
    run_parser.add_argument("--time", type=str)
    run_parser.add_argument("--mem", type=str)
    run_parser.add_argument("--cpus-per-task", type=int)
    run_parser.add_argument("--engine", type=str, default="vllm", choices=["ollama", "vllm"])
    run_parser.add_argument(
        "--sbatch-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Additional SBATCH directive (e.g., --sbatch-arg reservation=my_res). Can be used multiple times."
    )

    # Container rebuild control
    run_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the Apptainer/Singularity image before running",
    )

    # Debug mode
    run_parser.add_argument(
        "--debug",
        action="store_true",
        help="Preserve generated SLURM job script (job.sh) for debugging",
    )

    # VLLM specific options
    run_parser.add_argument("--vllm-engine-args", type=str, help="Additional arguments to pass to the vLLM engine")
    run_parser.add_argument(
        "--custom-config-path",
        type=str,
        help="Path to a custom model config YAML or models.yaml-style file (vLLM only)",
    )

    # Local execution toggle
    # Add support for this in the future - Can be directly used on the compute node
    # run_parser.add_argument("--local", action="store_true", help="Run locally without SLURM")

    run_parser.set_defaults(func=_run_command)

    # connect subcommand
    connect_parser = subparsers.add_parser("connect", help="Show connection info for a running serve job")
    connect_parser.add_argument("job_id", help="Slurm job ID of a running serve job")
    connect_parser.add_argument(
        "--local-port",
        type=int,
        default=8000,
        help="Unused; kept for CLI compatibility (access is direct to node:port)",
    )
    connect_parser.add_argument(
        "--wait-timeout",
        type=int,
        default=600,
        help="Seconds to wait for model to finish loading (default: 600)",
    )
    connect_parser.set_defaults(func=_connect_command)

    # serve subcommand
    serve_parser = subparsers.add_parser("serve", help="Start a model as a long-running service on a compute node")
    serve_parser.add_argument("--model", required=True, help="Model key from models.yaml")
    serve_parser.add_argument("--email", required=True, help="Email address for SLURM notification when service is ready")
    serve_parser.add_argument("--engine", type=str, default="vllm", choices=["ollama", "vllm"])

    # SLURM configuration
    serve_parser.add_argument("--account", type=str)
    serve_parser.add_argument("--partition", type=str)
    serve_parser.add_argument("--nodes", type=int)
    serve_parser.add_argument("--gpus-per-node", type=int)
    serve_parser.add_argument("--time", required=True, type=str, help="How long to keep the service up, e.g. 02:00:00")
    serve_parser.add_argument("--mem", type=str)
    serve_parser.add_argument("--cpus-per-task", type=int)
    serve_parser.add_argument(
        "--sbatch-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Additional SBATCH directive. Can be used multiple times.",
    )
    serve_parser.add_argument("--vllm-engine-args", type=str, help="Additional arguments to pass to the vLLM engine")
    serve_parser.add_argument("--rebuild", action="store_true", help="Force rebuild of the Apptainer/Singularity image")
    serve_parser.add_argument("--debug", action="store_true", help="Preserve generated SLURM job script for debugging")
    serve_parser.set_defaults(func=_serve_command)

    # benchmark subcommand
    benchmark_parser = subparsers.add_parser("benchmark", help="Run a benchmark job")
    benchmark_parser.add_argument("--model", required=True, help="Model key from list (use command llmflux show-models to see available models)")
    benchmark_parser.add_argument("--name", type=str, help="Benchmark run name (default: benchmark_{model})")
    benchmark_parser.add_argument("--num-prompts", type=int, default=50, help="Number of prompts to generate (default: 50)")
    benchmark_parser.add_argument("--input", type=str, help="Use existing prompts file instead of generating")
    benchmark_parser.add_argument("--output", type=str, help="Path to output JSON file")

    # Common tuning options
    benchmark_parser.add_argument("--batch-size", type=int, default=4)
    benchmark_parser.add_argument("--max-tokens", type=int)
    benchmark_parser.add_argument("--temperature", type=float)

    # SLURM configuration
    benchmark_parser.add_argument("--account", type=str)
    benchmark_parser.add_argument("--partition", type=str)
    benchmark_parser.add_argument("--nodes", type=int)
    benchmark_parser.add_argument("--gpus-per-node", type=int)
    benchmark_parser.add_argument("--time", type=str)
    benchmark_parser.add_argument("--mem", type=str)
    benchmark_parser.add_argument("--cpus-per-task", type=int)
    benchmark_parser.add_argument("--engine", type=str, default="vllm", choices=["ollama", "vllm"])
    benchmark_parser.add_argument(
        "--sbatch-arg",
        action="append",
        metavar="KEY=VALUE",
        help="Additional SBATCH directive (e.g., --sbatch-arg reservation=my_res). Can be used multiple times."
    )
    # Container rebuild control
    benchmark_parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Force rebuild of the Apptainer/Singularity image before running",
    )

    # Debug mode
    benchmark_parser.add_argument(
        "--debug",
        action="store_true",
        help="Preserve generated SLURM job script (job.sh) for debugging",
    )

    # VLLM specific options
    benchmark_parser.add_argument("--vllm-engine-args", type=str, help="Additional arguments to pass to the vLLM engine")
    benchmark_parser.add_argument(
        "--custom-config-path",
        type=str,
        help="Path to a custom model config YAML or models.yaml-style file (vLLM only)",
    )

    benchmark_parser.set_defaults(func=_benchmark_command)

    # show-models subcommand
    show_models_parser = subparsers.add_parser(
        "show-models",
        help="List all available model keys from models.yaml",
    )
    show_models_parser.set_defaults(func=_show_models_command)

    # jobs subcommand
    jobs_parser = subparsers.add_parser("jobs", help="List LLMFlux tracked Slurm jobs")
    jobs_parser.add_argument("--all", action="store_true", help="Include historical jobs")
    jobs_parser.add_argument(
        "--state",
        action="append",
        choices=["RUNNING", "PENDING", "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"],
        help="Filter by Slurm state (can be repeated)",
    )
    jobs_parser.set_defaults(func=_jobs_command)

    # status subcommand
    status_parser = subparsers.add_parser("status", help="Show detailed status for a job")
    status_parser.add_argument("job_id", help="Slurm job ID")
    status_parser.set_defaults(func=_status_command)

    # logs subcommand
    logs_parser = subparsers.add_parser("logs", help="Show last lines of stdout and stderr for a tracked job")
    logs_parser.add_argument("job_id", help="Slurm job ID")
    logs_parser.add_argument("--tail", type=int, default=100, help="Number of trailing lines to display")
    logs_parser.add_argument("-f", "--follow", action="store_true", help="Follow logs continuously")
    logs_parser.add_argument("--stdout-only", action="store_true", help="Show stdout only")
    logs_parser.add_argument("--stderr-only", action="store_true", help="Show stderr only")
    logs_parser.set_defaults(func=_logs_command)

    # cancel subcommand
    cancel_parser = subparsers.add_parser("cancel", help="Cancel a tracked running/pending job")
    cancel_parser.add_argument("job_id", help="Slurm job ID")
    cancel_parser.add_argument("--force", action="store_true", help="Force kill with scancel --signal=KILL")
    cancel_parser.set_defaults(func=_cancel_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
