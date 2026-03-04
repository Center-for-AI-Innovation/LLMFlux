"""Helpers for querying and controlling Slurm jobs for LLMFlux."""

import json
import os
import subprocess
from typing import Any, Optional

TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"}
ACTIVE_STATES = {"RUNNING", "PENDING"}


class SlurmCommandError(RuntimeError):
    """Raised when a Slurm command invocation fails."""


def _run_json_command(command: list[str]) -> Any:
    """Run a Slurm command and return the JSON payload, raising an error on failure."""
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise SlurmCommandError(f"Command failed: {' '.join(command)}\n{stderr}")

    payload = result.stdout.strip()
    if not payload:
        return {}

    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SlurmCommandError(
            f"Invalid JSON from command: {' '.join(command)}"
        ) from exc


def _extract_jobs(payload: Any) -> list[dict[str, Any]]:
    """Extract the list of job entries from a Slurm JSON payload."""
    if not isinstance(payload, dict):
        return []
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        return []
    return [entry for entry in jobs if isinstance(entry, dict)]


def _parse_state_value(raw: Any) -> Optional[str]:
    """Transforms the state string from any schema Slurm uses for the job state field into a standard string.
    Slurm 25+ sacct --json emits state as {"current": ["COMPLETED"], "reason": "None"}.
    Older versions and squeue use a plain string or a list of strings.
    """
    if isinstance(raw, dict):
        current = raw.get("current") or raw.get("CURRENT")
        if isinstance(current, list) and current:
            return str(current[0]).upper()
        if isinstance(current, str) and current.strip():
            return current.upper()
        return None
    if isinstance(raw, list):
        return str(raw[0]).upper() if raw else None
    if isinstance(raw, str) and raw.strip():
        return raw.upper()
    return None


def extract_state(job: dict[str, Any]) -> str:
    """Extract the state of a job from a Slurm JSON payload."""
    for key in ("job_state", "state"):
        result = _parse_state_value(job.get(key))
        if result:
            return result
    return "UNKNOWN"


def _extract_first_non_empty(job: dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    """Returns the first non-empty value from a list of keys in a Slurm JSON payload."""
    for key in keys:
        value = job.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _build_job_index(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Builds a dictionary of job IDs to job details from a list of Slurm JSON payloads."""
    indexed: dict[str, dict[str, Any]] = {}
    for job in jobs:
        # Removes the job ids for the batch step and extern step
        if not job.get("job_id") or isinstance(job.get("job_id"), str):
            continue
        job_id = str(job.get("job_id"))
        indexed[job_id] = job
    return indexed


def get_active_job_details(user: Optional[str] = None) -> dict[str, dict[str, Any]]:
    """Queries the Slurm queue for the active jobs of a user."""
    user_name = user if user else os.environ.get("USER")
    if not user_name:
        raise SlurmCommandError("Could not determine current user from USER env var.")
    payload = _run_json_command(["squeue", "--json", "-u", user_name])
    jobs = _extract_jobs(payload)
    return _build_job_index(jobs)


def get_list_of_jobs_details(job_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Query sacct for a specific set of job IDs from the LLMFlux registry.

    Using -j instead of -u avoids sacct's default one-day lookback window so
    jobs submitted on previous days are still returned.
    """
    if not job_ids:
        return {}
    payload = _run_json_command(["sacct", "--json", "-j", ",".join(job_ids)])
    jobs = _extract_jobs(payload)
    base_jobs = [job for job in jobs if ".batch" not in str(job.get("job_id", ""))]
    return _build_job_index(base_jobs)


def _try_job_source(command: list[str], job_id: str) -> dict[str, Any]:
    """Run a Slurm command and extract the job entry, returning {} on any failure.
    """
    try:
        payload = _run_json_command(command)
    except SlurmCommandError:
        return {}
    return _build_job_index(_extract_jobs(payload)).get(job_id, {})


def get_job_details(job_id: str) -> dict[str, Any]:
    """Queries the Slurm queue or accounting history for a specific job ID.
    Checks the queue first, then falls back to accounting history.
    """
    normalized_job_id = str(job_id)
    squeue_job = _try_job_source(["squeue", "--json", "-j", normalized_job_id], normalized_job_id)
    if squeue_job:
        return squeue_job

    # Job is no longer in the queue — fall back to accounting history
    return _try_job_source(["sacct", "--json", "-j", normalized_job_id], normalized_job_id)


def get_job_state(job_id: str) -> Optional[str]:
    "Gets the state of the job from the Job Details JSON payload."
    details = get_job_details(job_id)
    if not details:
        return None
    return extract_state(details)


def get_job_log_paths(job_id: str, logs_dir: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """Gets the log paths for a job from the Job Details JSON payload."""
    details = get_job_details(job_id)

    stdout_path = _extract_first_non_empty(details, ("stdout_expanded", "standard_output"))
    stderr_path = _extract_first_non_empty(details, ("stderr_expanded", "standard_error"))

    if not stdout_path and logs_dir:
        stdout_path = os.path.join(logs_dir, f"{job_id}.out")
    if not stderr_path and logs_dir:
        stderr_path = os.path.join(logs_dir, f"{job_id}.err")

    return stdout_path, stderr_path


def cancel_job(job_id: str, force: bool = False) -> None:
    """Cancels a job by sending a SIGKILL signal to the job."""
    command = ["scancel"]
    if force:
        command.append("--signal=KILL")
    command.append(str(job_id))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise SlurmCommandError(f"Failed to cancel job {job_id}: {stderr}")


