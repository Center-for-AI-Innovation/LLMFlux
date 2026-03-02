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
    if not isinstance(payload, dict):
        return []
    jobs = payload.get("jobs", [])
    if not isinstance(jobs, list):
        return []
    return [entry for entry in jobs if isinstance(entry, dict)]


def _extract_job_id(job: dict[str, Any]) -> Optional[str]:
    value = job.get("job_id")
    if value is None:
        return None
    return str(value).split(".")[0]


def _extract_state(job: dict[str, Any]) -> str:
    raw = job.get("job_state")
    if isinstance(raw, list):
        if not raw:
            return "UNKNOWN"
        return str(raw[0]).upper()
    if isinstance(raw, str):
        return raw.upper()
    return str(job.get("state") or "UNKNOWN").upper()


def _extract_first_non_empty(job: dict[str, Any], keys: tuple[str, ...]) -> Optional[str]:
    for key in keys:
        value = job.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def _build_job_index(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for job in jobs:
        job_id = _extract_job_id(job)
        if not job_id:
            continue
        indexed[job_id] = job
    return indexed


def current_user() -> str:
    user = os.environ.get("USER")
    if not user:
        raise SlurmCommandError("Could not determine current user from USER env var.")
    return user


def get_active_jobs(user: Optional[str] = None) -> dict[str, dict[str, Any]]:
    user_name = user or current_user()
    payload = _run_json_command(["squeue", "--json", "-u", user_name])
    jobs = _extract_jobs(payload)
    return _build_job_index(jobs)


def get_historical_jobs(user: Optional[str] = None) -> dict[str, dict[str, Any]]:
    user_name = user or current_user()
    payload = _run_json_command(["sacct", "--json", "-u", user_name])
    jobs = _extract_jobs(payload)
    base_jobs = [job for job in jobs if ".batch" not in str(job.get("job_id", ""))]
    return _build_job_index(base_jobs)


def get_job_details(job_id: str) -> dict[str, Any]:
    normalized_job_id = str(job_id)

    scontrol_payload = _run_json_command(["scontrol", "--json", "show", "job", normalized_job_id])
    scontrol_jobs = _build_job_index(_extract_jobs(scontrol_payload))
    scontrol_job = scontrol_jobs.get(normalized_job_id, {})

    squeue_payload = _run_json_command(["squeue", "--json", "-j", normalized_job_id])
    squeue_jobs = _build_job_index(_extract_jobs(squeue_payload))
    squeue_job = squeue_jobs.get(normalized_job_id, {})

    sacct_payload = _run_json_command(["sacct", "--json", "-j", normalized_job_id])
    sacct_jobs = _build_job_index(_extract_jobs(sacct_payload))
    sacct_job = sacct_jobs.get(normalized_job_id, {})

    merged = {**sacct_job, **squeue_job, **scontrol_job}
    return merged


def get_job_state(job_id: str) -> Optional[str]:
    details = get_job_details(job_id)
    if not details:
        return None
    return _extract_state(details)


def get_job_log_paths(job_id: str, logs_dir: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    details = get_job_details(job_id)

    stdout_path = _extract_first_non_empty(details, ("stdout_expanded", "standard_output"))
    stderr_path = _extract_first_non_empty(details, ("stderr_expanded", "standard_error"))

    if not stdout_path and logs_dir:
        stdout_path = os.path.join(logs_dir, f"{job_id}.out")
    if not stderr_path and logs_dir:
        stderr_path = os.path.join(logs_dir, f"{job_id}.err")

    return stdout_path, stderr_path


def cancel_job(job_id: str, force: bool = False) -> None:
    command = ["scancel"]
    if force:
        command.append("--signal=KILL")
    command.append(str(job_id))
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise SlurmCommandError(f"Failed to cancel job {job_id}: {stderr}")


def verify_cancelled(job_id: str) -> bool:
    state = get_job_state(job_id)
    if state is None:
        return True
    return state.startswith("CANCELLED")


def normalize_state(job: dict[str, Any]) -> str:
    return _extract_state(job)
