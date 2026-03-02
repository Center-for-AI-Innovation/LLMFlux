#!/usr/bin/env python3
"""Maintains a registry of jobs for LLMFlux-submitted jobs."""

import os
import json
from pathlib import Path
from typing import Any, Dict, Optional


class JobRegistry:
    """Stores LLMFlux job metadata keyed by Slurm job ID.

    Registry entries are immutable after creation. Dynamic state (RUNNING, FAILED,
    COMPLETED, etc.) must be obtained from Slurm commands at read time.
    """

    def __init__(self, registry_file: str = os.path.expanduser("~/.llmflux/jobs.json")):
        self.registry_file = Path(registry_file)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.registry_file.exists():
            self.registry_file.write_text("{}", encoding="utf-8")
        self.jobs = self.load_jobs()

    def load_jobs(self) -> Dict[str, Dict[str, Any]]:
        with self.registry_file.open("r", encoding="utf-8") as file:
            data = json.load(file)
        if isinstance(data, dict):
            return data
        return {}

    def save_jobs(self) -> None:
        with self.registry_file.open("w", encoding="utf-8") as file:
            json.dump(self.jobs, file, indent=2, sort_keys=True)

    def create_job(self, job_id: str, metadata: Dict[str, Any]) -> None:
        """Create a new immutable metadata record for a job.

        Raises:
            ValueError: if job_id already exists.
        """
        normalized_id = str(job_id)
        if normalized_id in self.jobs:
            raise ValueError(f"Job {normalized_id} already exists in registry.")
        self.jobs[normalized_id] = metadata
        self.save_jobs()

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Return job metadata for a specific Slurm job ID."""
        return self.jobs.get(str(job_id))

    def get_all_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Return all tracked jobs keyed by Slurm job ID."""
        return self.jobs

    def get_all_job_ids(self) -> list[str]:
        """Return all tracked Slurm job IDs."""
        return list(self.jobs.keys())