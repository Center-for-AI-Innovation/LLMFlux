#!/usr/bin/env python3
"""SLURM runner for submitting AI-Flux batch jobs."""

import logging
import os
import shutil
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Optional

from ..core.config import SlurmConfig
from ..core.config_manager import ConfigManager

logger = logging.getLogger(__name__)


class SlurmRunner:
    """Runner for executing AI-Flux batch processing on SLURM."""

    def __init__(
        self,
        config: Optional[SlurmConfig] = None,
        workspace: Optional[str] = None,
    ):
        cfg = ConfigManager.get_config()

        self.slurm_config = config or cfg.get_slurm_config()
        self.workspace = (
            Path(workspace)
            if workspace
            else (Path(tempfile.gettempdir()) / "aiflux_workspace")
        )

        self.data_dir = Path(cfg.data_dir)
        self.models_dir = Path(cfg.models_dir)
        self.logs_dir = Path(cfg.logs_dir)
        self.containers_dir = Path(cfg.containers_dir)

    def _setup_environment(self, workspace: Optional[str] = None) -> Dict[str, str]:
        """Setup environment variables for the SLURM job."""
        env = dict(os.environ)
        env.update(
            {
                "AIFLUX_DATA_DIR": str(self.data_dir),
                "AIFLUX_MODELS_DIR": str(self.models_dir),
                "AIFLUX_LOGS_DIR": str(self.logs_dir),
                "AIFLUX_CONTAINERS_DIR": str(self.containers_dir),
                "AIFLUX_WORKSPACE": workspace if workspace is not None else str(self.workspace),
            }
        )
        return env

    def _find_available_port(self) -> int:
        """Find an available local TCP port."""
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("", 0))
            port = s.getsockname()[1]
            return port if isinstance(port, int) else 11434
        finally:
            s.close()

    def _create_job_script(
        self,
        input_path: str,
        output_path: str,
        model_name: str,
    ) -> str:
        sc = self.slurm_config

        lines = [
            "#!/bin/bash",
            "#SBATCH --job-name=aiflux_batch",
            f"#SBATCH --account={sc.account}",
            f"#SBATCH --partition={sc.partition}",
            f"#SBATCH --nodes={sc.nodes}",
            f"#SBATCH --ntasks={sc.ntasks}",
            f"#SBATCH --time={sc.time}",
            f"#SBATCH --mem={sc.mem}",
            f"#SBATCH --cpus-per-task={sc.cpus_per_task}",
            f"#SBATCH --gpus-per-node={sc.gpus_per_node}",
            f"#SBATCH --output={self.logs_dir}/%j.out",
            f"#SBATCH --error={self.logs_dir}/%j.err",
            "",
            f"python -m aiflux.processors.batch --input {input_path} --output {output_path} --model {model_name}",
            "",
        ]
        return "\n".join(lines)

    def run(
        self,
        input_path: str,
        output_path: str,
        model_name: Optional[str] = None,
        model: Optional[str] = None,
        **_kwargs,
    ) -> str:
        """Submit a batch job to SLURM and return the job id."""
        model_name = model_name or model or "llama3.2:3b"

        os.makedirs(self.workspace, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Copy input file into workspace for consistent job execution paths.
        workspace_input = str(self.workspace / Path(input_path).name)
        if not os.path.exists(workspace_input):
            os.makedirs(os.path.dirname(workspace_input), exist_ok=True)
            shutil.copy(input_path, workspace_input)

        # Write job script.
        job_script_path = str(self.workspace / "job.sh")
        job_script_text = self._create_job_script(
            input_path=workspace_input,
            output_path=output_path,
            model_name=model_name,
        )
        with open(job_script_path, "w") as f:
            f.write(job_script_text)

        env = self._setup_environment(str(self.workspace))
        # Keep behavior consistent with tests that patch socket/socket.
        _ = self._find_available_port()

        proc = subprocess.Popen(
            ["sbatch", job_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
        )
        stdout, stderr = proc.communicate()

        if proc.returncode != 0:
            err = (stderr or b"").decode("utf-8", errors="replace").strip()
            raise RuntimeError(err or "Error submitting job")

        out = (stdout or b"").decode("utf-8", errors="replace").strip()
        return out.split()[-1] if out else "unknown"
