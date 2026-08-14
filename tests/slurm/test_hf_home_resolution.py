"""HF_HOME must be resolved before it becomes a bind source.

Apptainer binds the path it is given, not the path's target. A symlinked cache
directory therefore dangles inside the container, and the engine dies with
FileNotFoundError naming a directory that plainly exists when the user checks
from the login node. Observed on DeltaAI job 2945590, after the HF cache was
relocated by symlink to escape a home quota — which is exactly the situation
that makes someone symlink a cache in the first place.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestHfHomeResolution(unittest.TestCase):
    def test_symlinked_hf_home_is_resolved_to_its_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            real = Path(tmp) / "real-cache"
            real.mkdir()
            link = Path(tmp) / "linked-cache"
            link.symlink_to(real)

            from llmflux.core.config import Config, SlurmConfig, EngineConfig
            from llmflux.slurm.runner import SlurmRunner

            work = Path(tmp) / "work"
            for sub in ("data", "models", "logs", "containers"):
                (work / sub).mkdir(parents=True)
            cfg = Config(
                data_dir=str(work / "data"), models_dir=str(work / "models"),
                logs_dir=str(work / "logs"), containers_dir=str(work / "containers"),
                slurm=SlurmConfig(partition="gpu", nodes=1, gpus_per_node=1,
                                  time="01:00:00", memory="16G", cpus_per_task=4,
                                  account="acct"),
            )
            with patch("llmflux.slurm.runner.ConfigManager") as mgr:
                mgr.return_value.get_config.return_value = cfg
                runner = SlurmRunner(config=cfg.slurm,
                                     engine_config=EngineConfig(engine="vllm"))
                with patch.dict(os.environ, {"HF_HOME": str(link)}):
                    env = runner._setup_environment()

            self.assertEqual(
                env["HF_HOME"], str(real.resolve()),
                "a symlinked HF_HOME must be resolved, or the container bind dangles",
            )
            self.assertEqual(env["APPTAINERENV_HF_HOME"], str(real.resolve()))


if __name__ == "__main__":
    unittest.main()
