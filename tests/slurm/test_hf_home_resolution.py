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


class TestTritonCacheIsBound(unittest.TestCase):
    """Triton's cache must live inside a directory Apptainer binds.

    Triton defaults to ~/.triton. Apptainer binds $HOME but not what a symlink
    under it points at, so a relocated ~/.triton dangles inside the container
    and every worker dies with FileNotFoundError: '.../.triton/cache'. Observed
    on DeltaAI job 2946305 (4 nodes, 16 workers) after ~/.triton was relocated
    to /work to escape a home quota. FlashInfer needed the same treatment.
    """

    def test_triton_cache_dir_is_under_the_bound_xdg_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
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
                env = runner._setup_environment()

            triton = env["APPTAINERENV_TRITON_CACHE_DIR"]
            xdg = env["XDG_CACHE_HOME"]
            self.assertTrue(
                triton.startswith(xdg),
                f"triton cache {triton} must sit inside the bound {xdg}",
            )
            self.assertNotIn(".triton", triton.replace(xdg, ""),
                             "must not fall back to the home-relative default")
