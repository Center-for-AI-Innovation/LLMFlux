"""Tests for HF_HOME setup in SlurmRunner."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmflux.core.config import Config, SlurmConfig
from llmflux.slurm.runner import SlurmRunner


class TestHFHomeSetup(unittest.TestCase):
    """Validate HF_HOME handling for SLURM runs."""

    @staticmethod
    def _build_config(base_dir: Path) -> Config:
        """Create a minimal config rooted in a temp directory."""
        slurm_config = SlurmConfig(
            account="test-account",
            partition="test-partition",
            nodes=1,
            gpus_per_node=1,
            time="00:10:00",
            mem="8G",
            memory="8G",
            cpus_per_task=2,
            ntasks=1,
            ntasks_per_node=1,
        )
        return Config(
            data_dir=str(base_dir / "data"),
            models_dir=str(base_dir / "models"),
            logs_dir=str(base_dir / "logs"),
            containers_dir=str(base_dir / "containers"),
            slurm=slurm_config,
        )

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_setup_environment_creates_configured_hf_home(self, mock_config_manager):
        """HF_HOME from env should be created and passed through."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._build_config(temp_path)
            mock_config_manager.return_value.get_config.return_value = config

            hf_home = temp_path / "custom-hf-home"
            self.assertFalse(hf_home.exists())

            with patch.dict(os.environ, {"HF_HOME": str(hf_home)}, clear=False):
                runner = SlurmRunner(workspace=str(temp_path / "workspace"))
                env = runner._setup_environment()

            self.assertTrue(hf_home.exists())
            self.assertEqual(env["HF_HOME"], str(hf_home))
            self.assertEqual(env["APPTAINERENV_HF_HOME"], str(hf_home))

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_setup_environment_creates_default_hf_home(self, mock_config_manager):
        """Default HF_HOME should be created when env var is unset."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            config = self._build_config(temp_path)
            mock_config_manager.return_value.get_config.return_value = config

            fake_home = temp_path / "user-home"
            expected_hf_home = fake_home / ".cache" / "huggingface"
            self.assertFalse(expected_hf_home.exists())

            with patch("llmflux.slurm.runner.Path.home", return_value=fake_home):
                with patch.dict(os.environ, {}, clear=False):
                    os.environ.pop("HF_HOME", None)
                    runner = SlurmRunner(workspace=str(temp_path / "workspace"))
                    env = runner._setup_environment()

            self.assertTrue(expected_hf_home.exists())
            self.assertEqual(env["HF_HOME"], str(expected_hf_home))
            self.assertEqual(env["APPTAINERENV_HF_HOME"], str(expected_hf_home))


if __name__ == "__main__":
    unittest.main()
