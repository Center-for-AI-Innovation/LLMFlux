"""Tests for the SlurmRunner class."""

import os
import json
import subprocess
import tempfile
import unittest
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path

from llmflux.slurm.runner import SlurmRunner
from llmflux.core.config import Config, SlurmConfig, ModelConfig, ModelParameters


class TestSlurmRunner(unittest.TestCase):
    """Test suite for the SlurmRunner class."""

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        # Create test directories
        self.data_dir = self.test_dir / "data"
        self.models_dir = self.test_dir / "models"
        self.logs_dir = self.test_dir / "logs"
        self.containers_dir = self.test_dir / "containers"

        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.containers_dir, exist_ok=True)

        # Create test JSONL file
        self.jsonl_path = self.data_dir / "test.jsonl"
        self.entries = [
            {
                "custom_id": "test-1",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, world!"},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500,
                },
            }
        ]

        with open(self.jsonl_path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")

        # Create configs
        self.slurm_config = SlurmConfig(
            partition="gpu",
            nodes=1,
            ntasks=1,
            time="01:00:00",
            mem="16G",
            ntasks_per_node=1,
            cpus_per_task=4,
            gpus_per_node=1,
            account="project1",
        )

        self.model_params = ModelParameters(
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            top_k=40,
            stop_sequences=None,
        )

        self.model_config = ModelConfig(
            name="test:7b",
            hf_name="test/test-model",
            parameters=self.model_params,
        )

        self.config = Config(
            data_dir=str(self.data_dir),
            models_dir=str(self.models_dir),
            logs_dir=str(self.logs_dir),
            containers_dir=str(self.containers_dir),
            slurm=self.slurm_config,
            models=[self.model_config],
        )

        # Output path
        self.output_path = self.data_dir / "output.json"

    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_slurm_runner_initialization(self, mock_config_manager):
        """Test SlurmRunner initialization."""
        mock_config_manager.return_value.get_config.return_value = self.config

        runner = SlurmRunner()

        self.assertEqual(runner.data_dir, Path(self.config.data_dir))
        self.assertEqual(runner.models_dir, Path(self.config.models_dir))
        self.assertEqual(runner.logs_dir, Path(self.config.logs_dir))
        self.assertEqual(runner.containers_dir, Path(self.config.containers_dir))

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_setup_environment(self, mock_config_manager):
        """Test environment setup for SlurmRunner."""
        mock_config_manager.return_value.get_config.return_value = self.config

        runner = SlurmRunner()
        env_vars = runner._setup_environment("test_workspace")

        self.assertEqual(env_vars["MODELS_DIR"], str(self.models_dir))
        self.assertEqual(env_vars["LOGS_DIR"], str(self.logs_dir))
        self.assertEqual(env_vars["CONTAINERS_DIR"], str(self.containers_dir))
        self.assertEqual(env_vars["PROJECT_ROOT"], "test_workspace")

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.socket.socket")
    def test_find_available_port(self, mock_socket, mock_config_manager):
        """Test finding an available port."""
        mock_config_manager.return_value.get_config.return_value = self.config

        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance

        runner = SlurmRunner()
        port = runner._find_available_port()

        self.assertIsInstance(port, int)
        mock_socket_instance.bind.assert_called_once()
        mock_socket_instance.close.assert_called_once()

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_run_method(
        self,
        mock_run,
        mock_load_model_config,
        mock_registry,
        mock_config_manager,
    ):
        """Test the run method of SlurmRunner."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_config_manager.return_value.get_parameter.return_value = "4"
        mock_load_model_config.return_value = self.model_config

        mock_run.return_value.stdout = "Submitted batch job 12345"

        runner = SlurmRunner()
        job_id = runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model="test:7b",
        )

        mock_run.assert_called_once()
        self.assertEqual(job_id, "12345")

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_run_error_handling(self, mock_run, mock_load_model_config, mock_config_manager):
        """Test error handling when submitting job."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_config_manager.return_value.get_parameter.return_value = "4"
        mock_load_model_config.return_value = self.model_config

        mock_run.side_effect = subprocess.CalledProcessError(
            1, "sbatch", stderr="Error submitting job"
        )

        runner = SlurmRunner()

        with self.assertRaises(subprocess.CalledProcessError):
            runner.run(
                input_path=str(self.jsonl_path),
                output_path=str(self.output_path),
                model="test:7b",
            )

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_create_job_script(
        self,
        mock_run,
        mock_load_model_config,
        mock_registry,
        mock_config_manager,
    ):
        """Test creation of job script."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_config_manager.return_value.get_parameter.return_value = "4"
        mock_load_model_config.return_value = self.model_config

        mock_run.return_value.stdout = "Submitted batch job 12345"

        runner = SlurmRunner()
        job_script_path = runner.workspace / "job.sh"

        runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model="test:7b",
            debug=True,
        )

        try:
            script_content = job_script_path.read_text()
            self.assertIn("#SBATCH --partition=gpu", script_content)
            self.assertIn("#SBATCH --nodes=1", script_content)
            self.assertIn("BatchProcessor", script_content)
        finally:
            if job_script_path.exists():
                job_script_path.unlink()

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_input_file_handling(
        self,
        mock_run,
        mock_load_model_config,
        mock_registry,
        mock_config_manager,
    ):
        """Test handling of input files."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_config_manager.return_value.get_parameter.return_value = "4"
        mock_load_model_config.return_value = self.model_config

        mock_run.return_value.stdout = "Submitted batch job 12345"

        runner = SlurmRunner()
        runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model="test:7b",
        )

        # Verify the job was submitted, confirming the input file was resolved
        mock_run.assert_called_once()


class TestSlurmRunnerServe(unittest.TestCase):
    """Tests for SlurmRunner.serve()."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)

        self.slurm_config = SlurmConfig(
            partition="gpu",
            nodes=1,
            ntasks=1,
            time="02:00:00",
            mem="32G",
            ntasks_per_node=1,
            cpus_per_task=4,
            gpus_per_node=1,
            account="project1",
        )

        self.model_params = ModelParameters(
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            top_k=40,
            stop_sequences=None,
        )

        self.model_config = ModelConfig(
            name="test:7b",
            hf_name="test/test-model",
            parameters=self.model_params,
        )

        self.config = Config(
            data_dir=str(self.test_dir / "data"),
            models_dir=str(self.test_dir / "models"),
            logs_dir=str(self.test_dir / "logs"),
            containers_dir=str(self.test_dir / "containers"),
            slurm=self.slurm_config,
            models=[self.model_config],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_returns_job_id(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() returns the job ID on success."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        runner = SlurmRunner()
        job_id = runner.serve(email="user@example.com", model="test:7b")

        self.assertEqual(job_id, "55555")
        mock_run.assert_called_once()

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_sets_api_key_env(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() injects LLMFLUX_API_KEY into the sbatch environment."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        runner = SlurmRunner()
        runner.serve(email="user@example.com", model="test:7b")

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("LLMFLUX_API_KEY", env)
        self.assertTrue(env["LLMFLUX_API_KEY"].startswith("llmflux-"))

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_registry_metadata(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() writes type, email, and api_key to registry."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        mock_registry_instance = MagicMock()
        mock_registry.return_value = mock_registry_instance

        runner = SlurmRunner()
        runner.serve(email="user@example.com", model="test:7b")

        mock_registry_instance.create_job.assert_called_once()
        _, kwargs = mock_registry_instance.create_job.call_args
        metadata = kwargs["metadata"]
        self.assertEqual(metadata["type"], "serve")
        self.assertEqual(metadata["email"], "user@example.com")
        self.assertIn("api_key", metadata)
        self.assertTrue(metadata["api_key"].startswith("llmflux-"))

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.core.config.Config.load_model_config")
    def test_serve_invalid_model_raises(self, mock_load_model_config, mock_config_manager):
        """serve() raises ValueError when model is not found."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = None

        runner = SlurmRunner()
        with self.assertRaises(ValueError):
            runner.serve(email="user@example.com", model="nonexistent-model")

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.core.config.Config.load_model_config")
    def test_serve_engine_mismatch_raises(self, mock_load_model_config, mock_config_manager):
        """serve() raises ValueError when engine does not support the model."""
        from llmflux.core.config import EngineConfig

        mock_config_manager.return_value.get_config.return_value = self.config
        # Model with no Ollama name
        ollama_incompatible = ModelConfig(
            name="NA",
            hf_name="test/test-model",
            parameters=self.model_params,
        )
        mock_load_model_config.return_value = ollama_incompatible

        engine_config = EngineConfig(engine="ollama", home=str(self.test_dir / ".ollama"))
        runner = SlurmRunner(config=self.slurm_config, engine_config=engine_config)

        with self.assertRaises(ValueError):
            runner.serve(email="user@example.com", model="test:7b")

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_script_uses_serve_mode(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() generates a script with serve-mode markers (connection file, wait)."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        runner = SlurmRunner()
        runner.serve(email="user@example.com", model="test:7b", debug=True)

        job_script_path = runner.workspace / "job.sh"
        try:
            script = job_script_path.read_text()
            self.assertIn("connection.json", script)
            self.assertIn("LLMFLUX_API_KEY", script)
            self.assertIn("wait $VLLM_PID", script)
            self.assertIn("--mail-type=FAIL", script)
            self.assertNotIn("BatchProcessor", script)
        finally:
            if job_script_path.exists():
                job_script_path.unlink()

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_none_on_empty_sbatch_output(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() returns None when sbatch output is empty."""
        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = ""

        runner = SlurmRunner()
        result = runner.serve(email="user@example.com", model="test:7b")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
