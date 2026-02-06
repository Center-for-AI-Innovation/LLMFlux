"""Tests for the SlurmRunner class."""

import os
import json
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
                    "model": "test-model",
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
            name="test-model",
            type="ollama",
            size="7b",
            parameters=self.model_params,
            path=None,
            description="Test model",
            capabilities=["text"],
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
        # Mock config manager to return our test config
        mock_config_manager.get_config.return_value = self.config

        runner = SlurmRunner()

        # Check properties
        self.assertEqual(runner.data_dir, Path(self.config.data_dir))
        self.assertEqual(runner.models_dir, Path(self.config.models_dir))
        self.assertEqual(runner.logs_dir, Path(self.config.logs_dir))
        self.assertEqual(runner.containers_dir, Path(self.config.containers_dir))

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_setup_environment(self, mock_config_manager):
        """Test environment setup for SlurmRunner."""
        # Mock config manager to return our test config
        mock_config_manager.get_config.return_value = self.config

        runner = SlurmRunner()
        env_vars = runner._setup_environment("test_workspace")

        # Check that env_vars includes our config paths
        self.assertEqual(env_vars["LLMFLUX_DATA_DIR"], str(self.data_dir))
        self.assertEqual(env_vars["LLMFLUX_MODELS_DIR"], str(self.models_dir))
        self.assertEqual(env_vars["LLMFLUX_LOGS_DIR"], str(self.logs_dir))
        self.assertEqual(env_vars["LLMFLUX_CONTAINERS_DIR"], str(self.containers_dir))
        self.assertEqual(env_vars["LLMFLUX_WORKSPACE"], "test_workspace")

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.socket.socket")
    def test_find_available_port(self, mock_socket, mock_config_manager):
        """Test finding an available port."""
        # Mock config manager
        mock_config_manager.get_config.return_value = self.config

        # Mock socket to test port finding
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance

        runner = SlurmRunner()
        port = runner._find_available_port()

        # Check that we got a port
        self.assertIsInstance(port, int)

        # Check that socket was used to find an available port
        mock_socket_instance.bind.assert_called_once()
        mock_socket_instance.close.assert_called_once()

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.socket.socket")
    @patch("llmflux.slurm.runner.os.makedirs")
    @patch("llmflux.slurm.runner.shutil.copy")
    @patch("builtins.open", new_callable=mock_open)
    @patch("llmflux.slurm.runner.subprocess.Popen")
    def test_run_method(
        self,
        mock_popen,
        mock_file,
        mock_copy,
        mock_makedirs,
        mock_socket,
        mock_config_manager,
    ):
        """Test the run method of SlurmRunner."""
        # Mock config manager
        mock_config_manager.get_config.return_value = self.config

        # Mock socket for port finding
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance

        # Mock process creation
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Submitted batch job 12345", b"")
        mock_popen.return_value = mock_process

        # Create runner and run job
        runner = SlurmRunner()
        job_id = runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model_name="test-model",
        )

        # Check that popen was called to submit job
        mock_popen.assert_called_once()

        # Check that we got a job id
        self.assertEqual(job_id, "12345")

        # Check that file was opened to write job script
        mock_file.assert_called()

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.subprocess.Popen")
    def test_run_error_handling(self, mock_popen, mock_config_manager):
        """Test error handling when submitting job."""
        # Mock config manager
        mock_config_manager.get_config.return_value = self.config

        # Mock process creation with error
        mock_process = MagicMock()
        mock_process.returncode = 1
        mock_process.communicate.return_value = (b"", b"Error submitting job")
        mock_popen.return_value = mock_process

        # Create runner and run job (should raise exception)
        runner = SlurmRunner()

        with self.assertRaises(RuntimeError):
            runner.run(
                input_path=str(self.jsonl_path),
                output_path=str(self.output_path),
                model_name="test-model",
            )

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.os.makedirs")
    @patch("llmflux.slurm.runner.os.path.exists")
    @patch("llmflux.slurm.runner.shutil.copy")
    @patch("builtins.open", new_callable=mock_open)
    @patch("llmflux.slurm.runner.subprocess.Popen")
    def test_create_job_script(
        self,
        mock_popen,
        mock_file,
        mock_copy,
        mock_exists,
        mock_makedirs,
        mock_config_manager,
    ):
        """Test creation of job script."""
        # Mock config manager
        mock_config_manager.get_config.return_value = self.config

        # Mock file existence check
        mock_exists.return_value = True

        # Mock process creation
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"Submitted batch job 12345", b"")
        mock_popen.return_value = mock_process

        # Create runner and run job
        runner = SlurmRunner()
        job_id = runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model_name="test-model",
        )

        # Check that file was opened to write job script
        mock_file.assert_called()

        # Extract the written content to verify it
        # Since we're using mock_open, we need to get the contents from write calls
        calls = [call[0][0] for call in mock_file().write.call_args_list]

        # Check that key elements are in the script
        script_content = "".join(calls)

        # Check basic SLURM directives
        self.assertIn("#SBATCH --partition=gpu", script_content)
        self.assertIn("#SBATCH --nodes=1", script_content)

        # Check processor commands
        self.assertIn("python -m llmflux.processors.batch", script_content)

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.socket.socket")
    @patch("llmflux.slurm.runner.os.path.exists")
    @patch("llmflux.slurm.runner.shutil.copy")
    def test_input_file_handling(self, mock_copy, mock_exists, mock_socket, mock_config_manager):
        """Test handling of input files."""
        # Mock config manager
        mock_config_manager.get_config.return_value = self.config

        # Mock socket for port finding
        mock_socket_instance = MagicMock()
        mock_socket.return_value = mock_socket_instance

        # Mock file existence check - pretend workspace file doesn't exist
        mock_exists.side_effect = lambda path: "workspace" not in path

        with patch("builtins.open", mock_open()):
            with patch("llmflux.slurm.runner.subprocess.Popen") as mock_popen:
                # Mock process creation
                mock_process = MagicMock()
                mock_process.returncode = 0
                mock_process.communicate.return_value = (b"Submitted batch job 12345", b"")
                mock_popen.return_value = mock_process

                # Create runner and run job
                runner = SlurmRunner()
                runner.run(
                    input_path=str(self.jsonl_path),
                    output_path=str(self.output_path),
                    model_name="test-model",
                )

                # Check that copy was called to copy input file to workspace
                mock_copy.assert_called()


if __name__ == "__main__":
    unittest.main()
