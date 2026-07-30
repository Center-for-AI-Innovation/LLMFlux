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
            memory="16G",
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
    def test_setup_environment_exports_cache_vars_to_host(self, mock_config_manager):
        """Cache dirs used by the batch script's mkdir/--bind must be host vars.

        If they are only set as APPTAINERENV_*, the workspace cache dir is never
        created or bound and FlashInfer fails with a read-only filesystem error.
        """
        mock_config_manager.return_value.get_config.return_value = self.config

        runner = SlurmRunner()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("XDG_CACHE_HOME", None)
            env_vars = runner._setup_environment("test_workspace")

        self.assertEqual(env_vars["XDG_CACHE_HOME"], str(Path("test_workspace") / ".cache"))
        self.assertEqual(env_vars["FLASHINFER_WORKSPACE_BASE"], "test_workspace")
        self.assertEqual(env_vars["APPTAINERENV_XDG_CACHE_HOME"], env_vars["XDG_CACHE_HOME"])
        self.assertEqual(
            env_vars["APPTAINERENV_FLASHINFER_WORKSPACE_BASE"],
            env_vars["FLASHINFER_WORKSPACE_BASE"],
        )

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_setup_environment_prefers_huggingface_token(self, mock_config_manager):
        """HUGGINGFACE_TOKEN wins when both token env vars are set."""
        mock_config_manager.return_value.get_config.return_value = self.config

        runner = SlurmRunner()
        with patch.dict(os.environ, {"HUGGINGFACE_TOKEN": "hf_primary", "HF_TOKEN": "hf_secondary"}):
            env_vars = runner._setup_environment("test_workspace")

        self.assertEqual(env_vars["APPTAINERENV_HF_TOKEN"], "hf_primary")

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_setup_environment_falls_back_to_hf_token(self, mock_config_manager):
        """HF_TOKEN is used when HUGGINGFACE_TOKEN is not set."""
        mock_config_manager.return_value.get_config.return_value = self.config

        runner = SlurmRunner()
        with patch.dict(os.environ, {"HF_TOKEN": "hf_secondary"}, clear=False):
            os.environ.pop("HUGGINGFACE_TOKEN", None)
            env_vars = runner._setup_environment("test_workspace")

        self.assertEqual(env_vars["APPTAINERENV_HF_TOKEN"], "hf_secondary")

    @patch("llmflux.slurm.runner.ConfigManager")
    def test_setup_environment_without_hf_token(self, mock_config_manager):
        """No token env vars set means no token is forwarded to the container."""
        mock_config_manager.return_value.get_config.return_value = self.config

        runner = SlurmRunner()
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("HUGGINGFACE_TOKEN", None)
            os.environ.pop("HF_TOKEN", None)
            env_vars = runner._setup_environment("test_workspace")

        self.assertNotIn("APPTAINERENV_HF_TOKEN", env_vars)

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
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_run_multi_gpu_auto_sets_tensor_parallel(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """run() with gpus_per_node>1 auto-sets tensor-parallel-size in VLLM_ENGINE_ARGS."""
        from llmflux.core.config import EngineConfig

        multi_gpu_slurm = SlurmConfig(
            partition="gpu",
            nodes=1,
            ntasks=1,
            time="01:00:00",
            memory="64G",
            ntasks_per_node=1,
            cpus_per_task=4,
            gpus_per_node=4,
            account="project1",
        )
        config = Config(
            data_dir=str(self.data_dir),
            models_dir=str(self.models_dir),
            logs_dir=str(self.logs_dir),
            containers_dir=str(self.containers_dir),
            slurm=multi_gpu_slurm,
            models=[self.model_config],
        )
        mock_config_manager.return_value.get_config.return_value = config
        mock_config_manager.return_value.get_parameter.return_value = "4"
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 12345"

        engine_config = EngineConfig(engine="vllm", home=str(self.test_dir / ".vllm"))
        runner = SlurmRunner(config=multi_gpu_slurm, engine_config=engine_config)
        runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model="test:7b",
        )

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("VLLM_ENGINE_ARGS", env)
        self.assertIn("--tensor-parallel-size", env["VLLM_ENGINE_ARGS"])
        self.assertIn("4", env["VLLM_ENGINE_ARGS"])

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_run_multi_gpu_respects_user_tensor_parallel(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """run() with gpus_per_node>1 does not override user-supplied tensor-parallel-size."""
        from llmflux.core.config import EngineConfig

        multi_gpu_slurm = SlurmConfig(
            partition="gpu",
            nodes=1,
            ntasks=1,
            time="01:00:00",
            memory="64G",
            ntasks_per_node=1,
            cpus_per_task=4,
            gpus_per_node=4,
            account="project1",
        )
        config = Config(
            data_dir=str(self.data_dir),
            models_dir=str(self.models_dir),
            logs_dir=str(self.logs_dir),
            containers_dir=str(self.containers_dir),
            slurm=multi_gpu_slurm,
            models=[self.model_config],
        )
        mock_config_manager.return_value.get_config.return_value = config
        mock_config_manager.return_value.get_parameter.return_value = "4"
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 12345"

        engine_config = EngineConfig(engine="vllm", home=str(self.test_dir / ".vllm"))
        runner = SlurmRunner(config=multi_gpu_slurm, engine_config=engine_config)
        runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model="test:7b",
            vllm_engine_args={"tensor-parallel-size": 2},
        )

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("VLLM_ENGINE_ARGS", env)
        self.assertIn("--tensor-parallel-size", env["VLLM_ENGINE_ARGS"])
        self.assertIn("2", env["VLLM_ENGINE_ARGS"])
        self.assertNotIn("4", env["VLLM_ENGINE_ARGS"])

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_run_single_gpu_no_tensor_parallel(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """run() with gpus_per_node=1 does not inject tensor-parallel-size."""
        from llmflux.core.config import EngineConfig

        mock_config_manager.return_value.get_config.return_value = self.config
        mock_config_manager.return_value.get_parameter.return_value = "4"
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 12345"

        engine_config = EngineConfig(engine="vllm", home=str(self.test_dir / ".vllm"))
        runner = SlurmRunner(config=self.slurm_config, engine_config=engine_config)
        runner.run(
            input_path=str(self.jsonl_path),
            output_path=str(self.output_path),
            model="test:7b",
        )

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("VLLM_ENGINE_ARGS", env)
        self.assertNotIn("tensor-parallel-size", env["VLLM_ENGINE_ARGS"])

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
            memory="32G",
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

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_vllm_sets_engine_env_vars(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() with vllm engine sets VLLM_MODEL_NAME, VLLM_HOST, and VLLM_ENGINE_ARGS."""
        from llmflux.core.config import EngineConfig

        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        engine_config = EngineConfig(engine="vllm", home=str(self.test_dir / ".vllm"))
        runner = SlurmRunner(config=self.slurm_config, engine_config=engine_config)
        runner.serve(email="user@example.com", model="test:7b")

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("VLLM_MODEL_NAME", env)
        self.assertEqual(env["VLLM_MODEL_NAME"], "test/test-model")
        self.assertIn("VLLM_HOST", env)
        self.assertIn("VLLM_ENGINE_ARGS", env)

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_multi_gpu_auto_sets_tensor_parallel(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() with gpus_per_node>1 auto-sets tensor-parallel-size in VLLM_ENGINE_ARGS."""
        from llmflux.core.config import EngineConfig

        multi_gpu_slurm = SlurmConfig(
            partition="gpu",
            nodes=1,
            ntasks=1,
            time="02:00:00",
            memory="128G",
            ntasks_per_node=1,
            cpus_per_task=8,
            gpus_per_node=2,
            account="project1",
        )
        config = Config(
            data_dir=str(self.test_dir / "data"),
            models_dir=str(self.test_dir / "models"),
            logs_dir=str(self.test_dir / "logs"),
            containers_dir=str(self.test_dir / "containers"),
            slurm=multi_gpu_slurm,
            models=[self.model_config],
        )
        mock_config_manager.return_value.get_config.return_value = config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        engine_config = EngineConfig(engine="vllm", home=str(self.test_dir / ".vllm"))
        runner = SlurmRunner(config=multi_gpu_slurm, engine_config=engine_config)
        runner.serve(email="user@example.com", model="test:7b")

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("VLLM_ENGINE_ARGS", env)
        self.assertIn("--tensor-parallel-size", env["VLLM_ENGINE_ARGS"])
        self.assertIn("2", env["VLLM_ENGINE_ARGS"])

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_multi_gpu_respects_user_tensor_parallel(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() with gpus_per_node>1 does not override user-supplied tensor-parallel-size."""
        from llmflux.core.config import EngineConfig

        multi_gpu_slurm = SlurmConfig(
            partition="gpu",
            nodes=1,
            ntasks=1,
            time="02:00:00",
            memory="128G",
            ntasks_per_node=1,
            cpus_per_task=8,
            gpus_per_node=4,
            account="project1",
        )
        config = Config(
            data_dir=str(self.test_dir / "data"),
            models_dir=str(self.test_dir / "models"),
            logs_dir=str(self.test_dir / "logs"),
            containers_dir=str(self.test_dir / "containers"),
            slurm=multi_gpu_slurm,
            models=[self.model_config],
        )
        mock_config_manager.return_value.get_config.return_value = config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        engine_config = EngineConfig(engine="vllm", home=str(self.test_dir / ".vllm"))
        runner = SlurmRunner(config=multi_gpu_slurm, engine_config=engine_config)
        runner.serve(
            email="user@example.com",
            model="test:7b",
            vllm_engine_args={"tensor-parallel-size": 2},
        )

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("VLLM_ENGINE_ARGS", env)
        self.assertIn("--tensor-parallel-size", env["VLLM_ENGINE_ARGS"])
        self.assertIn("2", env["VLLM_ENGINE_ARGS"])
        self.assertNotIn("4", env["VLLM_ENGINE_ARGS"])

    @patch("llmflux.slurm.runner.ConfigManager")
    @patch("llmflux.slurm.runner.JobRegistry")
    @patch("llmflux.core.config.Config.load_model_config")
    @patch("llmflux.slurm.runner.subprocess.run")
    def test_serve_single_gpu_no_tensor_parallel(
        self, mock_run, mock_load_model_config, mock_registry, mock_config_manager
    ):
        """serve() with gpus_per_node=1 does not inject tensor-parallel-size."""
        from llmflux.core.config import EngineConfig

        mock_config_manager.return_value.get_config.return_value = self.config
        mock_load_model_config.return_value = self.model_config
        mock_run.return_value.stdout = "Submitted batch job 55555"

        engine_config = EngineConfig(engine="vllm", home=str(self.test_dir / ".vllm"))
        runner = SlurmRunner(config=self.slurm_config, engine_config=engine_config)
        runner.serve(email="user@example.com", model="test:7b")

        _, call_kwargs = mock_run.call_args
        env = call_kwargs["env"]
        self.assertIn("VLLM_ENGINE_ARGS", env)
        self.assertNotIn("tensor-parallel-size", env["VLLM_ENGINE_ARGS"])


class TestLoadVllmEngineArgs(unittest.TestCase):
    def _make_runner(self):
        with patch("llmflux.slurm.runner.ConfigManager") as mock_cm:
            mock_cm.return_value.get_config.return_value = MagicMock()
            return SlurmRunner()

    def test_none_returns_empty(self):
        runner = self._make_runner()
        self.assertEqual(runner._load_vllm_engine_args(None, "test"), {})

    def test_dict_returned_as_is(self):
        runner = self._make_runner()
        d = {"max_model_len": 4096}
        self.assertEqual(runner._load_vllm_engine_args(d, "test"), d)

    def test_empty_string_returns_empty(self):
        runner = self._make_runner()
        self.assertEqual(runner._load_vllm_engine_args("", "test"), {})
        self.assertEqual(runner._load_vllm_engine_args("   ", "test"), {})

    def test_valid_json_object_string(self):
        runner = self._make_runner()
        result = runner._load_vllm_engine_args('{"max_model_len": 4096}', "test")
        self.assertEqual(result, {"max_model_len": 4096})

    def test_invalid_json_string_returns_empty(self):
        runner = self._make_runner()
        self.assertEqual(runner._load_vllm_engine_args("{bad json}", "test"), {})

    def test_json_array_returns_empty(self):
        runner = self._make_runner()
        self.assertEqual(runner._load_vllm_engine_args("[1, 2, 3]", "test"), {})

    def test_non_string_non_dict_returns_empty(self):
        runner = self._make_runner()
        self.assertEqual(runner._load_vllm_engine_args(42, "test"), {})
        self.assertEqual(runner._load_vllm_engine_args(["a"], "test"), {})


class TestBuildVllmEngineArgs(unittest.TestCase):
    def _make_runner(self):
        with patch("llmflux.slurm.runner.ConfigManager") as mock_cm:
            mock_cm.return_value.get_config.return_value = MagicMock()
            return SlurmRunner()

    def test_bool_true_emits_flag_only(self):
        runner = self._make_runner()
        result = runner._build_vllm_engine_args({"enable-prefix-caching": True})
        self.assertIn("--enable-prefix-caching", result)
        # No value after the flag
        parts = result.split()
        idx = parts.index("--enable-prefix-caching")
        self.assertEqual(idx, len(parts) - 1)

    def test_bool_false_omits_flag(self):
        runner = self._make_runner()
        result = runner._build_vllm_engine_args({"enable-prefix-caching": False})
        self.assertNotIn("enable-prefix-caching", result)

    def test_none_value_omits_key(self):
        runner = self._make_runner()
        result = runner._build_vllm_engine_args({"max_model_len": None})
        self.assertEqual(result.strip(), "")

    def test_int_value(self):
        runner = self._make_runner()
        result = runner._build_vllm_engine_args({"max_model_len": 4096})
        self.assertIn("--max_model_len", result)
        self.assertIn("4096", result)

    def test_key_already_prefixed(self):
        runner = self._make_runner()
        result = runner._build_vllm_engine_args({"--max_model_len": 512})
        parts = result.split()
        flags = [p for p in parts if p.startswith("--")]
        # Should appear exactly once, not doubled
        self.assertEqual(flags.count("--max_model_len"), 1)

    def test_unsupported_type_omitted(self):
        runner = self._make_runner()
        result = runner._build_vllm_engine_args({"bad_arg": [1, 2, 3]})
        self.assertEqual(result.strip(), "")


class TestBuildJobName(unittest.TestCase):
    def _make_runner(self):
        with patch("llmflux.slurm.runner.ConfigManager") as mock_cm:
            mock_cm.return_value.get_config.return_value = MagicMock()
            return SlurmRunner()

    def test_special_chars_replaced(self):
        runner = self._make_runner()
        name = runner._build_job_name("meta-llama/Llama-3.2-3B")
        self.assertRegex(name, r"^llmflux_[a-zA-Z0-9_]+_")

    def test_empty_identifier_uses_fallback(self):
        runner = self._make_runner()
        name = runner._build_job_name("---")
        self.assertIn("llmflux_model_", name)

    def test_name_truncated_to_120(self):
        runner = self._make_runner()
        name = runner._build_job_name("a" * 200)
        self.assertLessEqual(len(name), 120)


if __name__ == "__main__":
    unittest.main()
