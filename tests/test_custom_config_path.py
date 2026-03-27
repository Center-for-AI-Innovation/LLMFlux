import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmflux.cli import build_parser
from llmflux.core.config import Config, EngineConfig, SlurmConfig
from llmflux.slurm.runner import SlurmRunner


class TestCustomConfigPath(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name)

        self.data_dir = self.workspace / "data"
        self.input_dir = self.data_dir / "input"
        self.output_dir = self.data_dir / "output"
        self.models_dir = self.workspace / "models"
        self.logs_dir = self.workspace / "logs"
        self.containers_dir = self.workspace / "containers"
        self.local_model_dir = self.workspace / "qwen-output"

        for path in [
            self.input_dir,
            self.output_dir,
            self.models_dir,
            self.logs_dir,
            self.containers_dir,
            self.local_model_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.input_path = self.input_dir / "prompts.jsonl"
        self.input_path.write_text('{"custom_id":"1","body":{"messages":[{"role":"user","content":"hello"}]}}\n')

        (self.local_model_dir / "config.json").write_text("{}")

        self.custom_config_path = self.workspace / "custom-models.yaml"
        self.custom_config_path.write_text(
            "\n".join(
                [
                    "models:",
                    "  my-custom-qwen:",
                    "    name: NA",
                    f"    hf_name: {self.local_model_dir}",
                    "    resources:",
                    "      gpu_layers: 24",
                    "      gpu_memory: 16GB",
                    "      batch_size: 4",
                    "      max_concurrent: 1",
                    "    parameters:",
                    "      temperature: 0.7",
                    "      top_p: 0.9",
                    "      max_tokens: 2048",
                    "      stop_sequences: []",
                ]
            )
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load_model_config_from_custom_models_yaml(self):
        config = Config()

        model_config = config.load_model_config(
            "my-custom-qwen",
            custom_config_path=str(self.custom_config_path),
        )

        self.assertIsNotNone(model_config)
        self.assertEqual(model_config.type, "my-custom-qwen")
        self.assertEqual(model_config.hf_name, str(self.local_model_dir))
        self.assertEqual(model_config.parameters.max_tokens, 2048)

    def test_cli_parser_accepts_custom_config_path(self):
        parser = build_parser()

        args = parser.parse_args(
            [
                "run",
                "--model",
                "my-custom-qwen",
                "--input",
                "data/input/prompts.jsonl",
                "--custom-config-path",
                str(self.custom_config_path),
            ]
        )

        self.assertEqual(args.custom_config_path, str(self.custom_config_path))

    @patch("llmflux.slurm.runner.subprocess.run")
    @patch("llmflux.slurm.runner.socket.socket")
    @patch("llmflux.slurm.runner.ConfigManager")
    def test_runner_passes_custom_config_path_and_local_model_bind(
        self,
        mock_config_manager_cls,
        mock_socket,
        mock_subprocess_run,
    ):
        config = Config(
            data_dir=str(self.data_dir),
            models_dir=str(self.models_dir),
            logs_dir=str(self.logs_dir),
            containers_dir=str(self.containers_dir),
            slurm=SlurmConfig(account="project1", partition="gpu"),
        )

        config_manager = mock_config_manager_cls.return_value
        config_manager.get_config.return_value = config
        config_manager.get_parameter.side_effect = (
            lambda param_name, code_value=None, obj=None, env_var=None, default=None:
            code_value if code_value is not None else default
        )

        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ("", 54321)
        mock_socket.return_value = mock_socket_instance

        mock_subprocess_run.return_value = MagicMock(stdout="Submitted batch job 12345")

        runner = SlurmRunner(
            config=config.get_slurm_config(),
            workspace=str(self.workspace),
            engine_config=EngineConfig(engine="vllm", home=str(self.workspace / ".vllm")),
        )

        job_id = runner.run(
            input_path=str(self.input_path),
            output_path=str(self.output_dir / "results.json"),
            model="my-custom-qwen",
            custom_config_path=str(self.custom_config_path),
            debug=True,
        )

        self.assertEqual(job_id, "12345")

        submitted_env = mock_subprocess_run.call_args.kwargs["env"]
        self.assertEqual(
            submitted_env["APPTAINERENV_CUSTOM_CONFIG_PATH"],
            str(self.custom_config_path.resolve()),
        )
        self.assertEqual(submitted_env["VLLM_MODEL_NAME"], str(self.local_model_dir))

        job_script = (self.workspace / "job.sh").read_text()
        self.assertIn(
            "custom_config_path = os.environ.get('APPTAINERENV_CUSTOM_CONFIG_PATH') or None",
            job_script,
        )
        self.assertIn("Bind-mounting local model path: $VLLM_MODEL_NAME", job_script)
        self.assertIn('--bind "$APPTAINER_BIND_PATHS"', job_script)

    @patch("llmflux.slurm.runner.subprocess.run")
    @patch("llmflux.slurm.runner.ConfigManager")
    def test_runner_rejects_custom_config_path_for_ollama(
        self,
        mock_config_manager_cls,
        mock_subprocess_run,
    ):
        config = Config(
            data_dir=str(self.data_dir),
            models_dir=str(self.models_dir),
            logs_dir=str(self.logs_dir),
            containers_dir=str(self.containers_dir),
            slurm=SlurmConfig(account="project1", partition="gpu"),
        )

        config_manager = mock_config_manager_cls.return_value
        config_manager.get_config.return_value = config

        runner = SlurmRunner(
            config=config.get_slurm_config(),
            workspace=str(self.workspace),
            engine_config=EngineConfig(engine="ollama", home=str(self.workspace / ".ollama")),
        )

        job_id = runner.run(
            input_path=str(self.input_path),
            output_path=str(self.output_dir / "results.json"),
            model="my-custom-qwen",
            custom_config_path=str(self.custom_config_path),
        )

        self.assertEqual(job_id, "1")
        mock_subprocess_run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
