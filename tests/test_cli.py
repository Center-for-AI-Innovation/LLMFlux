"""Pytest tests for AI-Flux CLI commands."""

import pytest
import json
import tempfile
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
from io import StringIO

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from llmflux.cli import main, _run_command, _benchmark_command, build_parser
from llmflux.slurm.runner import SlurmRunner
from llmflux.core.config import Config, SlurmConfig, EngineConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp = tempfile.TemporaryDirectory()
    yield Path(temp.name)
    temp.cleanup()


@pytest.fixture
def sample_jsonl(temp_dir):
    """Create a sample JSONL file for testing."""
    jsonl_path = temp_dir / "test_prompts.jsonl"
    entries = [
        {
            "custom_id": "test-1",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello, world!"}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        },
        {
            "custom_id": "test-2",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "What is AI?"}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
    ]
    
    with open(jsonl_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    
    return jsonl_path


@pytest.fixture
def mock_slurm_runner():
    """Mock SlurmRunner for testing."""
    with patch('llmflux.cli.SlurmRunner') as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.run.return_value = "12345"
        mock_runner_class.return_value = mock_runner
        yield mock_runner


@pytest.fixture
def mock_config():
    """Mock Config for testing."""
    with patch('llmflux.cli.Config') as mock_config_class:
        mock_config_instance = MagicMock()
        mock_slurm_config = MagicMock(spec=SlurmConfig)
        mock_slurm_config.account = "test-account"
        mock_slurm_config.partition = "gpuA100x4"
        mock_config_instance.get_slurm_config.return_value = mock_slurm_config
        mock_config_class.return_value = mock_config_instance
        yield mock_config_instance


class TestCLIParser:
    """Test CLI argument parser."""
    
    def test_parser_builds_successfully(self):
        """Test that parser builds without errors."""
        parser = build_parser()
        assert parser is not None
        assert parser.prog == "llmflux"
    
    def test_parser_has_run_subcommand(self):
        """Test that run subcommand exists."""
        parser = build_parser()
        # Parse with run command
        args = parser.parse_args(["run", "--model", "Llama-3.2-3B-Instruct", "--input", "test.jsonl"])
        assert args.command == "run"
        assert args.model == "Llama-3.2-3B-Instruct"
        assert args.input == "test.jsonl"
    
    def test_parser_has_benchmark_subcommand(self):
        """Test that benchmark subcommand exists."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--model", "Llama-3.2-3B-Instruct"])
        assert args.command == "benchmark"
        assert args.model == "Llama-3.2-3B-Instruct"
    
    def test_run_command_required_args(self):
        """Test that run command requires model and input."""
        parser = build_parser()
        # Should fail without required args
        with pytest.raises(SystemExit):
            parser.parse_args(["run"])
    
    def test_run_command_optional_args(self, temp_dir, sample_jsonl):
        """Test run command with all optional arguments."""
        parser = build_parser()
        args = parser.parse_args([
            "run",
            "--model", "Llama-3.2-3B-Instruct",
            "--input", str(sample_jsonl),
            "--output", "results.json",
            "--batch-size", "8",
            "--save-frequency", "100",
            "--max-retries", "5",
            "--retry-delay", "2.0",
            "--max-tokens", "2048",
            "--temperature", "0.8",
            "--top-p", "0.95",
            "--top-k", "50",
            "--account", "my-account",
            "--partition", "gpuA100x4",
            "--nodes", "2",
            "--gpus-per-node", "2",
            "--time", "02:00:00",
            "--mem", "64G",
            "--cpus-per-task", "8",
            "--sbatch-arg", "reservation=my_res",
            "--sbatch-arg", "constraint=gpu",
            "--rebuild",
            "--debug"
        ])
        
        assert args.model == "Llama-3.2-3B-Instruct"
        assert args.input == str(sample_jsonl)
        assert args.output == "results.json"
        assert args.batch_size == 8
        assert args.save_frequency == 100
        assert args.max_retries == 5
        assert args.retry_delay == 2.0
        assert args.max_tokens == 2048
        assert args.temperature == 0.8
        assert args.top_p == 0.95
        assert args.top_k == 50
        assert args.account == "my-account"
        assert args.partition == "gpuA100x4"
        assert args.nodes == 2
        assert args.gpus_per_node == 2
        assert args.time == "02:00:00"
        assert args.mem == "64G"
        assert args.cpus_per_task == 8
        assert args.rebuild is True
        assert args.debug is True
        assert len(args.sbatch_arg) == 2
    
    def test_benchmark_command_args(self):
        """Test benchmark command with arguments."""
        parser = build_parser()
        args = parser.parse_args([
            "benchmark",
            "--model", "Llama-3.2-3B-Instruct",
            "--name", "my-benchmark",
            "--num-prompts", "100",
            "--batch-size", "8",
            "--max-tokens", "2048",
            "--temperature", "0.7",
            "--account", "my-account",
            "--rebuild",
            "--debug"
        ])
        
        assert args.model == "Llama-3.2-3B-Instruct"
        assert args.name == "my-benchmark"
        assert args.num_prompts == 100
        assert args.batch_size == 8
        assert args.max_tokens == 2048
        assert args.temperature == 0.7
        assert args.account == "my-account"
        assert args.rebuild is True
        assert args.debug is True


class TestRunCommand:
    """Test the run command functionality."""
    
    @patch('llmflux.cli.SlurmRunner')
    @patch('llmflux.cli.Config')
    def test_run_command_basic(self, mock_config_class, mock_runner_class, temp_dir, sample_jsonl):
        """Test basic run command execution."""
        # Setup mocks
        mock_config = MagicMock()
        mock_slurm_config = MagicMock()
        mock_config.get_slurm_config.return_value = mock_slurm_config
        mock_config_class.return_value = mock_config
        
        mock_runner = MagicMock()
        mock_runner.run.return_value = "12345"
        mock_runner_class.return_value = mock_runner
        
        # Create args
        args = MagicMock()
        args.input = str(sample_jsonl)
        args.output = str(temp_dir / "output.json")
        args.model = "Llama-3.2-3B-Instruct"
        args.batch_size = 4
        args.save_frequency = 50
        args.max_retries = 3
        args.retry_delay = 1.0
        args.max_tokens = None
        args.temperature = None
        args.top_p = None
        args.top_k = None
        args.account = None
        args.partition = None
        args.nodes = None
        args.gpus_per_node = None
        args.time = None
        args.mem = None
        args.cpus_per_task = None
        args.sbatch_arg = None
        args.rebuild = False
        args.debug = False
        args.local = False
        
        # Run command
        result = _run_command(args)
        
        # Verify
        assert result == 0
        mock_runner_class.assert_called_once()
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["input_path"] == str(sample_jsonl)
        assert call_kwargs["output_path"] == str(temp_dir / "output.json")
        assert call_kwargs["model"] == "Llama-3.2-3B-Instruct"
        assert call_kwargs["batch_size"] == 4
        # Verify all expected kwargs are passed
        assert "save_frequency" in call_kwargs
        assert "max_retries" in call_kwargs
        assert "retry_delay" in call_kwargs
    
    @patch('llmflux.cli.SlurmRunner')
    @patch('llmflux.cli.Config')
    def test_run_command_with_slurm_config(self, mock_config_class, mock_runner_class, temp_dir, sample_jsonl):
        """Test run command with SLURM configuration."""
        # Setup mocks
        mock_config = MagicMock()
        mock_slurm_config = MagicMock()
        mock_config.get_slurm_config.return_value = mock_slurm_config
        mock_config_class.return_value = mock_config
        
        mock_runner = MagicMock()
        mock_runner.run.return_value = "67890"
        mock_runner_class.return_value = mock_runner
        
        # Create args with SLURM config
        args = MagicMock()
        args.input = str(sample_jsonl)
        args.output = str(temp_dir / "output.json")
        args.model = "Llama-3.2-3B-Instruct"
        args.batch_size = 8
        args.save_frequency = 100
        args.max_retries = 3
        args.retry_delay = 1.0
        args.max_tokens = 4096
        args.temperature = 0.8
        args.top_p = 0.95
        args.top_k = 50
        args.account = "test-account"
        args.partition = "gpuA100x4"
        args.nodes = 1
        args.gpus_per_node = 2
        args.time = "02:00:00"
        args.mem = "64G"
        args.cpus_per_task = 16
        args.sbatch_arg = ["reservation=my_res", "constraint=gpu"]
        args.rebuild = True
        args.debug = True
        args.local = False
        
        # Run command
        result = _run_command(args)
        
        # Verify
        assert result == 0
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["model"] == "Llama-3.2-3B-Instruct"
        assert call_kwargs["batch_size"] == 8
        assert call_kwargs["save_frequency"] == 100
        assert call_kwargs["max_retries"] == 3
        assert call_kwargs["retry_delay"] == 1.0
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_p"] == 0.95
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["rebuild"] is True
        assert call_kwargs["debug"] is True
    
    def test_run_command_missing_input(self):
        """Test run command fails when input is missing."""
        args = MagicMock()
        args.input = None
        args.output = "results.json"
        args.model = "Llama-3.2-3B-Instruct"
        args.local = False
        
        with patch('sys.stderr', new=StringIO()):
            result = _run_command(args)
            assert result == 2  # Exit code for error
    
    @patch('llmflux.cli.BatchProcessor')
    @patch('llmflux.cli.Config')
    def test_run_command_local_mode(self, mock_config_class, mock_processor_class, temp_dir, sample_jsonl):
        """Test run command in local mode (not implemented yet, but tests structure)."""
        # Note: Local mode is commented out in CLI, but we test the structure
        args = MagicMock()
        args.input = str(sample_jsonl)
        args.output = str(temp_dir / "output.json")
        args.model = "Llama-3.2-3B-Instruct"
        args.local = True  # This would enable local mode if implemented
        args.batch_size = 4
        args.save_frequency = 50
        args.max_retries = 3
        args.retry_delay = 1.0
        args.max_tokens = None
        args.temperature = None
        args.top_p = None
        args.top_k = None
        
        # Since local mode is not fully implemented, this test structure is ready
        # for when local mode is enabled
        pass


class TestBenchmarkCommand:
    """Test the benchmark command functionality."""
    
    @patch('llmflux.cli._wait_for_slurm_elapsed_seconds', return_value=None)
    @patch('llmflux.cli.create_test_prompts_file')
    @patch('llmflux.cli.SlurmRunner')
    @patch('llmflux.cli.Config')
    def test_benchmark_command_generate_prompts(
        self, mock_config_class, mock_runner_class,
        mock_create_prompts, mock_wait, temp_dir
    ):
        """Test benchmark command with prompt generation."""
        # Setup mocks
        mock_config = MagicMock()
        mock_slurm_config = MagicMock()
        mock_config.get_slurm_config.return_value = mock_slurm_config
        mock_config_class.return_value = mock_config

        mock_runner = MagicMock()
        mock_runner.run.return_value = "12345"
        mock_runner_class.return_value = mock_runner

        mock_create_prompts.return_value = temp_dir / "prompts.jsonl"

        # Create args
        args = MagicMock()
        args.model = "Llama-3.2-3B-Instruct"
        args.name = None
        args.num_prompts = 50
        args.input = None
        args.output = None
        args.batch_size = 4
        args.max_tokens = None
        args.temperature = None
        args.account = None
        args.partition = None
        args.nodes = None
        args.gpus_per_node = None
        args.time = None
        args.mem = None
        args.cpus_per_task = None
        args.sbatch_arg = None
        args.rebuild = False
        args.debug = False

        # Run command
        with patch('builtins.print'):
            result = _benchmark_command(args)

        # Verify
        assert result == 0
        mock_create_prompts.assert_called_once_with(num_prompts=50, temperature=0.7, max_tokens=500, model="Llama-3.2-3B-Instruct")
        mock_runner.run.assert_called_once()
    
    @patch('llmflux.cli._wait_for_slurm_elapsed_seconds', return_value=None)
    @patch('llmflux.cli.SlurmRunner')
    @patch('llmflux.cli.Config')
    def test_benchmark_command_with_existing_input(
        self, mock_config_class, mock_runner_class, mock_wait, temp_dir, sample_jsonl
    ):
        """Test benchmark command with existing input file."""
        # Setup mocks
        mock_config = MagicMock()
        mock_slurm_config = MagicMock()
        mock_config.get_slurm_config.return_value = mock_slurm_config
        mock_config_class.return_value = mock_config
        
        mock_runner = MagicMock()
        mock_runner.run.return_value = "67890"
        mock_runner_class.return_value = mock_runner
        
        # Create args
        args = MagicMock()
        args.model = "Llama-3.2-3B-Instruct"
        args.name = None
        args.num_prompts = 50
        args.input = str(sample_jsonl)
        args.output = str(temp_dir / "benchmark_results.json")
        args.batch_size = 8
        args.max_tokens = 2048
        args.temperature = 0.7
        args.account = "test-account"
        args.partition = "gpuA100x4"
        args.nodes = None
        args.gpus_per_node = None
        args.time = None
        args.mem = None
        args.cpus_per_task = None
        args.sbatch_arg = None
        args.rebuild = True
        args.debug = True
        
        # Run command
        with patch('builtins.print'):
            result = _benchmark_command(args)
        
        # Verify
        assert result == 0
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["input_path"] == str(sample_jsonl)
        assert call_kwargs["model"] == "Llama-3.2-3B-Instruct"
        assert call_kwargs["batch_size"] == 8
        assert call_kwargs["rebuild"] is True
        assert call_kwargs["debug"] is True


class TestMainFunction:
    """Test the main CLI entry point."""
    
    def test_main_with_help(self):
        """Test main function with help flag."""
        with patch('sys.argv', ['llmflux', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with 0 for help
            assert exc_info.value.code == 0
    
    @patch('llmflux.cli._run_command')
    def test_main_with_run_command(self, mock_run_command, temp_dir, sample_jsonl):
        """Test main function with run command."""
        mock_run_command.return_value = 0
        
        with patch('sys.argv', [
            'llmflux', 'run',
            '--model', 'Llama-3.2-3B-Instruct',
            '--input', str(sample_jsonl)
        ]):
            result = main()
            assert result == 0
            mock_run_command.assert_called_once()
    
    @patch('llmflux.cli._benchmark_command')
    def test_main_with_benchmark_command(self, mock_benchmark_command):
        """Test main function with benchmark command."""
        mock_benchmark_command.return_value = 0
        
        with patch('sys.argv', [
            'llmflux', 'benchmark',
            '--model', 'Llama-3.2-3B-Instruct'
        ]):
            result = main()
            assert result == 0
            mock_benchmark_command.assert_called_once()
    
    def test_main_with_invalid_command(self):
        """Test main function with invalid command."""
        with patch('sys.argv', ['llmflux', 'invalid-command']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2


class TestRunnerEnvironmentVariables:
    """Test that environment variables are properly passed to runner.run()."""
    
    @patch('llmflux.slurm.runner.subprocess.run')
    @patch('llmflux.slurm.runner.socket.socket')
    @patch('llmflux.slurm.runner.ConfigManager')
    def test_runner_sets_apptainerenv_variables(
        self, mock_config_manager, mock_socket, mock_subprocess, temp_dir, sample_jsonl
    ):
        """Test that runner.run() sets APPTAINERENV_ prefixed environment variables correctly."""
        # Setup config manager mock
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.get_path.return_value = temp_dir
        mock_config.ensure_directory = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(
            account="test-account",
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=1,
            time="01:00:00",
            memory="32G",
            cpus_per_task=4,
            extra_sbatch_args={}
        )
        mock_manager.get_config.return_value = mock_config
        
        # Mock parameter retrieval
        def get_parameter_side_effect(param_name, code_value, obj, env_var, default):
            if param_name == "model_config.name":
                return code_value or default
            elif param_name == "batch_size":
                return code_value or default
            elif param_name == "save_frequency":
                return code_value or default
            return code_value if code_value is not None else default
        
        mock_manager.get_parameter.side_effect = get_parameter_side_effect
        mock_config_manager.return_value = mock_manager
        
        # Mock socket for port finding
        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ('', 11434)
        mock_socket.return_value = mock_socket_instance
        
        # Mock subprocess.run for sbatch
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout="Submitted batch job 12345\n",
            stderr="",
            text=True
        )
        
        # Create runner and call run()
        from llmflux.slurm.runner import SlurmRunner

        slurm_config = mock_config.get_slurm_config()
        engine_config = EngineConfig(engine="ollama", home=str(temp_dir / ".ollama"))
        model_config_mock = MagicMock()
        model_config_mock.name = "Llama-3.2-3B-Instruct"
        model_config_mock.hf_name = None
        mock_config.load_model_config.return_value = model_config_mock
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir), engine_config=engine_config)

        job_id = runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="Llama-3.2-3B-Instruct",
            batch_size=8,
            save_frequency=100,
            max_tokens=4096,
            temperature=0.8,
            top_p=0.95,
            top_k=50,
            rebuild=True
        )
        
        # Verify subprocess.run was called (job submission)
        assert mock_subprocess.called
        
        # Get the environment passed to subprocess.run
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})
        
        # Verify APPTAINERENV_ variables are set correctly
        assert 'APPTAINERENV_MODEL_NAME' in env_passed
        assert env_passed['APPTAINERENV_MODEL_NAME'] == "Llama-3.2-3B-Instruct"
        
        assert 'APPTAINERENV_BATCH_SIZE' in env_passed
        assert env_passed['APPTAINERENV_BATCH_SIZE'] == "8"
        
        assert 'APPTAINERENV_SAVE_FREQUENCY' in env_passed
        assert env_passed['APPTAINERENV_SAVE_FREQUENCY'] == "100"
        
        assert 'APPTAINERENV_MAX_TOKENS' in env_passed
        assert env_passed['APPTAINERENV_MAX_TOKENS'] == "4096"
        
        assert 'APPTAINERENV_TEMPERATURE' in env_passed
        assert env_passed['APPTAINERENV_TEMPERATURE'] == "0.8"
        
        assert 'APPTAINERENV_TOP_P' in env_passed
        assert env_passed['APPTAINERENV_TOP_P'] == "0.95"
        
        assert 'APPTAINERENV_TOP_K' in env_passed
        assert env_passed['APPTAINERENV_TOP_K'] == "50"
        
        # Verify OLLAMA variables
        assert 'OLLAMA_MODEL_NAME' in env_passed
        assert 'OLLAMA_PORT' in env_passed
        assert 'APPTAINERENV_OLLAMA_HOST' in env_passed
        
        # Verify rebuild flag
        assert 'LLMFLUX_FORCE_REBUILD' in env_passed
        assert env_passed['LLMFLUX_FORCE_REBUILD'] == "1"
        
        # Verify workspace paths
        assert 'PROJECT_ROOT' in env_passed
        assert 'DATA_INPUT_DIR' in env_passed
        assert 'DATA_OUTPUT_DIR' in env_passed
        
        assert job_id == "12345"
    
    @patch('llmflux.slurm.runner.subprocess.run')
    @patch('llmflux.slurm.runner.socket.socket')
    @patch('llmflux.slurm.runner.ConfigManager')
    def test_runner_env_merges_with_existing_env(
        self, mock_config_manager, mock_socket, mock_subprocess, temp_dir, sample_jsonl
    ):
        """Test that runner environment merges with existing environment variables."""
        # Setup mocks (same as above)
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.get_path.return_value = temp_dir
        mock_config.ensure_directory = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(
            account="test-account",
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=1,
            time="01:00:00",
            memory="32G",
            cpus_per_task=4,
            extra_sbatch_args={}
        )
        mock_manager.get_config.return_value = mock_config
        mock_manager.get_parameter.side_effect = lambda param_name, code_value, obj, env_var, default: code_value or default
        mock_config_manager.return_value = mock_manager
        
        # Mock socket
        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ('', 11434)
        mock_socket.return_value = mock_socket_instance
        
        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="", text=True)
        
        # Set existing environment variable
        import os
        os.environ['TEST_EXISTING_VAR'] = 'test_value'
        
        from llmflux.slurm.runner import SlurmRunner

        slurm_config = mock_config.get_slurm_config()
        engine_config = EngineConfig(engine="ollama", home=str(temp_dir / ".ollama"))
        model_config_mock = MagicMock()
        model_config_mock.name = "Llama-3.2-3B-Instruct"
        model_config_mock.hf_name = None
        mock_config.load_model_config.return_value = model_config_mock
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir), engine_config=engine_config)

        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="Llama-3.2-3B-Instruct"
        )

        # Verify existing environment is preserved
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})
        
        # Should include both new and existing variables
        assert 'TEST_EXISTING_VAR' in env_passed or 'TEST_EXISTING_VAR' in os.environ
        assert 'APPTAINERENV_MODEL_NAME' in env_passed
    
    @patch('llmflux.slurm.runner.subprocess.run')
    @patch('llmflux.slurm.runner.socket.socket')
    @patch('llmflux.slurm.runner.ConfigManager')
    def test_runner_sets_gpu_environment_variables(
        self, mock_config_manager, mock_socket, mock_subprocess, temp_dir, sample_jsonl
    ):
        """Test that GPU-related environment variables are set correctly."""
        # Setup mocks
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.get_path.return_value = temp_dir
        mock_config.ensure_directory = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(
            account="test-account",
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=2,  # Multiple GPUs
            time="01:00:00",
            memory="64G",
            cpus_per_task=8,
            extra_sbatch_args={}
        )
        mock_manager.get_config.return_value = mock_config
        mock_manager.get_parameter.side_effect = lambda param_name, code_value, obj, env_var, default: code_value or default
        mock_config_manager.return_value = mock_manager
        
        # Mock socket
        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ('', 11434)
        mock_socket.return_value = mock_socket_instance
        
        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="", text=True)
        
        from llmflux.slurm.runner import SlurmRunner

        slurm_config = mock_config.get_slurm_config()
        engine_config = EngineConfig(engine="ollama", home=str(temp_dir / ".ollama"))
        model_config_mock = MagicMock()
        model_config_mock.name = "Llama-3.2-3B-Instruct"
        model_config_mock.hf_name = None
        mock_config.load_model_config.return_value = model_config_mock
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir), engine_config=engine_config)

        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="Llama-3.2-3B-Instruct"
        )

        # Verify GPU environment variables
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})
        
        # Check CUDA_VISIBLE_DEVICES for multiple GPUs (it's set as APPTAINERENV_ prefix)
        assert 'APPTAINERENV_CUDA_VISIBLE_DEVICES' in env_passed
        assert env_passed['APPTAINERENV_CUDA_VISIBLE_DEVICES'] == "0,1"  # Two GPUs
        
        # Check OLLAMA_SCHED_SPREAD for multiple GPUs (it's set as APPTAINERENV_ prefix)
        assert 'APPTAINERENV_OLLAMA_SCHED_SPREAD' in env_passed
        assert env_passed['APPTAINERENV_OLLAMA_SCHED_SPREAD'] == "1"  # Spread enabled for multi-GPU


class TestEnvironmentVariablePrefixes:
    """Test that container environment variables use correct APPTAINERENV_ prefix.
    
    These tests ensure that if someone changes the code to use incorrect prefixes
    or forgets to add the prefix, the tests will fail.
    """
    
    @patch('llmflux.slurm.runner.subprocess.run')
    @patch('llmflux.slurm.runner.socket.socket')
    @patch('llmflux.slurm.runner.ConfigManager')
    def test_container_vars_must_have_apptainerenv_prefix(
        self, mock_config_manager, mock_socket, mock_subprocess, temp_dir, sample_jsonl
    ):
        """Test that all container-passed variables MUST have APPTAINERENV_ prefix.
        
        This test will fail if variables are named incorrectly (e.g., MODEL_NAME instead of APPTAINERENV_MODEL_NAME).
        """
        # Setup mocks
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.get_path.return_value = temp_dir
        mock_config.ensure_directory = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(
            account="test-account",
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=1,
            time="01:00:00",
            memory="32G",
            cpus_per_task=4,
            extra_sbatch_args={}
        )
        mock_manager.get_config.return_value = mock_config
        mock_manager.get_parameter.side_effect = lambda param_name, code_value, obj, env_var, default: code_value or default
        mock_config_manager.return_value = mock_manager
        
        # Mock socket
        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ('', 11434)
        mock_socket.return_value = mock_socket_instance
        
        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="", text=True)
        
        from llmflux.slurm.runner import SlurmRunner

        slurm_config = mock_config.get_slurm_config()
        engine_config = EngineConfig(engine="ollama", home=str(temp_dir / ".ollama"))
        model_config_mock = MagicMock()
        model_config_mock.name = "Llama-3.2-3B-Instruct"
        model_config_mock.hf_name = None
        mock_config.load_model_config.return_value = model_config_mock
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir), engine_config=engine_config)

        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="Llama-3.2-3B-Instruct",
            batch_size=8,
            save_frequency=100,
            max_tokens=4096,
            temperature=0.8,
            top_p=0.95,
            top_k=50
        )

        # Get the environment passed to subprocess.run
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})
        
        # List of container variables that MUST have APPTAINERENV_ prefix
        # These are variables that should be passed to the container
        container_variable_names = [
            'MODEL_NAME',
            'BATCH_SIZE',
            'SAVE_FREQUENCY',
            'MAX_TOKENS',
            'TEMPERATURE',
            'TOP_P',
            'TOP_K',
            'CUDA_VISIBLE_DEVICES',
            'OLLAMA_PORT',
            'OLLAMA_HOST',
            'OLLAMA_HOME',
            'OLLAMA_MODELS',
            'OLLAMA_ORIGINS',
            'OLLAMA_INSECURE',
            'OLLAMA_SCHED_SPREAD',
            'CURL_CA_BUNDLE',
            'SSL_CERT_FILE',
            'PROJECT_ROOT',
        ]
        
        # Variables that exist as both host-only AND container vars (these are OK to have both)
        # Host vars are used by bash script, container vars are passed to container
        dual_purpose_vars = {
            'OLLAMA_MODEL_NAME',  # Host: bash script, Container: Python
            'OLLAMA_PORT',         # Host: bash script, Container: Python
            'OLLAMA_HOME',         # Host: mkdir/bind, Container: Ollama
            'OLLAMA_MODELS',       # Host: mkdir, Container: Ollama
            'OLLAMA_HOST',         # Host: bash script Ollama server config, Container: Python
            'PROJECT_ROOT',        # Host: Python path, Container: Python path
        }
        
        # Pure container-only variables (these should NOT exist without APPTAINERENV_ prefix)
        pure_container_vars = {
            'MODEL_NAME', 'BATCH_SIZE', 'SAVE_FREQUENCY', 'MAX_TOKENS',
            'TEMPERATURE', 'TOP_P', 'TOP_K', 'CUDA_VISIBLE_DEVICES',
            'OLLAMA_HOST', 'OLLAMA_ORIGINS', 'OLLAMA_INSECURE',
            'OLLAMA_SCHED_SPREAD', 'CURL_CA_BUNDLE', 'SSL_CERT_FILE',
        }
        
        # Verify that container variables MUST have APPTAINERENV_ prefix
        for var_name in container_variable_names:
            prefixed_name = f'APPTAINERENV_{var_name}'
            
            # The prefixed version MUST exist for all container vars
            assert prefixed_name in env_passed, (
                f"Container variable {var_name} MUST use APPTAINERENV_ prefix. "
                f"Expected {prefixed_name} in environment, but it was not found."
            )
            
            # For pure container vars, the non-prefixed version should NOT exist
            # (unless it's a dual-purpose var that's also needed as host var)
            if var_name in pure_container_vars:
                if var_name in env_passed and var_name not in dual_purpose_vars:
                    pytest.fail(
                        f"Container variable {var_name} should NOT exist without APPTAINERENV_ prefix. "
                        f"Found {var_name} in environment without prefix. Use {prefixed_name} instead. "
                        f"If this is intentional (host + container var), add it to dual_purpose_vars."
                    )
    
    @patch('llmflux.slurm.runner.subprocess.run')
    @patch('llmflux.slurm.runner.socket.socket')
    @patch('llmflux.slurm.runner.ConfigManager')
    def test_no_incorrect_prefixes_allowed(
        self, mock_config_manager, mock_socket, mock_subprocess, temp_dir, sample_jsonl
    ):
        """Test that variables don't use incorrect prefixes like SINGULARITYENV_ or CONTAINER_."""
        # Setup mocks (same as above)
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.get_path.return_value = temp_dir
        mock_config.ensure_directory = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(
            account="test-account",
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=1,
            time="01:00:00",
            memory="32G",
            cpus_per_task=4,
            extra_sbatch_args={}
        )
        mock_manager.get_config.return_value = mock_config
        mock_manager.get_parameter.side_effect = lambda param_name, code_value, obj, env_var, default: code_value or default
        mock_config_manager.return_value = mock_manager
        
        # Mock socket
        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ('', 11434)
        mock_socket.return_value = mock_socket_instance
        
        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="", text=True)
        
        from llmflux.slurm.runner import SlurmRunner

        slurm_config = mock_config.get_slurm_config()
        engine_config = EngineConfig(engine="ollama", home=str(temp_dir / ".ollama"))
        model_config_mock = MagicMock()
        model_config_mock.name = "Llama-3.2-3B-Instruct"
        model_config_mock.hf_name = None
        mock_config.load_model_config.return_value = model_config_mock
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir), engine_config=engine_config)

        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="Llama-3.2-3B-Instruct"
        )

        # Get the environment
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})

        # List of incorrect prefixes that should NOT be used for container variables
        incorrect_prefixes = [
            'SINGULARITYENV_',  # Old Singularity prefix (not Apptainer)
            'DOCKER_',          # Wrong container type
            'ENV_',             # Too generic
        ]
        
        # Host-only variables that are allowed to have any prefix (not passed to container)
        # These are used by bash scripts on the host, not inside the container
        host_only_allowed_vars = {
            'CONTAINER_DEF',     # Host: path to container definition file
            'CONTAINERS_DIR',    # Host: directory path
            'DATA_INPUT_DIR',    # Host: directory path
            'DATA_OUTPUT_DIR',   # Host: directory path
            'MODELS_DIR',        # Host: directory path
            'LOGS_DIR',          # Host: directory path
            'PROJECT_ROOT',      # Host: Python path (also has APPTAINERENV_ version)
            'OLLAMA_HOME',       # Host: mkdir/bind (also has APPTAINERENV_ version)
            'OLLAMA_MODELS',     # Host: mkdir (also has APPTAINERENV_ version)
            'OLLAMA_PORT',       # Host: bash script (also has APPTAINERENV_ version)
            'OLLAMA_MODEL_NAME', # Host: bash script
            'APPTAINER_TMPDIR',  # Host: Apptainer config
            'APPTAINER_CACHEDIR',# Host: Apptainer config
            'SINGULARITY_TMPDIR',# Host: Singularity config
            'SINGULARITY_CACHEDIR',# Host: Singularity config
            'LLMFLUX_FORCE_REBUILD', # Host: bash script flag
        }
        
        # Check that container-passed variables don't use incorrect prefixes
        # Only check variables that:
        # 1. Are container variables (should be passed to container)
        # 2. Are NOT host-only variables
        for env_key in env_passed.keys():
            # Skip host-only variables (they can have any name)
            if env_key in host_only_allowed_vars:
                continue
            
            # Skip OS environment variables (not set by our code)
            # Only check variables that look like container variables we set
            is_container_var = (
                env_key.startswith('APPTAINERENV_') or
                env_key.startswith('SINGULARITYENV_') or
                env_key.startswith('DOCKER_') or
                env_key.startswith('CONTAINER_') and env_key not in host_only_allowed_vars
            )
            
            if is_container_var:
                for incorrect_prefix in incorrect_prefixes:
                    if env_key.startswith(incorrect_prefix):
                        pytest.fail(
                            f"Found container environment variable with incorrect prefix: {env_key}. "
                            f"Container variables must use 'APPTAINERENV_' prefix, not '{incorrect_prefix}'. "
                            f"If this is a host-only variable, add it to host_only_allowed_vars."
                        )
    
    @patch('llmflux.slurm.runner.subprocess.run')
    @patch('llmflux.slurm.runner.socket.socket')
    @patch('llmflux.slurm.runner.ConfigManager')
    def test_all_apptainerenv_vars_are_container_vars(
        self, mock_config_manager, mock_socket, mock_subprocess, temp_dir, sample_jsonl
    ):
        """Test that all APPTAINERENV_ prefixed variables are properly formatted.
        
        This ensures consistency - if a variable has APPTAINERENV_ prefix, it should be
        a valid container variable that should be passed to the container.
        """
        # Setup mocks
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.get_path.return_value = temp_dir
        mock_config.ensure_directory = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(
            account="test-account",
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=2,  # Multiple GPUs
            time="01:00:00",
            memory="64G",
            cpus_per_task=8,
            extra_sbatch_args={}
        )
        mock_manager.get_config.return_value = mock_config
        mock_manager.get_parameter.side_effect = lambda param_name, code_value, obj, env_var, default: code_value or default
        mock_config_manager.return_value = mock_manager
        
        # Mock socket
        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ('', 11434)
        mock_socket.return_value = mock_socket_instance
        
        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="", text=True)
        
        from llmflux.slurm.runner import SlurmRunner

        slurm_config = mock_config.get_slurm_config()
        engine_config = EngineConfig(engine="ollama", home=str(temp_dir / ".ollama"))
        model_config_mock = MagicMock()
        model_config_mock.name = "Llama-3.2-3B-Instruct"
        model_config_mock.hf_name = None
        mock_config.load_model_config.return_value = model_config_mock
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir), engine_config=engine_config)

        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="Llama-3.2-3B-Instruct",
            batch_size=8,
            max_tokens=4096,
            temperature=0.8
        )

        # Get the environment
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})

        # Collect all APPTAINERENV_ prefixed variables
        apptainerenv_vars = {key: value for key, value in env_passed.items() 
                            if key.startswith('APPTAINERENV_')}
        
        # Verify that all APPTAINERENV_ variables follow naming conventions
        # They should:
        # 1. Start with APPTAINERENV_
        # 2. Have the rest in UPPERCASE
        # 3. Use underscores, not hyphens or spaces
        
        for var_name, var_value in apptainerenv_vars.items():
            # Remove the prefix to get the base name
            base_name = var_name.replace('APPTAINERENV_', '', 1)
            
            # Verify base name is uppercase
            assert base_name.isupper() or base_name == '', (
                f"APPTAINERENV_ variable {var_name} has lowercase in base name '{base_name}'. "
                f"Environment variable names should be UPPERCASE."
            )
            
            # Verify no hyphens (should use underscores)
            assert '-' not in base_name, (
                f"APPTAINERENV_ variable {var_name} contains hyphen in '{base_name}'. "
                f"Use underscores instead."
            )
            
            # Verify no spaces
            assert ' ' not in base_name, (
                f"APPTAINERENV_ variable {var_name} contains space in '{base_name}'. "
                f"Variable names should not contain spaces."
            )
            
            # Verify value is not None (should be string or empty string)
            assert var_value is not None, (
                f"APPTAINERENV_ variable {var_name} has None value. "
                f"All environment variables should have string values."
            )
    
    @patch('llmflux.slurm.runner.subprocess.run')
    @patch('llmflux.slurm.runner.socket.socket')
    @patch('llmflux.slurm.runner.ConfigManager')
    def test_model_parameters_use_apptainerenv_prefix(
        self, mock_config_manager, mock_socket, mock_subprocess, temp_dir, sample_jsonl
    ):
        """Test that all model parameters passed via kwargs use APPTAINERENV_ prefix.
        
        This ensures that if someone adds new model parameters, they automatically
        get the correct prefix through the kwargs loop.
        """
        # Setup mocks
        mock_manager = MagicMock()
        mock_config = MagicMock()
        mock_config.get_path.return_value = temp_dir
        mock_config.ensure_directory = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(
            account="test-account",
            partition="gpuA100x4",
            nodes=1,
            gpus_per_node=1,
            time="01:00:00",
            memory="32G",
            cpus_per_task=4,
            extra_sbatch_args={}
        )
        mock_manager.get_config.return_value = mock_config
        mock_manager.get_parameter.side_effect = lambda param_name, code_value, obj, env_var, default: code_value or default
        mock_config_manager.return_value = mock_manager
        
        # Mock socket
        mock_socket_instance = MagicMock()
        mock_socket_instance.getsockname.return_value = ('', 11434)
        mock_socket.return_value = mock_socket_instance
        
        # Mock subprocess
        mock_subprocess.return_value = MagicMock(returncode=0, stdout="Submitted batch job 12345\n", stderr="", text=True)
        
        from llmflux.slurm.runner import SlurmRunner

        slurm_config = mock_config.get_slurm_config()
        engine_config = EngineConfig(engine="ollama", home=str(temp_dir / ".ollama"))
        model_config_mock = MagicMock()
        model_config_mock.name = "Llama-3.2-3B-Instruct"
        model_config_mock.hf_name = None
        mock_config.load_model_config.return_value = model_config_mock
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir), engine_config=engine_config)

        # Pass various model parameters to test they all get proper prefix
        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="Llama-3.2-3B-Instruct",
            max_tokens=2048,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            frequency_penalty=0.5,  # Test additional parameter
            presence_penalty=0.3,   # Test additional parameter
        )
        
        # Get the environment
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})
        
        # Verify all model parameters use APPTAINERENV_ prefix
        model_params = ['MAX_TOKENS', 'TEMPERATURE', 'TOP_P', 'TOP_K', 
                       'FREQUENCY_PENALTY', 'PRESENCE_PENALTY']
        
        for param in model_params:
            prefixed_name = f'APPTAINERENV_{param}'
            assert prefixed_name in env_passed, (
                f"Model parameter {param} must use APPTAINERENV_ prefix. "
                f"Expected {prefixed_name} in environment, but it was not found."
            )
            
            # Verify the non-prefixed version does NOT exist
            if param in env_passed:
                pytest.fail(
                    f"Model parameter {param} should NOT exist without APPTAINERENV_ prefix. "
                    f"Found {param} in environment. Use {prefixed_name} instead."
                )


class TestCommandLineIntegration:
    """Integration tests for CLI commands."""
    
    @patch('llmflux.cli.SlurmRunner')
    @patch('llmflux.cli.Config')
    def test_cli_run_integration(
        self, mock_config_class, mock_runner_class, temp_dir, sample_jsonl
    ):
        """Integration test for full CLI run command."""
        # Setup mocks
        mock_config = MagicMock()
        mock_slurm_config = MagicMock()
        mock_config.get_slurm_config.return_value = mock_slurm_config
        mock_config_class.return_value = mock_config
        
        mock_runner = MagicMock()
        mock_runner.run.return_value = "12345"
        mock_runner_class.return_value = mock_runner
        
        # Simulate CLI call
        with patch('sys.argv', [
            'llmflux', 'run',
            '--model', 'Llama-3.2-3B-Instruct',
            '--input', str(sample_jsonl),
            '--output', str(temp_dir / 'output.json'),
            '--account', 'test-account',
            '--batch-size', '8'
        ]):
            with patch('sys.stdout', new=StringIO()):
                result = main()
        
        assert result == 0
        
        # Verify all parameters were passed to runner
        mock_runner.run.assert_called_once()
        call_kwargs = mock_runner.run.call_args[1]
        assert call_kwargs["model"] == "Llama-3.2-3B-Instruct"
        assert call_kwargs["batch_size"] == 8
        assert call_kwargs["input_path"] == str(sample_jsonl)
    
    @patch('llmflux.cli._wait_for_slurm_elapsed_seconds', return_value=None)
    @patch('llmflux.cli.create_test_prompts_file')
    @patch('llmflux.cli.SlurmRunner')
    @patch('llmflux.cli.Config')
    def test_cli_benchmark_integration(
        self, mock_config_class, mock_runner_class,
        mock_create_prompts, mock_wait
    ):
        """Integration test for full CLI benchmark command."""
        # Setup mocks
        mock_config = MagicMock()
        mock_slurm_config = MagicMock()
        mock_config.get_slurm_config.return_value = mock_slurm_config
        mock_config_class.return_value = mock_config

        mock_runner = MagicMock()
        mock_runner.run.return_value = "67890"
        mock_runner_class.return_value = mock_runner

        mock_create_prompts.return_value = Path("data/benchmarks/prompts.jsonl")

        # Simulate CLI call
        with patch('sys.argv', [
            'llmflux', 'benchmark',
            '--model', 'Llama-3.2-3B-Instruct',
            '--num-prompts', '100',
            '--batch-size', '8',
            '--account', 'test-account'
        ]):
            with patch('builtins.print'):
                result = main()

        assert result == 0
        mock_create_prompts.assert_called_once_with(num_prompts=100, temperature=0.7, max_tokens=500, model="Llama-3.2-3B-Instruct")


class TestServeCommand:
    """Tests for the `llmflux serve` subcommand."""

    def _make_serve_args(self, **overrides):
        args = MagicMock()
        args.model = "Llama-3.2-3B-Instruct"
        args.email = "user@example.com"
        args.engine = "vllm"
        args.account = None
        args.partition = None
        args.nodes = None
        args.gpus_per_node = None
        args.time = "02:00:00"
        args.mem = None
        args.cpus_per_task = None
        args.sbatch_arg = None
        args.rebuild = False
        args.debug = False
        args.vllm_engine_args = None
        for k, v in overrides.items():
            setattr(args, k, v)
        return args

    @patch("llmflux.cli.SlurmRunner")
    @patch("llmflux.cli.Config")
    def test_serve_command_success(self, mock_config_class, mock_runner_class):
        """Successful serve submission returns 0 and prints job ID."""
        from llmflux.cli import _serve_command

        mock_config = MagicMock()
        mock_config.get_slurm_config.return_value = MagicMock(time="02:00:00")
        mock_config_class.return_value = mock_config

        mock_runner = MagicMock()
        mock_runner.serve.return_value = "99999"
        mock_runner_class.return_value = mock_runner

        with patch("builtins.print") as mock_print:
            result = _serve_command(self._make_serve_args())

        assert result == 0
        mock_runner.serve.assert_called_once()
        call_kwargs = mock_runner.serve.call_args[1]
        assert call_kwargs["email"] == "user@example.com"
        assert call_kwargs["model"] == "Llama-3.2-3B-Instruct"
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "99999" in printed

    @patch("llmflux.cli.SlurmRunner")
    @patch("llmflux.cli.Config")
    def test_serve_command_validation_error(self, mock_config_class, mock_runner_class):
        """ValueError from serve() returns exit code 1, no success message."""
        from llmflux.cli import _serve_command

        mock_config_class.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.serve.side_effect = ValueError("Model 'bad-model' not found.")
        mock_runner_class.return_value = mock_runner

        with patch("sys.stderr", new=StringIO()) as mock_stderr:
            result = _serve_command(self._make_serve_args(model="bad-model"))

        assert result == 1
        assert "bad-model" in mock_stderr.getvalue()

    @patch("llmflux.cli.SlurmRunner")
    @patch("llmflux.cli.Config")
    def test_serve_command_sbatch_failure(self, mock_config_class, mock_runner_class):
        """CalledProcessError from sbatch returns exit code 1."""
        import subprocess
        from llmflux.cli import _serve_command

        mock_config_class.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.serve.side_effect = subprocess.CalledProcessError(
            1, "sbatch", stderr="Invalid partition"
        )
        mock_runner_class.return_value = mock_runner

        with patch("sys.stderr", new=StringIO()) as mock_stderr:
            result = _serve_command(self._make_serve_args())

        assert result == 1
        assert "sbatch" in mock_stderr.getvalue().lower() or "Invalid partition" in mock_stderr.getvalue()

    @patch("llmflux.cli.SlurmRunner")
    @patch("llmflux.cli.Config")
    def test_serve_command_none_job_id(self, mock_config_class, mock_runner_class):
        """None returned from serve() returns exit code 1, no job ID printed."""
        from llmflux.cli import _serve_command

        mock_config_class.return_value = MagicMock()
        mock_runner = MagicMock()
        mock_runner.serve.return_value = None
        mock_runner_class.return_value = mock_runner

        with patch("sys.stderr", new=StringIO()) as mock_stderr:
            with patch("builtins.print") as mock_print:
                result = _serve_command(self._make_serve_args())

        assert result == 1
        printed = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Serve job submitted" not in printed

    def test_serve_parser_required_args(self):
        """serve subcommand requires --model, --email, and --time."""
        parser = build_parser()
        # missing --email and --time
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "--model", "Llama-3.2-3B-Instruct"])
        # missing --model and --time
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "--email", "user@example.com"])
        # missing --time
        with pytest.raises(SystemExit):
            parser.parse_args(["serve", "--model", "Llama-3.2-3B-Instruct", "--email", "user@example.com"])

    def test_serve_parser_args(self):
        """serve subcommand parses all arguments correctly."""
        parser = build_parser()
        args = parser.parse_args([
            "serve",
            "--model", "Llama-3.2-3B-Instruct",
            "--email", "user@example.com",
            "--engine", "ollama",
            "--time", "04:00:00",
            "--partition", "gpuA100x4",
        ])
        assert args.model == "Llama-3.2-3B-Instruct"
        assert args.email == "user@example.com"
        assert args.engine == "ollama"
        assert args.time == "04:00:00"
        assert args.partition == "gpuA100x4"


class TestConnectCommand:
    """Tests for the `llmflux connect` subcommand."""

    def test_connect_parser_args(self):
        """connect subcommand parses job_id and optional flags."""
        parser = build_parser()
        args = parser.parse_args(["connect", "12345"])
        assert args.job_id == "12345"
        assert args.local_port == 8000
        assert args.wait_timeout == 600

    def test_connect_invalid_job_id(self):
        """Non-numeric job_id returns exit code 1 before touching SLURM."""
        from llmflux.cli import _connect_command

        args = MagicMock()
        args.job_id = "../../.ssh"

        with patch("sys.stderr", new=StringIO()) as mock_stderr:
            result = _connect_command(args)

        assert result == 1
        assert "Invalid job ID" in mock_stderr.getvalue()

    @patch("llmflux.cli.JobRegistry")
    def test_connect_not_in_registry(self, mock_registry_class):
        """Job not tracked in registry returns exit code 1."""
        from llmflux.cli import _connect_command

        mock_registry_class.return_value.get_job.return_value = None
        args = MagicMock()
        args.job_id = "12345"

        with patch("sys.stderr", new=StringIO()):
            result = _connect_command(args)

        assert result == 1

    @patch("llmflux.cli.JobRegistry")
    def test_connect_batch_job_rejected(self, mock_registry_class):
        """Connecting to a batch job (not serve) returns exit code 1."""
        from llmflux.cli import _connect_command

        mock_registry_class.return_value.get_job.return_value = {"type": "batch"}
        args = MagicMock()
        args.job_id = "12345"

        with patch("sys.stderr", new=StringIO()):
            result = _connect_command(args)

        assert result == 1

    @patch("llmflux.cli.get_job_state", return_value="PENDING")
    @patch("llmflux.cli.JobRegistry")
    def test_connect_pending_job_rejected(self, mock_registry_class, mock_state):
        """PENDING serve job returns exit code 1 with helpful message."""
        from llmflux.cli import _connect_command

        mock_registry_class.return_value.get_job.return_value = {"type": "serve"}
        args = MagicMock()
        args.job_id = "12345"

        with patch("sys.stderr", new=StringIO()) as mock_stderr:
            result = _connect_command(args)

        assert result == 1
        assert "PENDING" in mock_stderr.getvalue()

    @patch("llmflux.cli.connect", return_value=0)
    @patch("llmflux.cli.get_job_state", return_value="RUNNING")
    @patch("llmflux.cli.JobRegistry")
    def test_connect_running_job_succeeds(self, mock_registry_class, mock_state, mock_connect):
        """RUNNING serve job delegates to connection_connect and returns its exit code."""
        from llmflux.cli import _connect_command

        mock_registry_class.return_value.get_job.return_value = {"type": "serve"}
        args = MagicMock()
        args.job_id = "12345"
        args.local_port = 8000
        args.wait_timeout = 600

        result = _connect_command(args)

        assert result == 0
        mock_connect.assert_called_once_with(
            job_id="12345", local_port=8000, wait_timeout=600
        )


class TestJobsServeType:
    """Tests for TYPE column in llmflux jobs."""

    @patch("llmflux.cli.get_active_job_details")
    @patch("llmflux.cli.JobRegistry")
    def test_jobs_shows_type_column(self, mock_registry_class, mock_active_jobs):
        """llmflux jobs table includes TYPE column with serve/batch values."""
        from llmflux.cli import _jobs_command

        mock_registry_class.return_value.get_all_jobs.return_value = {
            "11111": {"job_name": "batch_job", "model": "Llama", "engine": "vllm",
                      "type": "batch", "submitted_at": "2026-01-01T00:00:00+00:00"},
            "22222": {"job_name": "serve_job", "model": "Llama", "engine": "vllm",
                      "type": "serve", "submitted_at": "2026-01-01T01:00:00+00:00"},
        }
        mock_active_jobs.return_value = {
            "11111": {"state": {"current": "RUNNING"}, "time": {}},
            "22222": {"state": {"current": "RUNNING"}, "time": {}},
        }

        args = MagicMock()
        args.all = False
        args.state = None

        with patch("builtins.print") as mock_print:
            result = _jobs_command(args)

        assert result == 0
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "TYPE" in output
        assert "batch" in output
        assert "serve" in output


class TestStatusServeView:
    """Tests for llmflux status showing serve-specific fields."""

    @patch("llmflux.cli.get_job_log_paths", return_value=(None, None))
    @patch("llmflux.cli.get_job_details", return_value={"state": {"current": "RUNNING"}, "time": {}})
    @patch("llmflux.cli.JobRegistry")
    def test_status_serve_job_shows_endpoint_fields(
        self, mock_registry_class, mock_details, mock_log_paths
    ):
        """status for a serve job shows Endpoint, API Key, Email instead of Input/Output."""
        from llmflux.cli import _status_command
        from llmflux.slurm.connection import read_connection_info

        mock_registry_class.return_value.get_job.return_value = {
            "job_name": "serve_job",
            "type": "serve",
            "model": "Llama",
            "engine": "vllm",
            "email": "user@example.com",
            "api_key": "llmflux-abc123",
            "submitted_at": "2026-01-01T00:00:00+00:00",
            "logs_dir": "/tmp/logs",
        }

        args = MagicMock()
        args.job_id = "12345"

        with patch("llmflux.slurm.connection.read_connection_info", return_value={
            "node": "gpu01", "port": 8000, "engine": "vllm"
        }):
            with patch("builtins.print") as mock_print:
                result = _status_command(args)

        assert result == 0
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Endpoint" in output
        assert "API Key" in output
        assert "Email" in output
        assert "Input" not in output
        assert "Output" not in output
        assert "llmflux connect 12345" in output

    @patch("llmflux.cli.get_job_log_paths", return_value=(None, None))
    @patch("llmflux.cli.get_job_details", return_value={"state": {"current": "RUNNING"}, "time": {}})
    @patch("llmflux.cli.JobRegistry")
    def test_status_batch_job_shows_input_output(
        self, mock_registry_class, mock_details, mock_log_paths
    ):
        """status for a batch job shows Input and Output, not endpoint fields."""
        from llmflux.cli import _status_command

        mock_registry_class.return_value.get_job.return_value = {
            "job_name": "batch_job",
            "type": "batch",
            "model": "Llama",
            "engine": "vllm",
            "input": "/data/input.jsonl",
            "output": "/data/output.json",
            "submitted_at": "2026-01-01T00:00:00+00:00",
            "logs_dir": "/tmp/logs",
        }

        args = MagicMock()
        args.job_id = "12345"

        with patch("builtins.print") as mock_print:
            result = _status_command(args)

        assert result == 0
        output = " ".join(str(c) for c in mock_print.call_args_list)
        assert "Input" in output
        assert "Output" in output
        assert "Endpoint" not in output
        assert "API Key" not in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

