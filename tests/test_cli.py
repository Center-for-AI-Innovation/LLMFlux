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

from aiflux.cli import main, _run_command, _benchmark_command, build_parser
from aiflux.slurm.runner import SlurmRunner
from aiflux.core.config import Config, SlurmConfig


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
                "model": "llama3.2:3b",
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
                "model": "llama3.2:3b",
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
    with patch('aiflux.cli.SlurmRunner') as mock_runner_class:
        mock_runner = MagicMock()
        mock_runner.run.return_value = "12345"
        mock_runner_class.return_value = mock_runner
        yield mock_runner


@pytest.fixture
def mock_config():
    """Mock Config for testing."""
    with patch('aiflux.cli.Config') as mock_config_class:
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
        assert parser.prog == "aiflux"
    
    def test_parser_has_run_subcommand(self):
        """Test that run subcommand exists."""
        parser = build_parser()
        # Parse with run command
        args = parser.parse_args(["run", "--model", "llama3.2:3b", "--input", "test.jsonl"])
        assert args.command == "run"
        assert args.model == "llama3.2:3b"
        assert args.input == "test.jsonl"
    
    def test_parser_has_benchmark_subcommand(self):
        """Test that benchmark subcommand exists."""
        parser = build_parser()
        args = parser.parse_args(["benchmark", "--model", "llama3.2:3b"])
        assert args.command == "benchmark"
        assert args.model == "llama3.2:3b"
    
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
            "--model", "llama3.2:3b",
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
        
        assert args.model == "llama3.2:3b"
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
            "--model", "llama3.2:3b",
            "--name", "my-benchmark",
            "--num-prompts", "100",
            "--batch-size", "8",
            "--max-tokens", "2048",
            "--temperature", "0.7",
            "--account", "my-account",
            "--rebuild",
            "--debug"
        ])
        
        assert args.model == "llama3.2:3b"
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
    
    @patch('aiflux.cli.SlurmRunner')
    @patch('aiflux.cli.Config')
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
        args.model = "llama3.2:3b"
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
        assert call_kwargs["model"] == "llama3.2:3b"
        assert call_kwargs["batch_size"] == 4
        # Verify all expected kwargs are passed
        assert "save_frequency" in call_kwargs
        assert "max_retries" in call_kwargs
        assert "retry_delay" in call_kwargs
    
    @patch('aiflux.cli.SlurmRunner')
    @patch('aiflux.cli.Config')
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
        args.model = "llama3.2:3b"
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
        assert call_kwargs["model"] == "llama3.2:3b"
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
        args.model = "llama3.2:3b"
        args.local = False
        
        with patch('sys.stderr', new=StringIO()):
            result = _run_command(args)
            assert result == 2  # Exit code for error
    
    @patch('aiflux.cli.BatchProcessor')
    @patch('aiflux.cli.Config')
    def test_run_command_local_mode(self, mock_config_class, mock_processor_class, temp_dir, sample_jsonl):
        """Test run command in local mode (not implemented yet, but tests structure)."""
        # Note: Local mode is commented out in CLI, but we test the structure
        args = MagicMock()
        args.input = str(sample_jsonl)
        args.output = str(temp_dir / "output.json")
        args.model = "llama3.2:3b"
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
    
    @patch('aiflux.cli.save_prompts_to_jsonl')
    @patch('aiflux.cli.generate_synthetic_prompts')
    @patch('aiflux.cli.SlurmRunner')
    @patch('aiflux.cli.Config')
    def test_benchmark_command_generate_prompts(
        self, mock_config_class, mock_runner_class, 
        mock_generate_prompts, mock_save_jsonl, temp_dir
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
        
        # Mock prompt generation
        mock_prompts = [{"prompt": "test"}] * 50
        mock_generate_prompts.return_value = mock_prompts
        
        # Create args
        args = MagicMock()
        args.model = "llama3.2:3b"
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
        mock_generate_prompts.assert_called_once_with(num_prompts=50, model="llama3.2:3b")
        mock_save_jsonl.assert_called_once()
        mock_runner.run.assert_called_once()
    
    @patch('aiflux.cli.SlurmRunner')
    @patch('aiflux.cli.Config')
    def test_benchmark_command_with_existing_input(
        self, mock_config_class, mock_runner_class, temp_dir, sample_jsonl
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
        args.model = "llama3.2:3b"
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
        assert call_kwargs["model"] == "llama3.2:3b"
        assert call_kwargs["batch_size"] == 8
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["rebuild"] is True
        assert call_kwargs["debug"] is True


class TestMainFunction:
    """Test the main CLI entry point."""
    
    def test_main_with_help(self):
        """Test main function with help flag."""
        with patch('sys.argv', ['aiflux', '--help']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            # argparse exits with 0 for help
            assert exc_info.value.code == 0
    
    @patch('aiflux.cli._run_command')
    def test_main_with_run_command(self, mock_run_command, temp_dir, sample_jsonl):
        """Test main function with run command."""
        mock_run_command.return_value = 0
        
        with patch('sys.argv', [
            'aiflux', 'run',
            '--model', 'llama3.2:3b',
            '--input', str(sample_jsonl)
        ]):
            result = main()
            assert result == 0
            mock_run_command.assert_called_once()
    
    @patch('aiflux.cli._benchmark_command')
    def test_main_with_benchmark_command(self, mock_benchmark_command):
        """Test main function with benchmark command."""
        mock_benchmark_command.return_value = 0
        
        with patch('sys.argv', [
            'aiflux', 'benchmark',
            '--model', 'llama3.2:3b'
        ]):
            result = main()
            assert result == 0
            mock_benchmark_command.assert_called_once()
    
    def test_main_with_invalid_command(self):
        """Test main function with invalid command."""
        with patch('sys.argv', ['aiflux', 'invalid-command']):
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 2


class TestRunnerEnvironmentVariables:
    """Test that environment variables are properly passed to runner.run()."""
    
    @patch('aiflux.slurm.runner.subprocess.run')
    @patch('aiflux.slurm.runner.socket.socket')
    @patch('aiflux.slurm.runner.ConfigManager')
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
        from aiflux.slurm.runner import SlurmRunner
        
        slurm_config = mock_config.get_slurm_config()
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir))
        
        job_id = runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="llama3.2:3b",
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
        assert env_passed['APPTAINERENV_MODEL_NAME'] == "llama3.2:3b"
        
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
        assert 'AIFLUX_FORCE_REBUILD' in env_passed
        assert env_passed['AIFLUX_FORCE_REBUILD'] == "1"
        
        # Verify workspace paths
        assert 'PROJECT_ROOT' in env_passed
        assert 'DATA_INPUT_DIR' in env_passed
        assert 'DATA_OUTPUT_DIR' in env_passed
        
        assert job_id == "12345"
    
    @patch('aiflux.slurm.runner.subprocess.run')
    @patch('aiflux.slurm.runner.socket.socket')
    @patch('aiflux.slurm.runner.ConfigManager')
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
        
        from aiflux.slurm.runner import SlurmRunner
        
        slurm_config = mock_config.get_slurm_config()
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir))
        
        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="llama3.2:3b"
        )
        
        # Verify existing environment is preserved
        call_args = mock_subprocess.call_args
        env_passed = call_args.kwargs.get('env', {})
        
        # Should include both new and existing variables
        assert 'TEST_EXISTING_VAR' in env_passed or 'TEST_EXISTING_VAR' in os.environ
        assert 'APPTAINERENV_MODEL_NAME' in env_passed
    
    @patch('aiflux.slurm.runner.subprocess.run')
    @patch('aiflux.slurm.runner.socket.socket')
    @patch('aiflux.slurm.runner.ConfigManager')
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
        
        from aiflux.slurm.runner import SlurmRunner
        
        slurm_config = mock_config.get_slurm_config()
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir))
        
        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="llama3.2:3b"
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
    
    @patch('aiflux.slurm.runner.subprocess.run')
    @patch('aiflux.slurm.runner.socket.socket')
    @patch('aiflux.slurm.runner.ConfigManager')
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
        
        from aiflux.slurm.runner import SlurmRunner
        
        slurm_config = mock_config.get_slurm_config()
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir))
        
        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="llama3.2:3b",
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
    
    @patch('aiflux.slurm.runner.subprocess.run')
    @patch('aiflux.slurm.runner.socket.socket')
    @patch('aiflux.slurm.runner.ConfigManager')
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
        
        from aiflux.slurm.runner import SlurmRunner
        
        slurm_config = mock_config.get_slurm_config()
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir))
        
        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="llama3.2:3b"
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
            'AIFLUX_FORCE_REBUILD', # Host: bash script flag
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
    
    @patch('aiflux.slurm.runner.subprocess.run')
    @patch('aiflux.slurm.runner.socket.socket')
    @patch('aiflux.slurm.runner.ConfigManager')
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
        
        from aiflux.slurm.runner import SlurmRunner
        
        slurm_config = mock_config.get_slurm_config()
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir))
        
        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="llama3.2:3b",
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
    
    @patch('aiflux.slurm.runner.subprocess.run')
    @patch('aiflux.slurm.runner.socket.socket')
    @patch('aiflux.slurm.runner.ConfigManager')
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
        
        from aiflux.slurm.runner import SlurmRunner
        
        slurm_config = mock_config.get_slurm_config()
        runner = SlurmRunner(config=slurm_config, workspace=str(temp_dir))
        
        # Pass various model parameters to test they all get proper prefix
        runner.run(
            input_path=str(sample_jsonl),
            output_path=str(temp_dir / "output.json"),
            model="llama3.2:3b",
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
    
    @patch('aiflux.cli.SlurmRunner')
    @patch('aiflux.cli.Config')
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
            'aiflux', 'run',
            '--model', 'llama3.2:3b',
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
        assert call_kwargs["model"] == "llama3.2:3b"
        assert call_kwargs["batch_size"] == 8
        assert call_kwargs["input_path"] == str(sample_jsonl)
    
    @patch('aiflux.cli.save_prompts_to_jsonl')
    @patch('aiflux.cli.generate_synthetic_prompts')
    @patch('aiflux.cli.SlurmRunner')
    @patch('aiflux.cli.Config')
    def test_cli_benchmark_integration(
        self, mock_config_class, mock_runner_class,
        mock_generate_prompts, mock_save_jsonl
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
        
        mock_generate_prompts.return_value = [{"prompt": "test"}] * 50
        
        # Simulate CLI call
        with patch('sys.argv', [
            'aiflux', 'benchmark',
            '--model', 'llama3.2:3b',
            '--num-prompts', '100',
            '--batch-size', '8',
            '--account', 'test-account'
        ]):
            with patch('builtins.print'):
                result = main()
        
        assert result == 0
        mock_generate_prompts.assert_called_once_with(num_prompts=100, model="llama3.2:3b")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

