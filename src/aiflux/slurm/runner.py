#!/usr/bin/env python3
import logging
import os
import subprocess
import socket
import shutil
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union
import tempfile
import json

from ..core.config import Config, SlurmConfig
from ..core.config_manager import ConfigManager
from ..core.processor import BaseProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SlurmRunner:
    """Runner for executing processors on SLURM."""
    
    def __init__(
        self,
        config: Optional[SlurmConfig] = None,
        workspace: Optional[str] = None
    ):
        """Initialize SLURM runner.
        
        Args:
            config: SLURM configuration
            workspace: Path to workspace directory
        """
        # Initialize config
        self.config_manager = ConfigManager()
        self.slurm_config = config or self.config_manager.get_config().get_slurm_config()
        
        # Set workspace
        self.workspace = Path(workspace) if workspace else Path(self.config_manager.get_config().workspace)
        
        # Get paths from config if available
        config = self.config_manager.get_config()
        
        # Get paths using config manager (following precedence rules)
        self.data_dir = Path(config.data_dir) if hasattr(config, 'data_dir') else (self.workspace / "data")
        self.data_input_dir = Path(config.data_input_dir) if hasattr(config, 'data_input_dir') else (self.data_dir / "input")
        self.data_output_dir = Path(config.data_output_dir) if hasattr(config, 'data_output_dir') else (self.data_dir / "output")
        self.models_dir = Path(config.models_dir) if hasattr(config, 'models_dir') else (self.workspace / "models")
        self.logs_dir = Path(config.logs_dir) if hasattr(config, 'logs_dir') else (self.workspace / "logs")
        self.containers_dir = Path(config.containers_dir) if hasattr(config, 'containers_dir') else (self.workspace / "containers")
        
        # Create directories
        for directory in [
            self.data_dir,
            self.data_input_dir,
            self.data_output_dir,
            self.models_dir,
            self.logs_dir,
            self.containers_dir,
            self.workspace / "tmp",
            self.workspace / "tmp" / "cache"
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _setup_environment(self, workspace: Optional[str] = None) -> Dict[str, str]:
        """Setup environment variables for SLURM job.
        
        Args:
            workspace: Optional workspace path to override the default
            
        Returns:
            Dictionary of environment variables
        """
        # Get package root directory for container definition
        package_root = Path(__file__).parent.parent
        container_def = package_root / "container" / "container.def"
        
        # Use workspace if provided, otherwise use the default
        workspace_path = Path(workspace) if workspace else self.workspace
        
        # Calculate GPU configuration values
        cuda_visible_devices = '0'  # Default to single GPU
        ollama_sched_spread = '0'   # Default to no spread
        
        # Update values if multiple GPUs are requested
        if self.slurm_config.gpus_per_node > 1:
            # Generate comma-separated list of GPU indices (0,1,2,...)
            cuda_visible_devices = ','.join(str(i) for i in range(self.slurm_config.gpus_per_node))
            ollama_sched_spread = '1'
        
        # Use config manager to get environment with proper precedence
        # Variables are categorized into:
        # - Host-only vars: Used by bash script on host (not passed to container)
        # - APPTAINERENV_ vars: Automatically passed to container with --cleanenv
        
        # HOST-ONLY variables (used by bash script, NOT passed to container)
        host_vars = {
            'DATA_INPUT_DIR': str(self.data_input_dir),
            'DATA_OUTPUT_DIR': str(self.data_output_dir),
            'MODELS_DIR': str(self.models_dir),
            'LOGS_DIR': str(self.logs_dir),
            'CONTAINERS_DIR': str(self.containers_dir),
            'CONTAINER_DEF': str(container_def),
            'APPTAINER_TMPDIR': str(workspace_path / "tmp"),
            'APPTAINER_CACHEDIR': str(workspace_path / "tmp" / "cache"),
            'SINGULARITY_TMPDIR': str(workspace_path / "tmp"),
            'SINGULARITY_CACHEDIR': str(workspace_path / "tmp" / "cache"),
            'OLLAMA_HOME': str(self.workspace / ".ollama"),  # Used for mkdir and --bind
            'OLLAMA_MODELS': str(self.workspace / ".ollama" / "models"),  # Used for mkdir
            'PROJECT_ROOT': str(workspace_path),  # Used in bash script for Python path
        }
        
        # CONTAINER variables (automatically injected with --cleanenv via APPTAINERENV_ prefix)
        # The prefix is removed inside the container, so APPTAINERENV_FOO becomes FOO
        container_vars = {
            'APPTAINERENV_PROJECT_ROOT': str(workspace_path),
            'APPTAINERENV_OLLAMA_HOME': str(self.workspace / ".ollama"),
            'APPTAINERENV_OLLAMA_MODELS': str(self.workspace / ".ollama" / "models"),
            'APPTAINERENV_OLLAMA_ORIGINS': '*',
            'APPTAINERENV_OLLAMA_INSECURE': 'true',
            'APPTAINERENV_CUDA_VISIBLE_DEVICES': cuda_visible_devices,
            'APPTAINERENV_OLLAMA_SCHED_SPREAD': ollama_sched_spread,
            'APPTAINERENV_CURL_CA_BUNDLE': '',  # Disable SSL cert checking
            'APPTAINERENV_SSL_CERT_FILE': '',   # Disable SSL cert checking
        }
        
        # Get base environment
        env = dict(os.environ)
        
        # Check for None values and log them
        all_vars = {**host_vars, **container_vars}
        none_values = {k: v for k, v in all_vars.items() if v is None}
        if none_values:
            logger.warning(f"The following environment variables are None and will be skipped: {list(none_values.keys())}")
        
        # Add all variables, filtering out None values
        env.update({k: v for k, v in host_vars.items() if v is not None})
        env.update({k: v for k, v in container_vars.items() if v is not None})
        
        return env
    
    def _find_available_port(self) -> int:
        """Find an available port for the server.
        
        Returns:
            Available port number
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(('', 0))
            port = s.getsockname()[1]
            # Ensure we return an integer, not a MagicMock
            if isinstance(port, int):
                return port
            else:
                # Fallback to a default port if we're in a test environment
                return 11434
        finally:
            s.close()
    
    def run(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        processor: Optional[BaseProcessor] = None,
        **kwargs
    ) -> str:
        """Run processor on SLURM.
        
        Args:
            input_path: Path to JSONL input file
            output_path: Path to save results
            processor: Optional processor to run (not used in SLURM mode)
            **kwargs: Additional parameters for processor
            
        Returns:
            Job ID of the submitted SLURM job
        """
        # Setup paths following precedence: code paths > environment variables > defaults
        # Use config manager to resolve paths
        
        # 1. For input:
        # If input_path is a file path, use it directly
        input_file = Path(input_path)
        if not input_file.exists():
            # If it doesn't exist, check if it's relative to the data input directory
            config = self.config_manager.get_config()
            data_input_dir = config.get_path('DATA_INPUT_DIR')
            potential_path = data_input_dir / input_file.name
            if potential_path.exists():
                input_file = potential_path
            else:
                # If still not found, use the input_path as is
                # It might be created by the script or specified as an output location
                pass
        
        # 2. For output:
        if output_path:
            output_file = Path(output_path)
        else:
            config = self.config_manager.get_config()
            output_dir = config.get_path('DATA_OUTPUT_DIR')
            output_file = output_dir / f"results_{int(time.time())}.json"
        
        # Ensure directories exist
        config = self.config_manager.get_config()
        config.ensure_directory(input_file.parent if input_file.is_file() else input_file)
        config.ensure_directory(output_file.parent)
        
        # Copy input to workspace if needed
        if not input_file.is_relative_to(self.workspace) and input_file.exists():
            workspace_input = self.data_input_dir / input_file.name
            if input_file.is_file():
                workspace_input.parent.mkdir(parents=True, exist_ok=True)
                workspace_input.write_bytes(input_file.read_bytes())
            else:
                # If directory exists, remove it first to avoid FileExistsError
                if workspace_input.exists():
                    shutil.rmtree(workspace_input)
                shutil.copytree(input_file, workspace_input)
            input_file = workspace_input
        
        # Setup environment
        env = self._setup_environment()

        # Optionally force container rebuild via CLI flag or env var
        try:
            rebuild_requested = bool(kwargs.get("rebuild", False))
        except Exception:
            rebuild_requested = False
        # Host-only variable (used in bash script if condition)
        env["AIFLUX_FORCE_REBUILD"] = "1" if rebuild_requested or os.getenv("AIFLUX_FORCE_REBUILD") == "1" else "0"
        
        # Add processor configuration to environment following the established priority system
        # Use ConfigManager for consistent parameter prioritization
        
        # Set model name using proper configuration priority system
        model_name = self.config_manager.get_parameter(
            param_name="model_config.name",
            code_value=kwargs.get('model'),
            obj=processor,
            env_var="MODEL_NAME",
            default="llama3.2:3b"  # Default model from templates
        )
        # Host-only (used in bash script for model pull)
        env['OLLAMA_MODEL_NAME'] = str(model_name)
        # Container variable (used in Python inside container)
        env['APPTAINERENV_MODEL_NAME'] = str(model_name)
        
        # Get batch size using config manager priority system
        batch_size = self.config_manager.get_parameter(
            param_name="batch_size",
            code_value=kwargs.get('batch_size'),
            obj=processor,
            env_var="BATCH_SIZE",
            default="4"
        )
        env['APPTAINERENV_BATCH_SIZE'] = str(batch_size)
        
        # Get save_frequency using config manager priority system
        save_frequency = self.config_manager.get_parameter(
            param_name="save_frequency",
            code_value=kwargs.get('save_frequency'),
            obj=processor,
            env_var="SAVE_FREQUENCY",
            default="50"
        )
        env['APPTAINERENV_SAVE_FREQUENCY'] = str(save_frequency)
        
        # Add additional parameters from kwargs through config manager
        for key, value in kwargs.items():
            # Skip parameters that are already handled
            if key in ['model', 'batch_size', 'save_frequency']:
                continue
                
            # Use config manager to get the value with proper priority
            param_value = self.config_manager.get_parameter(
                param_name=key,
                code_value=value,
                obj=processor if hasattr(processor, key) else None,
                env_var=key.upper(),
                default=None
            )
            
            if param_value is not None:
                if isinstance(param_value, (str, int, float, bool)):
                    env[f'APPTAINERENV_{key.upper()}'] = str(param_value)
                elif isinstance(param_value, (dict, list)):
                    env[f'APPTAINERENV_{key.upper()}'] = json.dumps(param_value)
        
        # Find available port
        port = self._find_available_port()
        # Host variable (used in bash curl commands)
        env['OLLAMA_PORT'] = str(port)
        # Container variables (used in Python inside container and ollama server)
        env['APPTAINERENV_OLLAMA_PORT'] = str(port)
        env['APPTAINERENV_OLLAMA_HOST'] = f"0.0.0.0:{port}"

        # Get LLM Engine
        # Todo add vllm, ollama is default
        # Move scripts to engine/ollama, engine/vllm
        
        # Create SLURM job script
        job_script = [
            "#!/bin/bash",
            f"#SBATCH --job-name=llm_processor",
            f"#SBATCH --account={self.slurm_config.account}",
            f"#SBATCH --partition={self.slurm_config.partition}",
            f"#SBATCH --nodes={self.slurm_config.nodes}",
            f"#SBATCH --gpus-per-node={self.slurm_config.gpus_per_node}",
            f"#SBATCH --time={self.slurm_config.time}",
            f"#SBATCH --mem={self.slurm_config.memory}",
            f"#SBATCH --cpus-per-task={self.slurm_config.cpus_per_task}",
            f"#SBATCH --output={self.logs_dir}/%j.out",
            f"#SBATCH --error={self.logs_dir}/%j.err",
        ]
        
        # Add extra SBATCH directives if provided
        if self.slurm_config.extra_sbatch_args:
            for key, value in self.slurm_config.extra_sbatch_args.items():
                job_script.append(f"#SBATCH --{key}={value}")
        
        job_script.extend([
            "",
            "# Create all necessary directories",
            "mkdir -p $DATA_INPUT_DIR $DATA_OUTPUT_DIR $MODELS_DIR $LOGS_DIR $CONTAINERS_DIR $APPTAINER_TMPDIR $APPTAINER_CACHEDIR",
            "",
            "# Start Ollama server",
            "mkdir -p $OLLAMA_HOME $OLLAMA_MODELS",
            "",
            "# Build container if needed (or if forced)",
            "if [ \"$AIFLUX_FORCE_REBUILD\" = \"1\" ] || [ ! -f \"$CONTAINERS_DIR/llm_processor.sif\" ]; then",
            "    apptainer build --force $CONTAINERS_DIR/llm_processor.sif $CONTAINER_DEF",
            "fi",
            "",
            "# Start server with clean environment",
            "# All APPTAINERENV_* variables are automatically passed in (prefix removed)",
            "OLLAMA_DEBUG=1 apptainer exec --nv --cleanenv \\",
            "    --bind $DATA_INPUT_DIR:/app/data/input,$DATA_OUTPUT_DIR:/app/data/output,$MODELS_DIR:/app/models,$LOGS_DIR:/app/logs,$OLLAMA_HOME:$OLLAMA_HOME \\",
            "    $CONTAINERS_DIR/llm_processor.sif \\",
            "    ollama serve &",
            "",
            "OLLAMA_PID=$!",
            "",
            "# Wait for server",
            "for i in {1..60}; do",
            "    if curl -s \"http://localhost:$OLLAMA_PORT/api/version\" &>/dev/null; then",
            "        echo \"Ollama server started\"",
            "        break",
            "    fi",
            "    if ! ps -p $OLLAMA_PID > /dev/null; then",
            "        echo \"Ollama server died\"",
            "        exit 1",
            "    fi",
            "    echo \"Waiting... ($i/60)\"",
            "    sleep 1",
            "done",
            "",
            "# Pull model if needed",
            "MODEL_NAME=\"${OLLAMA_MODEL_NAME:-llama3.2:3b}\"",
            "echo \"Checking if model ${MODEL_NAME} exists...\"",
            "",
            # "# Extract base model name for Ollama (e.g. llama3.2:3b -> llama3.2)",
            # "if [[ \"$MODEL_NAME\" == *\":\"* ]]; then",
            # "    BASE_MODEL=$(echo \"$MODEL_NAME\" | cut -d':' -f1)",
            # "    echo \"Using base model name for Ollama: $BASE_MODEL\"",
            # "else",
            # "    BASE_MODEL=\"$MODEL_NAME\"",
            # "fi",
            "",
            "# Check if model exists, try to pull if it doesn't",
            "if ! curl -s \"http://localhost:$OLLAMA_PORT/api/tags\" | grep -q \"\\\"name\\\":\\\"$MODEL_NAME\\\"\"; then",
            "    echo \"Model not found, pulling base model ${MODEL_NAME}...\"",
            "    curl -X POST \"http://localhost:$OLLAMA_PORT/api/pull\" -d '{\"name\": \"'\"$MODEL_NAME\"'\"}' -H \"Content-Type: application/json\"",
            "    if [ $? -ne 0 ]; then",
            "        echo \"Failed to pull model ${MODEL_NAME}\"",
            "        echo \"Available models:\"",
            "        curl -s \"http://localhost:$OLLAMA_PORT/api/tags\" | grep -o '\"name\":\"[^\"]*\"' || echo \"None found\"",
            "        exit 1",
            "    else",
            "        echo \"Successfully pulled model ${MODEL_NAME}\"",
            "    fi",
            "else",
            "    echo \"Model ${MODEL_NAME} already exists\"",
            "fi",
            "",
            "# Run processor",
            f"python3 -c \"",
            "import sys",
            "import os",
            "sys.path.append('$PROJECT_ROOT')",
            "from aiflux.core.config import Config",
            "from aiflux.processors import BatchProcessor",
            "",
            "# Ensure OLLAMA environment variables are available in Python",
            "ollama_port = os.environ.get('OLLAMA_PORT')",
            "if ollama_port:",
            "    os.environ['OLLAMA_HOST'] = f'http://localhost:{ollama_port}'",
            "",
            "# Load model configuration",
            "config = Config()",
            "model_name = os.environ.get('MODEL_NAME', 'llama3.2:3b')",
            "model_type = model_name.split(':')[0] if ':' in model_name else model_name",
            "model_size = model_name.split(':')[1] if ':' in model_name else '3b'",
            "",
            "try:",
            "    model_config = config.load_model_config(model_type, model_size)",
            "except Exception as e:",
            "    print(f'Error loading model config for {model_type}:{model_size}: {e}')",
            "    # Fallback to default model",
            "    model_config = config.load_model_config('qwen2.5', '7b')",
            "",
            "# Create batch processor with JSONL input",
            "batch_processor = BatchProcessor(",
            "    model_config=model_config,",
            "    batch_size=int(os.environ.get('BATCH_SIZE', '4')),",
            "    save_frequency=int(os.environ.get('SAVE_FREQUENCY', '50')),",
            "    max_retries=int(os.environ.get('MAX_RETRIES', '3')),",
            "    retry_delay=float(os.environ.get('RETRY_DELAY', '1.0'))",
            ")",
            "",
            "# Prepare run kwargs",
            "run_kwargs = {}",
            "",
            "# Add any other kwargs from environment variables",
            "for key in ['max_tokens', 'temperature', 'top_p', 'top_k']:",
            "    if key.upper() in os.environ:",
            "        run_kwargs[key] = os.environ[key.upper()]",
            "",
            f"batch_processor.run('{input_file}', '{output_file}', **run_kwargs)",
            "\"",
            "",
            "# Cleanup",
            "pkill -f \"ollama serve\" || true",
            "# Only remove temporary directories that we created",
            "if [ -d \"$APPTAINER_TMPDIR\" ] && [ -w \"$APPTAINER_TMPDIR\" ]; then",
            "    rm -rf \"$APPTAINER_TMPDIR\"",
            "fi",
            "if [ -d \"$APPTAINER_CACHEDIR\" ] && [ -w \"$APPTAINER_CACHEDIR\" ]; then",
            "    rm -rf \"$APPTAINER_CACHEDIR\"",
            "fi"
        ])
        
        # Write job script
        job_script_path = self.workspace / "job.sh"
        debug_mode = kwargs.get('debug', False)
        
        try:
            with open(job_script_path, 'w') as f:
                f.write('\n'.join(job_script))
            
            if debug_mode:
                logger.info(f"Debug mode: job script saved to {job_script_path}")
            
            # Submit job
            try:
                result = subprocess.run(
                    ['sbatch', str(job_script_path)],
                    env=env,
                    check=True,
                    capture_output=True,
                    text=True
                )
                logger.info("Job submitted successfully")
                
                # Extract job ID from output
                output = result.stdout.strip()
                job_id = output.split()[-1] if output else "unknown"
                return job_id
                
            except subprocess.CalledProcessError as e:
                logger.error(f"Error submitting job: {e}")
                logger.error(f"STDERR: {e.stderr}")
                logger.error(f"STDOUT: {e.stdout}")
                raise
            
        finally:
            # Cleanup job script if it exists (unless debug mode)
            if not debug_mode and job_script_path.exists():
                job_script_path.unlink() 