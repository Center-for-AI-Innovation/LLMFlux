#!/usr/bin/env python3
import logging
import os
import secrets
import subprocess
import socket
import shutil
import time
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any
import json

from .engine import create_vllm_batch_script
from .engine import create_ollama_batch_script
from .topology import resolve as resolve_topology

from ..core.config import SlurmConfig, EngineConfig
from ..core.config_manager import ConfigManager
from ..core.processor import BaseProcessor
from ..core.registry import JobRegistry

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
        workspace: Optional[str] = None,
        engine_config: Optional['EngineConfig'] = None
    ):
        """Initialize SLURM runner.
        
        Args:
            config: SLURM configuration
            workspace: Path to workspace directory
            engine_config: Engine configuration
        """
        # Initialize config
        self.config_manager = ConfigManager()
        self.slurm_config = config or self.config_manager.get_config().get_slurm_config()
        
        # Set workspace
        self.workspace = Path(workspace).expanduser().resolve() if workspace else Path(self.config_manager.get_config().workspace)
        
        # Get paths from config if available
        config = self.config_manager.get_config()
        # Use provided engine config or fall back to config
        self.engine = engine_config or config.engine
        
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
        vllm_sched_spread = '0'

        # Update values if multiple GPUs are requested
        if self.slurm_config.gpus_per_node > 1:
            # Generate comma-separated list of GPU indices (0,1,2,...)
            cuda_visible_devices = ','.join(str(i) for i in range(self.slurm_config.gpus_per_node))
            ollama_sched_spread = '1'
            vllm_sched_spread = '1'

        # Use config manager to get environment with proper precedence
        # Variables are categorized into:
        # - Host-only vars: Used by bash script on host (not passed to container)
        # - APPTAINERENV_ vars: Automatically passed to container with --cleanenv

        # Resolve HuggingFace cache directory with a default under workspace.
        hf_home = os.getenv('HF_HOME') or str(workspace_path / ".cache" / "huggingface")

        # Cache locations for engine-side JIT artifacts (e.g. FlashInfer kernels).
        # These must also be host vars: the batch script uses them for mkdir and
        # --bind, and an unbound cache dir is read-only inside the container.
        xdg_cache_home = str(workspace_path / ".cache")
        flashinfer_workspace_base = str(workspace_path)

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
            'VLLM_HOME': str(self.workspace / ".vllm"),
            'VLLM_MODELS': str(self.workspace / ".vllm" / "models"),
            'HF_HOME': hf_home,  # Used for mkdir and --bind
            'XDG_CACHE_HOME': xdg_cache_home,  # Used for mkdir and --bind
            'FLASHINFER_WORKSPACE_BASE': flashinfer_workspace_base,  # Used for mkdir and --bind
            'PROJECT_ROOT': str(workspace_path),  # Used in bash script for Python path
        }

        # CONTAINER variables (automatically injected with --cleanenv via APPTAINERENV_ prefix)
        # The prefix is removed inside the container, so APPTAINERENV_FOO becomes FOO
        container_vars = {
            'APPTAINERENV_PROJECT_ROOT': str(workspace_path),
            'APPTAINERENV_OLLAMA_HOME': str(self.workspace / ".ollama"),
            'APPTAINERENV_OLLAMA_MODELS': str(self.workspace / ".ollama" / "models"),
            'APPTAINERENV_VLLM_HOME': str(self.workspace / ".vllm"),
            'APPTAINERENV_VLLM_MODELS': str(self.workspace / ".vllm" / "models"),
            'APPTAINERENV_OLLAMA_ORIGINS': '*',
            'APPTAINERENV_OLLAMA_INSECURE': 'true',
            'APPTAINERENV_CUDA_VISIBLE_DEVICES': cuda_visible_devices,
            'APPTAINERENV_OLLAMA_SCHED_SPREAD': ollama_sched_spread,
            'APPTAINERENV_VLLM_SCHED_SPREAD': vllm_sched_spread,
            'APPTAINERENV_HF_HOME': hf_home,
            'APPTAINERENV_XDG_CACHE_HOME': xdg_cache_home,
            'APPTAINERENV_FLASHINFER_WORKSPACE_BASE': flashinfer_workspace_base,
            # Use system CA bundle for HTTPS (e.g. HuggingFace model downloads).
            # Empty values break downloads with "No CA certificates were loaded".
            'APPTAINERENV_CURL_CA_BUNDLE': '/etc/ssl/certs/ca-certificates.crt',
            'APPTAINERENV_SSL_CERT_FILE': '/etc/ssl/certs/ca-certificates.crt',
        }
        
        # Add HuggingFace token if available (for accessing gated models)
        hf_token = os.getenv('HUGGINGFACE_TOKEN') or os.getenv('HF_TOKEN')
        if hf_token:
            container_vars['APPTAINERENV_HF_TOKEN'] = hf_token
        

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

    def _load_vllm_engine_args(self, raw_value: Any, source: str) -> Dict[str, Any]:
        """Parse JSON engine args into a dict."""
        if raw_value is None:
            return {}
        if isinstance(raw_value, dict):
            return raw_value
        if isinstance(raw_value, str):
            if not raw_value.strip():
                return {}
            try:
                parsed = json.loads(raw_value)
            except json.JSONDecodeError as exc:
                logger.warning(f"Invalid JSON for {source}: {exc}")
                return {}
            if not isinstance(parsed, dict):
                logger.warning(f"{source} must be a JSON object")
                return {}
            return parsed

        logger.warning(f"{source} must be a JSON object string")
        return {}

    def _build_vllm_engine_args(self, merged_args: Dict[str, Any]) -> str:
        """Build a safe flag string for vLLM engine args."""
        parts: list[str] = []
        for key, value in merged_args.items():
            if not isinstance(key, str):
                logger.warning(f"Skipping non-string vLLM engine arg key: {key}")
                continue
            flag = key if key.startswith("--") else f"--{key}"
            if isinstance(value, bool):
                if value:
                    parts.append(flag)
                continue
            if value is None:
                continue
            if isinstance(value, (int, float, str)):
                parts.append(flag)
                parts.append(shlex.quote(str(value)))
                continue
            logger.warning(
                f"Skipping unsupported vLLM engine arg type for '{key}': {type(value).__name__}"
            )

        return " ".join(parts)

    def _topology(self):
        """Validate this runner's allocation shape and return the topology.

        One definition, called by run(), serve() and the engine-args resolver.
        Spelling it out at each call site is how the entry points drift apart —
        the defect this branch already fixes twice over.

        Raises:
            TopologyError: if the requested shape cannot be served.
        """
        return resolve_topology(
            self.slurm_config.nodes,
            self.slurm_config.gpus_per_node,
            self.engine.engine,
        )

    def _resolve_vllm_engine_args(self, kwargs: Dict[str, Any]) -> str:
        """Merge vLLM engine args and apply parallelism implied by the topology.

        Precedence: user CLI args beat environment args, and both beat anything
        derived from the allocation — a value the user typed is never silently
        overridden.

        Shared by run() and serve(). It was previously duplicated verbatim
        between them, which is how the two paths drift.
        """
        env_args = self._load_vllm_engine_args(
            os.getenv("VLLM_ENGINE_ARGS"), "VLLM_ENGINE_ARGS"
        )
        cli_args = self._load_vllm_engine_args(
            kwargs.get("vllm_engine_args"), "--vllm-engine-args"
        )
        merged_args = {**env_args, **cli_args}

        topology = self._topology()
        if topology.tensor_parallel_size > 1 and "tensor-parallel-size" not in merged_args:
            merged_args["tensor-parallel-size"] = topology.tensor_parallel_size

        return self._build_vllm_engine_args(merged_args)

    def _build_job_name(self, model_identifier: str) -> str:
        """Generate a Slurm-safe, identifiable job name."""
        safe_model = "".join(
            char if char.isalnum() else "_"
            for char in str(model_identifier).strip()
        ).strip("_")
        if not safe_model:
            safe_model = "model"
        return f"llmflux_{safe_model}_{self.engine.engine}"[:120]
    
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
        # Validate the requested node/GPU shape before doing anything else, so
        # an unservable request fails in milliseconds instead of after a queue
        # wait. Raises TopologyError.
        self._topology()

        # Setup paths following precedence: code paths > environment variables > defaults
        # Use config manager to resolve paths
        
        # 1. For input:
        # If input_path is a file path, use it directly
        input_file = Path(input_path).resolve()

        if not input_file.exists():
            # If it doesn't exist, check if it's relative to the data input directory
            potential_path = self.data_input_dir / input_file.name
            if potential_path.exists():
                input_file = potential_path
            else:
                # If still not found, use the input_path as is
                # It might be created by the script or specified as an output location
                pass
        
        # 2. For output:
        if output_path:
            output_file = Path(output_path).resolve()
        else:
            output_file = self.data_output_dir / f"results_{int(time.time())}.json"
        
        # Ensure directories exist
        config = self.config_manager.get_config()
        config.ensure_directory(input_file.parent if input_file.is_file() else input_file)
        config.ensure_directory(output_file.parent)
        
        # Copy input into the data input dir only if the job can't already see it
        already_visible = (
            input_file.is_relative_to(self.workspace)
            or input_file.is_relative_to(self.data_input_dir)
        )
        if not already_visible and input_file.exists():
            if input_file.is_dir():
                raise ValueError(
                    f"Input must be a JSONL file, got a directory: {input_file}"
                )
            workspace_input = self.data_input_dir / input_file.name
            workspace_input.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(input_file, workspace_input)
            input_file = workspace_input
        
        # Setup environment
        env = self._setup_environment()

        # Optionally force container rebuild via CLI flag or env var
        rebuild_requested = bool(kwargs.get("rebuild", False))
        # Host-only variable (used in bash script if condition)
        env["LLMFLUX_FORCE_REBUILD"] = "1" if rebuild_requested or os.getenv("LLMFLUX_FORCE_REBUILD") == "1" else "0"

        # Add processor configuration to environment following the established priority system
        # Use ConfigManager for consistent parameter prioritization
        
        # Load the full model configuration from the user's input
        model_identifier = kwargs.get('model', 'Llama-3.2-3B-Instruct')
        custom_config_path = kwargs.get('custom_config_path')
        if custom_config_path:
            custom_config_path = str(Path(custom_config_path).expanduser().resolve())
            if self.engine.engine != 'vllm':
                logger.error(
                    "Custom config path is only supported with the vLLM engine. "
                    "Please rerun with --engine vllm."
                )
                return "1"

        model_config = self.config_manager.get_config().load_model_config(
            model_identifier,
            custom_config_path=custom_config_path
        )

        if not model_config:
            logger.error(f"Model '{model_identifier}' not found. Check available models in models.yaml.")
            return "1"

        # Validate engine compatibility with model
        if self.engine.engine == 'ollama' and (model_config.name is None or model_config.name == 'NA'):
            logger.error(
                f"Model '{model_identifier}' is not available for Ollama engine. "
                f"This model only supports vLLM engine. "
                f"Please use --engine vllm to run this model."
            )
            return "1"
        
        if self.engine.engine == 'vllm' and (model_config.hf_name is None or model_config.hf_name == 'NA'):
            logger.error(
                f"Model '{model_identifier}' is not available for vLLM engine. "
                f"This model only supports Ollama engine. "
                f"Please use --engine ollama to run this model."
            )
            return "1"

        # Select the appropriate model name based on the engine
        if self.engine.engine == 'ollama':
            selected_model_name = model_config.name  # e.g., "qwen2.5:7b"
            logger.info(f"Using Ollama model name: {selected_model_name}")
        elif self.engine.engine == 'vllm':
            selected_model_name = model_config.hf_name  # e.g., "Qwen/Qwen2.5-7B-Instruct"
            logger.info(f"Using vLLM (HuggingFace) model name: {selected_model_name}")
        else:
            logger.error(f"Unknown engine: {self.engine.engine}")
            return "1"

        # Set engine-specific environment variables
        # Host-only variables (used in bash scripts)
        if self.engine.engine == 'ollama':
            env['OLLAMA_MODEL_NAME'] = str(selected_model_name)
            # Also set the port for ollama
            env['OLLAMA_HOST'] = '0.0.0.0'
        elif self.engine.engine == 'vllm':
            env['VLLM_MODEL_NAME'] = str(selected_model_name)
            env['VLLM_HOST'] = '0.0.0.0'
            # Load vLLM engine args from environment and CLI, merging them with priority to CLI
            env['VLLM_ENGINE_ARGS'] = self._resolve_vllm_engine_args(kwargs)
        # Container variables (used in Python inside container)
        # Always set MODEL_IDENTIFIER for reference
        env['APPTAINERENV_MODEL_IDENTIFIER'] = str(model_identifier)
        # Always set ENGINE so the container knows which engine it's running
        env['APPTAINERENV_ENGINE'] = str(self.engine.engine)
        if custom_config_path and self.engine.engine == 'vllm':
            env['APPTAINERENV_CUSTOM_CONFIG_PATH'] = str(custom_config_path)
        
        # Set engine-specific model names for the container
        if self.engine.engine == 'ollama':
            env['APPTAINERENV_MODEL_NAME'] = str(selected_model_name)
            env['APPTAINERENV_OLLAMA_MODEL_NAME'] = str(selected_model_name)
        elif self.engine.engine == 'vllm':
            env['APPTAINERENV_VLLM_MODEL_NAME'] = str(selected_model_name)

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
            if key in ['model', 'batch_size', 'save_frequency', 'vllm_engine_args', 'custom_config_path']:
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
        env['VLLM_PORT'] = str(port)
        # Container variables (used in Python inside container and ollama server)
        env['APPTAINERENV_OLLAMA_PORT'] = str(port)
        env['APPTAINERENV_OLLAMA_HOST'] = f"0.0.0.0:{port}"

        env['APPTAINERENV_VLLM_PORT'] = str(port)
        env['APPTAINERENV_VLLM_HOST'] = f"0.0.0.0"

        job_name = self._build_job_name(model_identifier)

        # Get LLM Engine
        # Create SLURM job script
        logger.info(f"engine: {self.engine.engine}")
        if self.engine.engine == "ollama":
            job_script = create_ollama_batch_script(
                self.slurm_config.account,
                self.slurm_config.partition,
                str(self.slurm_config.nodes),
                str(self.slurm_config.gpus_per_node),
                self.slurm_config.time,
                self.slurm_config.memory,
                str(self.slurm_config.cpus_per_task),
                self.logs_dir,
                input_file,
                output_file,
                job_name,
                self.slurm_config,
            )
        elif self.engine.engine == "vllm":
            job_script = create_vllm_batch_script(
                self.slurm_config.account,
                self.slurm_config.partition,
                str(self.slurm_config.nodes),
                str(self.slurm_config.gpus_per_node),
                self.slurm_config.time,
                self.slurm_config.memory,
                str(self.slurm_config.cpus_per_task),
                self.logs_dir,
                input_file,
                output_file,
                job_name,
                self.slurm_config
            )
        else:
            logger.error("Unknown engine choice: {}".format(self.engine))
            raise NotImplementedError

        job_script_text = "\n".join(job_script)
        logging.debug(f'Job script:\n{job_script_text}')
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
                if job_id != "unknown":
                    registry = JobRegistry()
                    try:
                        registry.create_job(
                            job_id=job_id,
                            metadata={
                                "job_name": job_name,
                                "model": str(model_identifier),
                                "engine": str(self.engine.engine),
                                "input": str(input_file),
                                "output": str(output_file.resolve()),
                                "logs_dir": str(self.logs_dir),
                                "submitted_at": datetime.now(timezone.utc).isoformat(),
                            },
                        )
                    except ValueError:
                        logger.warning(f"Job {job_id} is already tracked in registry; skipping duplicate write.")
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

    def serve(self, email: str, **kwargs) -> str:
        """Start a model as a long-running service on a SLURM compute node.

        Unlike run(), there is no input file to process. The SLURM job keeps
        the engine alive for the full wall-time and writes a connection file
        so llmflux connect can display the endpoint info.

        Args:
            email: Address to notify when the service is ready (SLURM BEGIN mail)
            **kwargs: model, engine args, rebuild, debug, vllm_engine_args

        Returns:
            SLURM job ID string, or None if submission failed.
        """
        # Same topology gate as run(); serve has its own code path.
        self._topology()

        env = self._setup_environment()

        rebuild_requested = bool(kwargs.get("rebuild", False))
        env["LLMFLUX_FORCE_REBUILD"] = "1" if rebuild_requested or os.getenv("LLMFLUX_FORCE_REBUILD") == "1" else "0"

        model_identifier = kwargs.get('model', 'Llama-3.2-3B-Instruct')
        model_config = self.config_manager.get_config().load_model_config(model_identifier)

        if not model_config:
            raise ValueError(f"Model '{model_identifier}' not found. Check available models with: llmflux show-models")

        if self.engine.engine == 'ollama' and (model_config.name is None or model_config.name == 'NA'):
            raise ValueError(
                f"Model '{model_identifier}' is not available for the Ollama engine. "
                f"Please use --engine vllm to run this model."
            )

        if self.engine.engine == 'vllm' and (model_config.hf_name is None or model_config.hf_name == 'NA'):
            raise ValueError(
                f"Model '{model_identifier}' is not available for the vLLM engine. "
                f"Please use --engine ollama to run this model."
            )

        if self.engine.engine == 'ollama':
            selected_model_name = model_config.name
        else:
            selected_model_name = model_config.hf_name

        if self.engine.engine == 'ollama':
            env['OLLAMA_MODEL_NAME'] = str(selected_model_name)
            env['OLLAMA_HOST'] = '0.0.0.0'
        elif self.engine.engine == 'vllm':
            env['VLLM_MODEL_NAME'] = str(selected_model_name)
            env['VLLM_HOST'] = '0.0.0.0'
            env['VLLM_ENGINE_ARGS'] = self._resolve_vllm_engine_args(kwargs)

        env['APPTAINERENV_MODEL_IDENTIFIER'] = str(model_identifier)
        env['APPTAINERENV_ENGINE'] = str(self.engine.engine)

        if self.engine.engine == 'ollama':
            env['APPTAINERENV_MODEL_NAME'] = str(selected_model_name)
            env['APPTAINERENV_OLLAMA_MODEL_NAME'] = str(selected_model_name)
        elif self.engine.engine == 'vllm':
            env['APPTAINERENV_VLLM_MODEL_NAME'] = str(selected_model_name)

        # Port finding happens on the compute node in the SLURM script so it
        # finds a consecutive free port on the actual machine running the model.
        # Pass 8000 as the base; the script increments until it finds one free.
        base_port = 8000
        env['OLLAMA_PORT'] = str(base_port)
        env['VLLM_PORT'] = str(base_port)
        env['APPTAINERENV_OLLAMA_PORT'] = str(base_port)
        env['APPTAINERENV_OLLAMA_HOST'] = f"0.0.0.0:{base_port}"
        env['APPTAINERENV_VLLM_PORT'] = str(base_port)
        env['APPTAINERENV_VLLM_HOST'] = "0.0.0.0"

        # Generate a unique API key for this serve session
        api_key = f"llmflux-{secrets.token_hex(16)}"
        env['LLMFLUX_API_KEY'] = api_key

        job_name = self._build_job_name(model_identifier)
        debug_mode = kwargs.get('debug', False)

        # input_file/output_file are unused in serve mode but required by the
        # engine function signatures (they only appear in the batch branch).
        # Use an obviously-fake placeholder so it's clear if it ever leaks downstream.
        unused_path = Path("unused-in-serve-mode")

        if self.engine.engine == "ollama":
            job_script = create_ollama_batch_script(
                self.slurm_config.account,
                self.slurm_config.partition,
                str(self.slurm_config.nodes),
                str(self.slurm_config.gpus_per_node),
                self.slurm_config.time,
                self.slurm_config.memory,
                str(self.slurm_config.cpus_per_task),
                self.logs_dir,
                unused_path,
                unused_path,
                job_name,
                self.slurm_config,
                mode="serve",
                email=email,
            )
        elif self.engine.engine == "vllm":
            job_script = create_vllm_batch_script(
                self.slurm_config.account,
                self.slurm_config.partition,
                str(self.slurm_config.nodes),
                str(self.slurm_config.gpus_per_node),
                self.slurm_config.time,
                self.slurm_config.memory,
                str(self.slurm_config.cpus_per_task),
                self.logs_dir,
                unused_path,
                unused_path,
                job_name,
                self.slurm_config,
                mode="serve",
                email=email,
            )
        else:
            logger.error(f"Unknown engine: {self.engine.engine}")
            raise NotImplementedError

        job_script_path = self.workspace / "job.sh"

        try:
            with open(job_script_path, 'w') as f:
                f.write('\n'.join(job_script))

            if debug_mode:
                logger.info(f"Debug mode: job script saved to {job_script_path}")

            result = subprocess.run(
                ['sbatch', str(job_script_path)],
                env=env,
                check=True,
                capture_output=True,
                text=True,
            )

            output = result.stdout.strip()
            job_id = output.split()[-1] if output else None

            if not job_id:
                logger.error("sbatch did not return a job ID in its output.")
                return None

            registry = JobRegistry()
            try:
                registry.create_job(
                    job_id=job_id,
                    metadata={
                        "job_name": job_name,
                        "model": str(model_identifier),
                        "engine": str(self.engine.engine),
                        "type": "serve",
                        "email": email,
                        "api_key": api_key,
                        "logs_dir": str(self.logs_dir),
                        "submitted_at": datetime.now(timezone.utc).isoformat(),
                    },
                )
            except ValueError:
                logger.warning(f"Job {job_id} is already tracked in registry; skipping duplicate write.")

            return job_id

        except subprocess.CalledProcessError as e:
            logger.error(f"Error submitting serve job: {e}")
            logger.error(f"STDERR: {e.stderr}")
            logger.error(f"STDOUT: {e.stdout}")
            raise

        finally:
            if not debug_mode and job_script_path.exists():
                job_script_path.unlink()
