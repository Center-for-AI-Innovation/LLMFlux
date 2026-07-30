#!/usr/bin/env python3
import os
import re
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import logging
from pydantic import BaseModel, Field, field_validator

class EngineConfig(BaseModel):
    """Model engine configuration."""
    engine: str = Field(..., pattern=r"^ollama|vllm$")
    origins: str = Field(default="*")
    insecure: str = Field(default="true")
    home: Optional[str] = None

class ResourceConfig(BaseModel):
    """Model resource configuration."""
    gpu_layers: int = Field(..., ge=1)
    gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    batch_size: int = Field(..., ge=1)
    max_concurrent: int = Field(..., ge=1)

class ParameterConfig(BaseModel):
    """Model parameter configuration."""
    temperature: float = Field(..., ge=0.0, le=1.0)
    top_p: float = Field(..., ge=0.0, le=1.0)
    max_tokens: int = Field(..., ge=1)
    stop_sequences: List[str]

class ModelParameters(BaseModel):
    """Model parameters for inference."""
    temperature: float = Field(0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(2048, ge=1)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    top_k: int = Field(40, ge=0)
    stop_sequences: Optional[List[str]] = None

class SystemConfig(BaseModel):
    """Model system configuration."""
    prompt: str

class ValidationConfig(BaseModel):
    """Model validation configuration."""
    temperature_range: List[float] = Field(..., min_length=2, max_length=2)
    max_tokens_limit: int = Field(..., ge=1)
    batch_size_range: List[int] = Field(..., min_length=2, max_length=2)
    concurrent_range: List[int] = Field(..., min_length=2, max_length=2)

    @field_validator("temperature_range")
    @classmethod
    def validate_temperature_range(cls, v):
        if not (0.0 <= v[0] <= v[1] <= 1.0):
            raise ValueError("Temperature range must be between 0.0 and 1.0")
        return v

    @field_validator("batch_size_range", "concurrent_range")
    @classmethod
    def validate_range(cls, v):
        if not (v[0] <= v[1]):
            raise ValueError("Range start must be less than or equal to end")
        return v

class RequirementsConfig(BaseModel):
    """Model requirements configuration."""
    min_gpu_memory: str = Field(..., pattern=r"^\d+GB$")
    recommended_gpu: str
    cuda_version: str = Field(..., pattern=r"^>=\d+\.\d+$")
    cpu_threads: int = Field(..., ge=1)
    gpu_memory_utilization: float = Field(..., ge=0.0, le=1.0)

class ModelConfig(BaseModel):
    """Complete model configuration."""
    # name: str = Field(..., pattern=r"^[a-zA-Z0-9.-]+([-][a-zA-Z0-9.]+)*:((8x)?\d+b|mini|medium|small|vision|large|tiny|instruct)$")
    name: str = None
    hf_name: Optional[str] = None
    engine: str = Field("vllm", pattern=r"^ollama|vllm$")
    type: str = Field("vllm")
    size: Optional[str] = None
    parameters: ModelParameters = Field(default_factory=ModelParameters)
    path: Optional[str] = None
    description: Optional[str] = None
    capabilities: List[str] = Field(default_factory=lambda: ["text"])
    resources: Optional[ResourceConfig] = None
    system: Optional[SystemConfig] = None
    validation: Optional[ValidationConfig] = None
    requirements: Optional[RequirementsConfig] = None
    
    def get_model_name_for_engine(self) -> str:
        """Get the appropriate model name for the configured engine.
        
        Returns:
            str: The Ollama name (e.g., 'qwen2.5:7b') if engine is 'ollama',
                 or the HuggingFace name (e.g., 'Qwen/Qwen2.5-7B-Instruct') if engine is 'vllm'
        """
        if self.engine == 'vllm':
            if self.hf_name:
                return self.hf_name
            else:
                raise ValueError(f"Model config for '{self.name}' does not have hf_name set for vLLM engine")
        else:
            return self.name

def _parse_extra_sbatch_args() -> Optional[Dict[str, str]]:
    """Parse SLURM_EXTRA_ARGS from environment variable.

    Supports JSON format: {"reservation": "my_res", "qos": "high"}
    Or key=value format: "reservation=my_res,qos=high"

    Returns:
        Dictionary of extra SBATCH arguments or None
    """
    extra_args_str = os.getenv('SLURM_EXTRA_ARGS')
    if not extra_args_str:
        return None

    # Try JSON format first
    if extra_args_str.strip().startswith('{'):
        try:
            return json.loads(extra_args_str)
        except json.JSONDecodeError:
            pass

    # Try key=value format
    result = {}
    for pair in extra_args_str.split(','):
        pair = pair.strip()
        if '=' in pair:
            key, value = pair.split('=', 1)
            result[key.strip()] = value.strip()

    return result if result else None

class SlurmConfig(BaseModel):
    """SLURM configuration with env var support."""
    account: str = Field(default_factory=lambda: os.getenv('SLURM_ACCOUNT'))
    partition: str = Field(
        default_factory=lambda: os.getenv('SLURM_PARTITION', 'a100')
    )
    nodes: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_NODES', '1'))
    )
    gpus_per_node: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_GPUS_PER_NODE', '1'))
    )
    time: str = Field(
        default_factory=lambda: os.getenv('SLURM_TIME', '00:30:00')
    )
    memory: str = Field(
        default_factory=lambda: os.getenv('SLURM_MEM', '32G')
    )
    cpus_per_task: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_CPUS_PER_TASK', '4'))
    )
    ntasks: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_NTASKS', '1'))
    )
    ntasks_per_node: int = Field(
        default_factory=lambda: int(os.getenv('SLURM_NTASKS_PER_NODE', '1'))
    )
    engine: str = Field(
        default_factory=lambda: os.getenv('SLURM_ENGINE', 'vllm')
    )
    extra_sbatch_args: Optional[Dict[str, str]] = Field(
        default_factory=_parse_extra_sbatch_args
    )

def parse_gpu_memory(memory_str: str) -> int:
    """Convert GPU memory string to GB value."""
    match = re.match(r"^(\d+)GB$", memory_str)
    if not match:
        raise ValueError(f"Invalid GPU memory format: {memory_str}")
    return int(match.group(1))

class Config:
    """Central configuration management."""
    
    def __init__(self,
                 workspace: Optional[str] = None,
                 data_dir: Optional[str] = None,
                 data_input_dir: Optional[str] = None,
                 data_output_dir: Optional[str] = None,
                 models_dir: Optional[str] = None,
                 logs_dir: Optional[str] = None,
                 containers_dir: Optional[str] = None,
                 slurm: Optional[SlurmConfig] = None,
                 models: Optional[List[ModelConfig]] = None,
                 engine: Optional[str] = None):
        """Initialize configuration.

        Args:
            workspace: Optional path to workspace directory (defaults to current working directory)
            data_dir: Optional path to data directory
            data_input_dir: Optional path to input directory (defaults to {data_dir}/input)
            data_output_dir: Optional path to output directory (defaults to {data_dir}/output)
            models_dir: Optional path to models directory
            logs_dir: Optional path to logs directory
            containers_dir: Optional path to containers directory
            slurm: Optional SLURM configuration
            models: Optional list of model configurations
            engine: Optional string for which engine to use
        """
        self.package_dir = Path(__file__).parent.parent
        self.templates_dir = self.package_dir / 'templates'

        # Load environment variables from .env file if it exists
        self._load_env_file()

        # Initialize workspace paths
        self.workspace = Path(workspace or os.getenv('LLMFLUX_WORKSPACE') or Path.cwd()).expanduser().resolve()
        
        # Set directories from parameters or environment variables
        self.data_dir = data_dir or os.getenv('LLMFLUX_DATA_DIR') or str(self.workspace / "data")
        self.data_input_dir = data_input_dir or os.getenv('LLMFLUX_DATA_INPUT_DIR') or str(Path(self.data_dir) / "input")
        self.data_output_dir = data_output_dir or os.getenv('LLMFLUX_DATA_OUTPUT_DIR') or str(Path(self.data_dir) / "output")
        self.models_dir = models_dir or os.getenv('LLMFLUX_MODELS_DIR') or str(self.workspace / "models")
        self.logs_dir = logs_dir or os.getenv('LLMFLUX_LOGS_DIR') or str(self.workspace / "logs")
        self.containers_dir = containers_dir or os.getenv('LLMFLUX_CONTAINERS_DIR') or str(self.workspace / "containers")
        
        # Set SLURM configuration
        self.slurm = slurm or SlurmConfig()
        
        # Set model configurations
        self.models = models or []

        # Set engine configuration from environment variable
        engine_value = os.getenv('SLURM_ENGINE', 'vllm')
        if engine_value == "ollama":
            self.engine = EngineConfig(
                engine="ollama",
                home=str(self.workspace / ".ollama")
            )
        else:
            self.engine = EngineConfig(
                engine="vllm",
                home=str(self.workspace / ".vllm")
            )

    def _load_env_file(self):
        """Load environment variables from .env file in project root."""
        # Try to find .env file in current directory or parent directories
        current_dir = Path.cwd()
        env_file = current_dir / '.env'
        
        # Check up to 3 parent directories
        for _ in range(3):
            if env_file.exists():
                self._load_env_from_file(env_file)
                return
            
            # Move up one directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root directory
                break
            current_dir = parent_dir
            env_file = current_dir / '.env'
    
    def _load_env_from_file(self, env_file: Path):
        """Load environment variables from the specified .env file.
        
        Environment variables are loaded with the following precedence:
        1. Existing environment variables (highest priority)
        2. Variables from .env file (middle priority)
        3. Default values (lowest priority)
        """
        try:
            with open(env_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if not line or line.startswith('#'):
                        continue
                    
                    # Parse key-value pairs
                    if '=' in line:
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        # Remove inline comments if present
                        if '#' in value:
                            value = value.split('#', 1)[0].strip()
                        
                        # Only set if not already in environment and value is not empty
                        if key and value and key not in os.environ:
                            os.environ[key] = value
        except Exception as e:
            # Log the error but continue execution
            import logging
            logging.warning(f"Error loading .env file: {str(e)}")
            pass
    
    def ensure_directory(self, path: Path) -> Path:
        """Ensure a directory exists.
        
        Args:
            path: Path to ensure
            
        Returns:
            The same path
        """
        if path.is_dir():
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path
    
    def load_model_config(
        self,
        model_key: str,
        custom_config_path: Optional[str] = None,
    ) -> ModelConfig:
        """Load and validate model configuration.
        
        Args:
            model_key: Key of the model in models.yaml (e.g., 'Llama-3.2-3B-Instruct', 'Qwen2.5-32B-Instruct')
            custom_config_path: Optional path to custom config
        Returns:
            Validated ModelConfig
        """

        try:
            config_data = None

            if custom_config_path:
                with open(custom_config_path, 'r') as f:
                    raw_config_data = yaml.safe_load(f) or {}

                if not isinstance(raw_config_data, dict):
                    raise TypeError(f"Custom config file '{custom_config_path}' must contain a YAML mapping")

                if 'models' in raw_config_data:
                    models = raw_config_data.get('models', {})
                    if not isinstance(models, dict):
                        raise TypeError(f"Custom config file '{custom_config_path}' has invalid 'models' structure")
                    if model_key not in models:
                        raise FileNotFoundError(f"Model '{model_key}' not found in custom config: {custom_config_path}")
                    config_data = models[model_key]
                elif model_key in raw_config_data and isinstance(raw_config_data[model_key], dict):
                    config_data = raw_config_data[model_key]
                else:
                    config_data = raw_config_data
            else:
                with open(self.templates_dir / 'models.yaml', 'r') as f:
                    all_models_data = yaml.safe_load(f)

                models = all_models_data.get('models', {})

                if model_key in models:
                    config_data = models[model_key]
                else:
                    raise FileNotFoundError(f"Model '{model_key}' not found in models.yaml")

            if not isinstance(config_data, dict):
                raise TypeError(f"Model '{model_key}' config must be a mapping")

            # Create basic model config
            model_config = ModelConfig(
                name=config_data.get("name"),
                hf_name=config_data.get("hf_name"),
                type=model_key,
                parameters=ModelParameters(**config_data.get("parameters", {})),
                resources=ResourceConfig(**config_data.get("resources", {})) if "resources" in config_data else None,
                system=SystemConfig(**config_data.get("system", {})) if "system" in config_data else None,
                validation=ValidationConfig(**config_data.get("validation", {})) if "validation" in config_data else None,
                requirements=RequirementsConfig(**config_data.get("requirements", {})) if "requirements" in config_data else None,
            )
            return model_config
            
        except (FileNotFoundError, KeyError, TypeError) as e:
            logging.warning(f"Could not load config for '{model_key}'. Reason: {e}")
            return None
    
    def get_slurm_config(
        self,
        overrides: Optional[Dict[str, Any]] = None
    ) -> SlurmConfig:
        """Get SLURM configuration with overrides.
        
        Args:
            overrides: Optional configuration overrides
            
        Returns:
            SlurmConfig instance
        """
        config = self.slurm
        if overrides:
            for key, value in overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        return config
    
    def to_env_dict(self) -> Dict[str, str]:
        """Convert configurations to environment variables.
        
        Returns:
            Dictionary of environment variables
        """
        slurm_config = self.get_slurm_config()
        
        env_dict = {
            'SLURM_ACCOUNT': slurm_config.account,
            'SLURM_PARTITION': slurm_config.partition,
            'SLURM_NODES': str(slurm_config.nodes),
            'SLURM_GPUS_PER_NODE': str(slurm_config.gpus_per_node),
            'SLURM_TIME': slurm_config.time,
            'SLURM_MEM': slurm_config.memory,
            'SLURM_CPUS_PER_TASK': str(slurm_config.cpus_per_task)
        }
        
        return {k: v for k, v in env_dict.items() if v is not None}
