#!/usr/bin/env python3
"""Configuration Manager for LLMFlux."""

from typing import Optional, Dict, Any, List
from pathlib import Path
from .config import Config, ModelConfig, SlurmConfig, EngineConfig
import os

# Singleton instance
_config_instance = None

class ConfigManager:
    """Manager for accessing the singleton Config instance."""
    
    @staticmethod
    def get_config() -> Config:
        """Get the singleton Config instance.
        
        This ensures that the same Config instance is used throughout the application.
        
        Returns:
            Config: The singleton Config instance
        """
        global _config_instance
        if _config_instance is None:
            _config_instance = Config()
        return _config_instance
    
    @staticmethod
    def get_parameter(param_name: str, 
                      code_value: Any = None, 
                      obj: Any = None, 
                      env_var: Optional[str] = None, 
                      default: Any = None) -> Any:
        """Get parameter value following established priority system.
        
        Prioritizes values in this order:
        1. Direct code parameters (highest priority)
        2. Object attributes
        3. Environment variables
        4. Default values (lowest priority)
        
        Args:
            param_name: Name of the parameter
            code_value: Explicitly provided value (highest priority)
            obj: Object to check for attribute
            env_var: Environment variable name to check
            default: Default value to use as fallback
            
        Returns:
            The parameter value following priority rules
        """
        # 1. Check explicit code parameter (highest priority)
        if code_value is not None:
            return code_value
        
        # 2. Check object attribute
        if obj is not None:
            # Handle nested attributes like obj.model_config.name
            if hasattr(obj, param_name):
                return getattr(obj, param_name)
            elif '.' in param_name:
                # For nested attributes like 'model_config.name'
                parts = param_name.split('.')
                current = obj
                for part in parts:
                    if hasattr(current, part):
                        current = getattr(current, part)
                    else:
                        break
                else:
                    # If all parts were found
                    return current
        
        # 3. Check environment variable
        if env_var and env_var in os.environ:
            return os.environ[env_var]
        
        # 4. Use default (lowest priority)
        return default
    
    @staticmethod
    def reset_config(workspace: Optional[str] = None,
                     data_dir: Optional[str] = None,
                     data_input_dir: Optional[str] = None,
                     data_output_dir: Optional[str] = None,
                     models_dir: Optional[str] = None,
                     logs_dir: Optional[str] = None,
                     containers_dir: Optional[str] = None,
                     slurm: Optional[SlurmConfig] = None,
                     models: Optional[List[ModelConfig]] = None,
                     engine: Optional[EngineConfig] = None) -> Config:
        """Reset the singleton Config instance with new values.

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
            engine: Whether to use VLLM or OLLAMA

        Returns:
            Config: The new singleton Config instance
        """
        global _config_instance
        _config_instance = Config(
            workspace=workspace,
            data_dir=data_dir,
            data_input_dir=data_input_dir,
            data_output_dir=data_output_dir,
            models_dir=models_dir,
            logs_dir=logs_dir,
            containers_dir=containers_dir,
            slurm=slurm,
            models=models,
            engine=engine,
        )
        return _config_instance
    
    @staticmethod
    def update_config(data_dir: Optional[str] = None,
                      data_input_dir: Optional[str] = None,
                      data_output_dir: Optional[str] = None,
                      models_dir: Optional[str] = None,
                      logs_dir: Optional[str] = None,
                      containers_dir: Optional[str] = None,
                      slurm: Optional[SlurmConfig] = None,
                      models: Optional[List[ModelConfig]] = None,
                      engine: Optional[EngineConfig] = None) -> Config:
        """Update the singleton Config instance with new values.

        Only updates the provided values, keeping the rest unchanged.

        Args:
            data_dir: Optional path to data directory
            data_input_dir: Optional path to input directory (defaults to {data_dir}/input)
            data_output_dir: Optional path to output directory (defaults to {data_dir}/output)
            models_dir: Optional path to models directory
            logs_dir: Optional path to logs directory
            containers_dir: Optional path to containers directory
            slurm: Optional SLURM configuration
            models: Optional list of model configurations
            engine: Optional, either VLLM or OLLAMA
            
        Returns:
            Config: The updated singleton Config instance
        """
        config = ConfigManager.get_config()
        
        # Update only the provided values
        if data_dir:
            config.data_dir = data_dir
            # Re-derive input/output so they follow the new data_dir unless
            # explicitly overridden below
            config.data_input_dir = str(Path(data_dir) / "input")
            config.data_output_dir = str(Path(data_dir) / "output")
        if data_input_dir:
            config.data_input_dir = data_input_dir
        if data_output_dir:
            config.data_output_dir = data_output_dir
        if models_dir:
            config.models_dir = models_dir
        if logs_dir:
            config.logs_dir = logs_dir
        if containers_dir:
            config.containers_dir = containers_dir
        if slurm:
            config.slurm = slurm
        if models:
            config.models = models
        if engine:
            config.engine = engine

        return config
