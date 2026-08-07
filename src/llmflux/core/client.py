#!/usr/bin/env python3
"""OpenAI-compatible client for Ollama LLM service."""

import os
import logging
import time
import json
from typing import Dict, Any, Optional, List
import requests
from dotenv import load_dotenv


# Load environment variables
load_dotenv(override=True)

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LLMClient:
    """OpenAI-compatible client for LLM services."""
    
    def __init__(
        self,
        engine: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ):
        """Initialize LLM client.

        Args:
            host: Optional host address
            port: Optional port number
            engine: The LLM engine ('ollama' or 'vllm')
            api_key: Optional bearer token. Falls back to LLMFLUX_API_KEY.
                Required to reach an endpoint started by `llmflux serve`, which
                runs vLLM with --api-key and rejects unauthenticated requests.
        """
        self.engine = engine

        # Use env var if set, otherwise use provided host or default
        if host is None:
            if engine.lower() == "ollama":
                host = os.getenv('OLLAMA_HOST', None)
            else:
                host = os.getenv('VLLM_HOST', None)
            
        # If <ENGINE>_HOST contains a full URL, use it directly
        if host and (host.startswith('http://') or host.startswith('https://')):
            self.base_url = host
        else:
            # Otherwise construct URL from host and port
            if host is None:
                host = 'localhost'
                
            if port is None:
                if engine.lower() == "ollama":
                    port_str = os.getenv('OLLAMA_PORT', '11434')
                else:
                    port_str = os.getenv('VLLM_PORT', '11434')
                try:
                    port = int(port_str)
                except ValueError:
                    logger.warning(f"Invalid <ENGINE>_PORT value: {port_str}, using default 11434")
                    port = 11434
            
            self.base_url = f"http://{host}:{port}"
        
        logger.info(f"Connecting to LLM engine at: {self.base_url}")
        self.session = requests.Session()

        # Set on the session so every request (chat, list_models, pull_model)
        # carries it. Empty strings are treated as unset so an exported-but-blank
        # LLMFLUX_API_KEY does not send "Bearer ".
        self.api_key = api_key or os.getenv('LLMFLUX_API_KEY') or None
        if self.api_key:
            self.session.headers['Authorization'] = f"Bearer {self.api_key}"
            logger.debug("Using bearer token authentication")
    
    def list_models(self) -> List[str]:
        """List available models using Ollama's native tags API.
        
        Returns:
            List of available model names
        """
        url = f"{self.base_url}/api/tags"
        try:
            logger.debug(f"Listing models from: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            data = response.json()
            # Extract model names from the Ollama native API response
            if 'models' in data:
                return [model['name'] for model in data.get('models', [])]
            else:
                logger.error(f"Unexpected response format from API: {data}")
                return []
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error listing models: {e}")
            return []
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error parsing API response: {e}")
            return []
    
    def _list_models_native(self) -> List[str]:
        """DEPRECATED: Use list_models() directly instead.
        This method is kept temporarily for backwards compatibility.
        """
        return self.list_models()
        
    def model_exists(self, model_name: str) -> bool:
        """Check if a model exists.
        
        For model names in format like "llama3.2:3b", this method extracts 
        the base model name (e.g., "llama3.2") to check with Ollama API.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model exists, False otherwise
        """
        # For vLLM, the model is loaded at server startup, so we don't need to check.
        if self.engine == 'vllm':
            logger.debug(f"Running with vLLM engine, skipping model existence check for '{model_name}'.")
            return True

        # Extract base model name before colon
        # base_model = model_name
        # if ":" in model_name:
        #     base_model = model_name.split(":")[0]
        
        logger.debug(f"Checking if base model '{model_name}' exists")
        models = self.list_models()
        exists = model_name in models
        return exists
    
    def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry.
        
        For model names in format like "llama3.2:3b", this method extracts 
        the base model name (e.g., "llama3.2") to use with Ollama API.
        
        Args:
            model_name: Name of the model to pull
            
        Returns:
            True if pull was successful, False otherwise
        """
        # Extract base model name before colon
            
        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name}
        
        try:
            logger.info(f"Pulling model {model_name} from {url}")
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            logger.info(f"Successfully pulled model {model_name}")
            return True
        except requests.exceptions.RequestException as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    def ensure_model_available(self, model_name: str, max_retries: int = 3) -> bool:
        """Ensure a model is available, pulling it if necessary.
        
        Args:
            model_name: Name of the model to ensure
            max_retries: Maximum number of pull attempts
            
        Returns:
            True if model is available, False otherwise
        """
        if self.model_exists(model_name):
            logger.info(f"Model '{model_name}' is already available")
            return True
            
        # Extract base model name for logging clarity
        # base_model = model_name
        # if ":" in model_name:
        #     base_model = model_name.split(":")[0]
            
        logger.info(f"Model '{model_name}' not found, attempting to pull")
        
        for attempt in range(1, max_retries + 1):
            logger.info(f"Pull attempt {attempt}/{max_retries} for model '{model_name}'")
            
            if self.pull_model(model_name):
                logger.info(f"Successfully pulled model '{model_name}'")
                return True

            logger.warning(f"Failed to pull model '{model_name}' (attempt {attempt}/{max_retries})")
            time.sleep(2 ** attempt)  # Exponential backoff
            
        logger.error(f"Failed to pull model '{model_name}' after {max_retries} attempts")
        return False
    
    def chat(
        self,
        model: str,
        engine: str,
        messages: List[Dict[str, Any]],
        return_usage: bool = False,
        **kwargs
    ):
        """Generate response using OpenAI-compatible chat completions endpoint.

        Args:
            model: Name of the model to use
            messages: Array of messages in OpenAI format
            engine: Either 'ollama' or 'vllm'
            return_usage: If True, return a (content, usage_dict) tuple instead
                of a bare string. usage_dict is the OpenAI-style "usage" object
                from the response (empty dict if the server didn't provide one).
            **kwargs: Additional model parameters:
                - temperature: float
                - top_p: float
                - max_tokens: int
                - stop: List[str]

        Returns:
            Model response text, or (text, usage_dict) if return_usage is True

        Raises:
            requests.exceptions.RequestException: If API call fails
            ValueError: If model is not available
        """
        # Ensure model is available for ollama (not necessary for vllm)
        if engine == 'ollama':
            if not self.ensure_model_available(model):
                error_msg = f"Model {model} is not available and could not be pulled"
                logger.error(error_msg)
                raise ValueError(error_msg)
        
        url = f"{self.base_url}/v1/chat/completions"
        
        # Create payload in OpenAI format
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        
        # Add other parameters if provided
        if "temperature" in kwargs:
            payload["temperature"] = kwargs["temperature"]
        if "top_p" in kwargs:
            payload["top_p"] = kwargs["top_p"]
        if "max_tokens" in kwargs:
            payload["max_tokens"] = kwargs["max_tokens"]
        if "stop" in kwargs and kwargs["stop"]:
            payload["stop"] = kwargs["stop"]
        
        try:
            response = self.session.post(url, json=payload)
            response.raise_for_status()
            
            # Parse OpenAI response format
            response_data = response.json()
            
            # Extract the content from the response
            usage = response_data.get('usage') or {}
            if (
                'choices' in response_data and
                len(response_data['choices']) > 0 and
                'message' in response_data['choices'][0] and
                'content' in response_data['choices'][0]['message']
            ):
                content = response_data['choices'][0]['message']['content']
            else:
                logger.warning(f"Unexpected response format: {response_data}")
                content = ""

            return (content, usage) if return_usage else content

        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status in (401, 403) and not self.api_key:
                logger.error(
                    f"Authentication failed (HTTP {status}) and no API key is set. "
                    f"Endpoints started with `llmflux serve` require one: pass "
                    f"api_key=... or set LLMFLUX_API_KEY. Get the key with "
                    f"`llmflux connect <job_id>`."
                )
            else:
                logger.error(f"Error generating response: {e}")
            raise
        except ValueError as e:
            logger.error(f"Error decoding response: {e}")
            return (response.text, {}) if return_usage else response.text