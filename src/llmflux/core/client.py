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

    #: Request fields LLMFlux sets itself, beyond model and messages (which are
    #: parameters). A caller's value for one of these is dropped: this client
    #: reads one non-streamed JSON response, and vLLM rejects stream_options
    #: without stream.
    CONTROLLED_FIELDS = frozenset({"stream", "stream_options"})

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
        # carries it. Blank and whitespace-only values are treated as unset so an
        # exported-but-blank LLMFLUX_API_KEY does not send "Bearer ". Surrounding
        # whitespace is stripped: a quoted .env value keeps its trailing space, and
        # vLLM compares the header byte-for-byte, so " abc" would 401. A trailing
        # newline would raise ValueError from requests before the request is sent.
        candidates = (api_key, os.getenv('LLMFLUX_API_KEY'))
        self.api_key = next((c.strip() for c in candidates if c and c.strip()), None)
        if self.api_key:
            self.session.headers['Authorization'] = f"Bearer {self.api_key}"
            logger.debug("Using bearer token authentication")

        # Every request needs a timeout. Without one, requests blocks on a
        # socket read indefinitely: if the engine dies after the job's health
        # check passed, the batch processor waits for the rest of the job's
        # walltime and the job is billed for doing nothing. A timeout turns that
        # into a per-item failure that BatchProcessor's existing retry and error
        # accounting can see and report.
        #
        # (connect, read). The read budget is generous because generating a long
        # completion legitimately takes minutes; it only has to be finite.
        self.timeout = (
            float(os.getenv("LLMFLUX_CONNECT_TIMEOUT", "10")),
            float(os.getenv("LLMFLUX_READ_TIMEOUT", "600")),
        )
    
    def list_models(self) -> List[str]:
        """List available models using Ollama's native tags API.
        
        Returns:
            List of available model names
        """
        url = f"{self.base_url}/api/tags"
        try:
            logger.debug(f"Listing models from: {url}")
            response = self.session.get(url, timeout=self.timeout)
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
            response = self.session.post(url, json=payload, timeout=self.timeout)
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
    
    def _chat_payload(self, model: str, messages: List[Dict[str, Any]], params: Dict[str, Any]) -> Dict[str, Any]:
        """Build a /v1/chat/completions request: every param is sent as a field,
        except CONTROLLED_FIELDS and an empty stop. model and messages always
        come from the arguments, never from params."""
        payload = {
            key: value for key, value in params.items()
            if key not in self.CONTROLLED_FIELDS and not (key == "stop" and not value)
        }
        payload.update(model=model, messages=messages, stream=False)
        return payload

    @staticmethod
    def _engine_error_message(response: Optional[requests.Response]) -> Optional[str]:
        """The engine's explanation of an HTTP error, from the response body.

        vLLM 0.15 sends {"error": {"message": ...}} ({"error": "Unauthorized"}
        for a bad key); older vLLM sent {"object": "error", "message": ...}.
        Anything else falls back to the body text, truncated.
        """
        if response is None:
            return None
        try:
            data = response.json()
        except ValueError:
            data = None
        if isinstance(data, dict):
            error = data.get("error")
            if isinstance(error, dict) and isinstance(error.get("message"), str):
                return error["message"]
            if isinstance(error, str):
                return error
            if isinstance(data.get("message"), str):
                return data["message"]
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()[:1000]
        return None

    def _post_chat(self, engine: str, model: str, payload: Dict[str, Any]) -> requests.Response:
        """POST a chat request; return the response, raising on HTTP errors."""
        # Ensure model is available for ollama (not necessary for vllm)
        if engine == 'ollama':
            if not self.ensure_model_available(model):
                error_msg = f"Model {model} is not available and could not be pulled"
                logger.error(error_msg)
                raise ValueError(error_msg)

        url = f"{self.base_url}/v1/chat/completions"
        try:
            response = self.session.post(url, json=payload, timeout=self.timeout)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            error = e
            if isinstance(e, requests.exceptions.HTTPError):
                # raise_for_status() says only "400 Client Error: Bad Request";
                # the reason (a bad response_format, an unknown tool) is in the body.
                detail = self._engine_error_message(e.response)
                if detail:
                    error = requests.exceptions.HTTPError(f"{e}: {detail}", request=e.request, response=e.response)
            status = getattr(getattr(e, 'response', None), 'status_code', None)
            if status in (401, 403) and not self.api_key:
                logger.error(
                    f"Authentication failed (HTTP {status}) and no API key is set. "
                    f"Endpoints started with `llmflux serve` require one: pass "
                    f"api_key=... or set LLMFLUX_API_KEY. Get the key with "
                    f"`llmflux connect <job_id>`."
                )
            else:
                logger.error(f"Error generating response: {error}")
            if error is e:
                raise
            raise error from e

    def chat_completion(
        self,
        model: str,
        engine: str,
        messages: List[Dict[str, Any]],
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a chat completion request and return the engine's response as received.

        Unlike chat(), nothing is extracted or rebuilt: every choice, its
        finish_reason, usage and any engine-specific fields come back intact.

        Args:
            model: Name of the model to use
            engine: Either 'ollama' or 'vllm'
            messages: Array of messages in OpenAI format
            params: Request fields, sent as given — temperature, max_tokens,
                response_format, seed, n, tools, vLLM's structured_outputs, and so on.
                A dict rather than keyword arguments, so a field can never collide
                with this method's own parameter names. model, messages, stream
                and stream_options are set by LLMFlux and a value given here is
                ignored; an empty stop is not sent.

        Returns:
            The decoded JSON response

        Raises:
            requests.exceptions.RequestException: If the API call fails
            ValueError: If the model is not available, or the response is not JSON
        """
        response = self._post_chat(engine, model, self._chat_payload(model, messages, params or {}))
        return response.json()

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
            **kwargs: Request fields, sent as given (temperature, top_p,
                max_tokens, top_k, stop, response_format, seed, ...); see
                chat_completion(). Use chat_completion() for the full
                response, including finish_reason.

        Returns:
            Model response text, or (text, usage_dict) if return_usage is True

        Raises:
            requests.exceptions.RequestException: If API call fails
            ValueError: If model is not available
        """
        response = self._post_chat(engine, model, self._chat_payload(model, messages, kwargs))
        try:
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

        except ValueError as e:
            logger.error(f"Error decoding response: {e}")
            return (response.text, {}) if return_usage else response.text
