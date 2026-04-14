#!/usr/bin/env python3
"""Batch processor for LLMFlux."""

import logging
import os
import time
import datetime
import uuid
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Generator, Union
import tempfile

from ..core.client import LLMClient
from ..core.config import ModelConfig
from ..io.base import OutputHandler, OutputResult
from ..converters.utils import read_jsonl
from ..benchmark_utils import VllmMetricsScraper, GpuUtilScraper

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO').upper(),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    """Processor for batch processing JSONL inputs with LLM."""
    
    def __init__(
        self,
        model_config: ModelConfig,
        batch_size: int = 4,
        save_frequency: int = 50,
        temp_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        output_handler: Optional[OutputHandler] = None
    ):
        """Initialize batch processor.
        
        Args:
            model_config: Model configuration
            batch_size: Number of items to process in a batch
            save_frequency: How often to save intermediate results (items)
            temp_dir: Directory for storing temporary files
            max_retries: Maximum number of retry attempts for failed items
            retry_delay: Delay between retry attempts in seconds
            output_handler: Optional output handler
        """
        self.model_config = model_config
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.output_handler = output_handler
        self.client = None
        self._vllm_metrics: Dict[str, Any] = {}
        self._gpu_metrics: Dict[str, Any] = {}
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.temp_file = os.path.join(self.temp_dir, f"llmflux_{int(time.time())}.jsonl")
        
        # Ensure temp directory exists
        os.makedirs(self.temp_dir, exist_ok=True)
    
    def setup(self, engine: str = "ollama"):
        """Initialize LLM client and warm up model."""
        # Initialize client, passing the engine from the model config
        logger.info("Initializing LLM client")
        self.client = LLMClient(engine=self.model_config.engine)
                
        # Get the appropriate model name for this engine
        model = self.model_config.get_model_name_for_engine()
        logger.info(f"Warming up model: {model}")
        
        try:
            # Simple warmup query to ensure model is loaded
            warmup_messages = [{"role": "user", "content": "Hello, world!"}]
            self.client.chat(
                model=model,
                engine=engine,
                messages=warmup_messages,
                max_tokens=5
            )
            logger.info(f"Model {model} warmed up successfully")
        except Exception as e:
            logger.error(f"Error warming up model: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        if self.client:
            logger.info("Closing client session")
            self.client.session.close()
            self.client = None
    
    def process_batch(self, engine: str, batch: List[Dict[str, Any]]) -> List[OutputResult]:
        """Process a batch of JSONL items.
        
        Args:
            engine: Engine used for this run
            batch: List of parsed JSONL items
            
        Returns:
            List of processed output results
        """
        results = []
        
        for item in batch:
            try:
                # Extract request details from JSONL item
                custom_id = item.get('custom_id', str(uuid.uuid4()))
                method = item.get('method', 'POST')
                url = item.get('url', '/v1/chat/completions')
                body = item.get('body', {})
                metadata = item.get('metadata', {})
                
                # Process with client based on URL endpoint
                if url == '/v1/chat/completions':
                    response = self._process_chat_completion(engine, body)
                elif url == '/v1/completions':
                    response = self._process_completion(engine, body)
                else:
                    raise ValueError(f"Unsupported URL: {url}")
                
                # Build result
                result = OutputResult(
                    input=item,
                    output=response,
                    metadata={
                        "model": self.model_config.get_model_name_for_engine(),
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        **metadata
                    }
                )
                results.append(result)
                logger.debug(f"Processed item with ID: {custom_id}")
                
            except Exception as e:
                logger.error(f"Error processing item: {e}")
                # Add error result
                result = OutputResult(
                    input=item,
                    output=None,
                    error=str(e),
                    metadata={
                        "model": self.model_config.get_model_name_for_engine(),
                        "timestamp": datetime.datetime.utcnow().isoformat(),
                        "error": True,
                        **item.get("metadata", {})
                    }
                )
                results.append(result)
        
        return results
    
    def _process_chat_completion(self, engine: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Process chat completion request.
        
        Args:
            engine: engine used for this run
            body: Request body
            
        Returns:
            Chat completion response
        """
        messages = body.get('messages', [])
        # Use the engine-appropriate model name
        model = body.get('model', self.model_config.get_model_name_for_engine())
        
        # Extract parameters with defaults from model config
        temperature = body.get('temperature', self.model_config.parameters.temperature)
        max_tokens = body.get('max_tokens', self.model_config.parameters.max_tokens)
        top_p = body.get('top_p', self.model_config.parameters.top_p)
        stop = body.get('stop', self.model_config.parameters.stop_sequences)
        
        # Generate response
        response = self.client.chat(
            model=model,
            engine=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        
        # Format in OpenAI-compatible format
        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response
                    },
                    "finish_reason": "stop"
                }
            ]
        }
    
    def _process_completion(self, engine: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Process completion request.

        Args:
            engine: Engine used for this run
            body: Request body

        Returns:
            Completion response
        """
        prompt = body.get('prompt', '')
        model = body.get('model', self.model_config.get_model_name_for_engine())

        # Extract parameters with defaults from model config
        temperature = body.get('temperature', self.model_config.parameters.temperature)
        max_tokens = body.get('max_tokens', self.model_config.parameters.max_tokens)
        top_p = body.get('top_p', self.model_config.parameters.top_p)
        stop = body.get('stop', self.model_config.parameters.stop_sequences)

        # Convert prompt to messages
        messages = [{"role": "user", "content": prompt}]

        # Generate response
        response = self.client.chat(
            model=model,
            engine=engine,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop
        )
        
        # Format in OpenAI-compatible format
        return {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "text": response,
                    "index": 0,
                    "finish_reason": "stop"
                }
            ]
        }
    
    def run(self, input_path: str, output_path: str, engine: str, **kwargs) -> List[Dict[str, Any]]:
        """Run batch processing on JSONL input file.
        
        Args:
            input_path: Path to JSONL input file
            output_path: Path to save output results
            engine: the engine being used for this run
            **kwargs: Additional parameters (ignored for compatibility)
            
        Returns:
            List of results
        """
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        self.setup(engine)
        print("Setup complete")

        # Start vLLM metrics scraper (vLLM only — Ollama doesn't expose the same endpoint)
        scraper = None
        if engine == "vllm":
            scraper = VllmMetricsScraper(base_url=self.client.base_url)
            scraper.start()

        # GPU utilization scraper runs for all engines
        gpu_scraper = GpuUtilScraper()
        gpu_scraper.start()

        try:
            # Process JSONL file
            all_results = []
            current_batch = []
            processed_count = 0

            # Read JSONL file line by line
            for item in read_jsonl(input_path):
                current_batch.append(item)

                # Process batch when it reaches batch size
                if len(current_batch) >= self.batch_size:
                    print("processing batch")
                    batch_results = self.process_batch(engine, current_batch)
                    all_results.extend(batch_results)
                    current_batch = []

                    # Save intermediate results
                    processed_count += len(batch_results)
                    if processed_count % self.save_frequency == 0:
                        self._save_intermediate_results(all_results, output_path)
                        logger.info(f"Processed {processed_count} items")

            # Process remaining items
            if current_batch:
                print("processing  batch")
                batch_results = self.process_batch(engine, current_batch)
                all_results.extend(batch_results)
                processed_count += len(batch_results)
                logger.info(f"Processed {processed_count} items")

            # Stop scrapers before saving so metrics are included in output
            if scraper:
                self._vllm_metrics = scraper.stop()
            self._gpu_metrics = gpu_scraper.stop()

            # Save final results
            self._save_results(all_results, output_path)
            logger.info(f"Processing complete. Results saved to {output_path}")

            return all_results

        finally:
            if scraper and scraper._thread and scraper._thread.is_alive():
                scraper.stop()
            if gpu_scraper._thread and gpu_scraper._thread.is_alive():
                gpu_scraper.stop()
            self.cleanup()
    
    def _save_intermediate_results(self, results: List[OutputResult], output_path: str):
        """Save intermediate results to a temporary file.
        
        Args:
            results: List of results to save
            output_path: Path to the final output file
        """
        try:
            # Convert results to serializable format
            serializable_results = [result.to_dict() for result in results]
            
            # Write to temporary file
            with open(self.temp_file, 'w') as f:
                json.dump(serializable_results, f)
            
            logger.info(f"Saved intermediate results to {self.temp_file}")
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")
    
    def _save_results(self, results: List[OutputResult], output_path: str):
        """Save final results using the output handler.
        
        Args:
            results: List of results to save
            output_path: Path to save the results
        """
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            # Use output handler to save results if provided
            if self.output_handler:
                self.output_handler.save(results, output_path)
            else:
                serializable_results = [result.to_dict() for result in results]
                output = {"results": serializable_results}
                if self._vllm_metrics:
                    output["vllm_metrics"] = self._vllm_metrics
                if self._gpu_metrics:
                    output["gpu_metrics"] = self._gpu_metrics
                with open(output_path, 'w') as f:
                    json.dump(output, f, indent=2)
            
            # Clean up temporary file
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
                logger.debug(f"Removed temporary file: {self.temp_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise 
