#!/usr/bin/env python3
"""Batch processor for AI-Flux."""

import argparse
import datetime
import json
import logging
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..converters.utils import read_jsonl
from ..core.client import LLMClient
from ..core.config import Config, ModelConfig
from ..io.base import OutputHandler, OutputResult

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Processor for batch processing JSONL inputs with an LLM client."""

    def __init__(
        self,
        model_config: ModelConfig,
        batch_size: int = 4,
        save_frequency: int = 50,
        temp_dir: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        output_handler: Optional[OutputHandler] = None,
    ):
        self.model_config = model_config
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.output_handler = output_handler

        self.client: Optional[LLMClient] = None
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.temp_file = os.path.join(self.temp_dir, f"aiflux_{int(time.time())}.json")
        os.makedirs(self.temp_dir, exist_ok=True)

    def setup(self) -> None:
        """Initialize LLM client and warm up the model."""
        self.client = LLMClient(engine=self.model_config.engine)

        try:
            self.client.generate(
                model=self.model_config.get_model_name_for_engine(),
                messages=[{"role": "user", "content": "Hello, world!"}],
                temperature=self.model_config.parameters.temperature,
                max_tokens=5,
                top_p=self.model_config.parameters.top_p,
                stop=None,
            )
        except Exception as e:
            # Warmup failures shouldn't prevent processing; the first real request
            # will surface any persistent connectivity/model issues.
            logger.warning(f"Model warmup failed: {e}")

    def cleanup(self) -> None:
        """Clean up resources."""
        if not self.client:
            return
        session = getattr(self.client, "session", None)
        close = getattr(session, "close", None)
        if callable(close):
            close()
        self.client = None

    def process_batch(self, batch: List[Dict[str, Any]]) -> List[OutputResult]:
        """Process a batch of JSONL items."""
        if not self.client:
            raise RuntimeError("BatchProcessor.setup() must be called before process_batch().")

        results: List[OutputResult] = []

        for item in batch:
            try:
                url = item.get("url", "/v1/chat/completions")
                body = item.get("body", {}) or {}
                metadata = item.get("metadata", {}) or {}

                if url == "/v1/chat/completions":
                    output = self._process_chat_completion(body)
                elif url == "/v1/completions":
                    output = self._process_completion(body)
                else:
                    raise ValueError(f"Unsupported URL: {url}")

                results.append(
                    OutputResult(
                        input=item,
                        output=output,
                        metadata={
                            "model": self.model_config.get_model_name_for_engine(),
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                            **metadata,
                        },
                    )
                )
            except Exception as e:
                results.append(
                    OutputResult(
                        input=item,
                        output=None,
                        error=str(e),
                        metadata={
                            "model": self.model_config.get_model_name_for_engine(),
                            "timestamp": datetime.datetime.utcnow().isoformat(),
                            "error": True,
                            **(item.get("metadata", {}) or {}),
                        },
                    )
                )

        return results

    def _process_chat_completion(self, body: Dict[str, Any]) -> Dict[str, Any]:
        messages = body.get("messages", []) or []
        model = body.get("model", self.model_config.get_model_name_for_engine())

        temperature = body.get("temperature", self.model_config.parameters.temperature)
        max_tokens = body.get("max_tokens", self.model_config.parameters.max_tokens)
        top_p = body.get("top_p", self.model_config.parameters.top_p)
        stop = body.get("stop", self.model_config.parameters.stop_sequences)

        text = self.client.generate(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )

        return {
            "id": str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }
            ],
        }

    def _process_completion(self, body: Dict[str, Any]) -> Dict[str, Any]:
        prompt = body.get("prompt", "") or ""
        model = body.get("model", self.model_config.get_model_name_for_engine())

        temperature = body.get("temperature", self.model_config.parameters.temperature)
        max_tokens = body.get("max_tokens", self.model_config.parameters.max_tokens)
        top_p = body.get("top_p", self.model_config.parameters.top_p)
        stop = body.get("stop", self.model_config.parameters.stop_sequences)

        text = self.client.generate(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop,
        )

        return {
            "id": str(uuid.uuid4()),
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"text": text, "index": 0, "finish_reason": "stop"}],
        }

    def run(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        **_kwargs: Any,
    ) -> List[OutputResult]:
        """Run batch processing on a JSONL input file."""
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        self.setup()
        try:
            all_results: List[OutputResult] = []
            current_batch: List[Dict[str, Any]] = []

            for item in read_jsonl(str(input_path)):
                current_batch.append(item)
                if len(current_batch) >= self.batch_size:
                    all_results.extend(self.process_batch(current_batch))
                    current_batch = []
                    if len(all_results) % self.save_frequency == 0:
                        self._save_intermediate_results(all_results)

            if current_batch:
                all_results.extend(self.process_batch(current_batch))

            self._save_results(all_results, str(output_path))
            return all_results
        finally:
            self.cleanup()

    def _save_intermediate_results(self, results: List[OutputResult]) -> None:
        try:
            with open(self.temp_file, "w") as f:
                json.dump([r.to_dict() for r in results], f)
        except Exception as e:
            logger.error(f"Error saving intermediate results: {e}")

    def _save_results(self, results: List[OutputResult], output_path: str) -> None:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if self.output_handler:
            self.output_handler.save(results, output_path)
            return

        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)

        if os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except OSError:
                pass


def _load_model_config(model_name: str, engine: Optional[str]) -> ModelConfig:
    # If the name looks like "type:size" and templates exist, try loading from Config.
    if ":" in model_name:
        try:
            model_type, model_size = model_name.split(":", 1)
            config = Config()
            model_config = config.load_model_config(model_type, model_size)
            if engine:
                model_config.engine = engine
            return model_config
        except Exception:
            pass

    model_config = ModelConfig(name=model_name)
    if engine:
        model_config.engine = engine
    return model_config


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m aiflux.processors.batch")
    parser.add_argument("--input", required=True, dest="input_path")
    parser.add_argument("--output", required=True, dest="output_path")
    parser.add_argument("--model", required=True, dest="model_name")
    parser.add_argument("--engine", default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--save-frequency", type=int, default=50)
    args = parser.parse_args(argv)

    model_config = _load_model_config(args.model_name, args.engine)
    processor = BatchProcessor(
        model_config=model_config,
        batch_size=args.batch_size,
        save_frequency=args.save_frequency,
    )
    processor.run(args.input_path, args.output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
