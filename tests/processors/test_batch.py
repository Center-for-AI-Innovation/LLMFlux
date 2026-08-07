"""Tests for the BatchProcessor class."""

import os
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from llmflux.processors.batch import BatchProcessor
from llmflux.core.config import ModelConfig, ModelParameters
from llmflux.io.base import OutputResult

class TestBatchProcessor(unittest.TestCase):
    """Test suite for the BatchProcessor class."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_dir = Path(self.temp_dir.name)
        
        # Create a test model config
        self.model_params = ModelParameters(
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            top_k=40,
            stop_sequences=None
        )
        
        self.model_config = ModelConfig(
            name="test:7b",
            hf_name="test/test-model",
            parameters=self.model_params,
        )
        
        # Create a test JSONL file
        self.jsonl_path = self.test_dir / "test.jsonl"
        
        # Sample JSONL entries
        self.entries = [
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
                        {"role": "user", "content": "How are you?"}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 100
                }
            }
        ]
        
        with open(self.jsonl_path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")
        
        # Output path
        self.output_path = self.test_dir / "output.json"
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    @patch('llmflux.processors.batch.LLMClient')
    def test_batch_processor_initialization(self, mock_client_class):
        """Test BatchProcessor initialization."""
        processor = BatchProcessor(model_config=self.model_config)
        
        # Check properties
        self.assertEqual(processor.model_config, self.model_config)
        self.assertEqual(processor.batch_size, 4)  # Default value
        self.assertEqual(processor.save_frequency, 50)  # Default value
        self.assertIsNone(processor.client)  # Client initialized in setup
    
    @patch('llmflux.processors.batch.LLMClient')
    def test_batch_processor_setup(self, mock_client_class):
        """Test BatchProcessor setup."""
        # Mock client instance
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        
        processor = BatchProcessor(model_config=self.model_config)
        processor.setup()
        
        # Check that client was created
        self.assertIsNotNone(processor.client)
        mock_client_class.assert_called_once()
        
        # Check that warmup was called
        mock_client.chat.assert_called_once()

    @patch('llmflux.processors.batch.LLMClient')
    def test_setup_forwards_api_key_to_client(self, mock_client_class):
        """The API key must reach LLMClient, or `serve` endpoints answer 401."""
        mock_client_class.return_value = MagicMock()

        processor = BatchProcessor(model_config=self.model_config, api_key="llmflux-abc")
        processor.setup()

        self.assertEqual(mock_client_class.call_args.kwargs["api_key"], "llmflux-abc")

    @patch('llmflux.processors.batch.LLMClient')
    def test_setup_passes_none_api_key_by_default(self, mock_client_class):
        """Without a key, LLMClient falls back to LLMFLUX_API_KEY on its own."""
        mock_client_class.return_value = MagicMock()

        processor = BatchProcessor(model_config=self.model_config)
        processor.setup()

        self.assertIsNone(mock_client_class.call_args.kwargs["api_key"])

    @patch('llmflux.processors.batch.LLMClient')
    def test_process_batch(self, mock_client_class):
        """Test processing a batch of items."""
        # Mock client instance
        mock_client = MagicMock()
        mock_client.chat.return_value = ("This is a test response.", {})
        mock_client_class.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)
        processor.setup()

        # Process batch
        results = processor.process_batch("vllm", self.entries)

        # Check that we have two results
        self.assertEqual(len(results), 2)

        # Check first result
        self.assertEqual(results[0].input, self.entries[0])
        self.assertIsNotNone(results[0].output)
        self.assertIn("This is a test response.", str(results[0].output))

        # Check that chat was called with correct parameters
        mock_client.chat.assert_any_call(
            model="test/test-model",
            engine="vllm",
            messages=self.entries[0]["body"]["messages"],
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            stop=None,
            return_usage=True
        )

    @patch('llmflux.processors.batch.LLMClient')
    def test_process_batch_accepts_matching_jsonl_model(self, mock_client_class):
        """Model in JSONL body is accepted when it matches requested llmflux model."""
        mock_client = MagicMock()
        mock_client.chat.return_value = ("This is a test response.", {})
        mock_client_class.return_value = mock_client

        matching_entry = {
            "custom_id": "test-match-model",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "test/test-model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        }

        processor = BatchProcessor(model_config=self.model_config)
        processor.setup()
        results = processor.process_batch("vllm", [matching_entry])

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].error)
        mock_client.chat.assert_any_call(
            model="test/test-model",
            engine="vllm",
            messages=matching_entry["body"]["messages"],
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            stop=None,
            return_usage=True,
        )

    @patch('llmflux.processors.batch.LLMClient')
    def test_process_batch_rejects_mismatched_jsonl_model(self, mock_client_class):
        """Model mismatch between JSONL body and requested llmflux model returns error."""
        mock_client = MagicMock()
        mock_client.chat.return_value = ("This is a test response.", {})
        mock_client_class.return_value = mock_client

        mismatch_entry = {
            "custom_id": "test-mismatch-model",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "different/model",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        }

        processor = BatchProcessor(model_config=self.model_config)
        processor.setup()
        results = processor.process_batch("vllm", [mismatch_entry])

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].output)
        self.assertIn("does not match requested model", results[0].error)
        self.assertTrue(results[0].metadata.get("error"))
        mock_client.chat.assert_called_once()  # setup() warmup only
    
    @patch('llmflux.processors.batch.LLMClient')
    def test_run_with_jsonl(self, mock_client_class):
        """Test running the processor with a JSONL file."""
        # Mock client instance
        mock_client = MagicMock()
        mock_client.chat.return_value = (
            "This is a test response.",
            {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
        )
        mock_client_class.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)

        # Run processor
        results = processor.run(self.jsonl_path, self.output_path, "vllm")

        # Check results
        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0].error)
        self.assertIsNone(results[1].error)

        # Check that output file was created
        self.assertTrue(os.path.exists(self.output_path))

        # Read output file
        with open(self.output_path, "r") as f:
            output_data = json.load(f)

        # Output is now {"results": [...], "vllm_metrics": {...}}
        rows = output_data["results"]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["input"]["custom_id"], "test-1")
        self.assertEqual(rows[1]["input"]["custom_id"], "test-2")
        self.assertIn("run_metrics", output_data)
        self.assertIn("elapsed_sec", output_data["run_metrics"])
        self.assertIn("request_latency_p95_ms", output_data["run_metrics"])
        self.assertIn("error_rate_by_type_pct", output_data["run_metrics"])
        self.assertIn("retry_rate_pct", output_data["run_metrics"])
        self.assertEqual(output_data["run_metrics"]["output_tokens_avg"], 5)
    
    @patch('llmflux.processors.batch.LLMClient')
    def test_error_handling(self, mock_client_class):
        """Test error handling in processing."""
        # Mock client instance to raise an exception
        mock_client = MagicMock()
        mock_client.chat.side_effect = Exception("Test error")
        mock_client_class.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)
        # Inject client directly to avoid setup()'s warmup call consuming the side_effect
        processor.client = mock_client

        # Process batch
        results = processor.process_batch("vllm", self.entries)
        
        # Check that we have two results with errors
        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0].output)
        self.assertEqual(results[0].error, "Test error")
        self.assertTrue(results[0].metadata.get("error"))
        self.assertEqual(results[0].metadata.get("retry_count"), 3)

    @patch('llmflux.processors.batch.LLMClient')
    def test_compute_run_metrics_includes_retry_and_error_breakdown(self, mock_client_class):
        """Test aggregate run metrics include retry and error taxonomy fields."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)
        results = [
            OutputResult(
                input={"custom_id": "ok-1"},
                output={"usage": {"completion_tokens": 12}},
                metadata={"request_latency_ms": 10.0, "retry_count": 1},
            ),
            OutputResult(
                input={"custom_id": "ok-2"},
                output={"usage": {"completion_tokens": 8}},
                metadata={"request_latency_ms": 20.0, "retry_count": 0},
            ),
            OutputResult(
                input={"custom_id": "bad-1"},
                output=None,
                error="timeout exceeded",
                metadata={"request_latency_ms": 30.0, "retry_count": 2},
            ),
        ]

        metrics = processor._compute_run_metrics(results, elapsed_sec=2.0)
        self.assertEqual(metrics["retry_count_total"], 3)
        self.assertAlmostEqual(metrics["retry_rate_pct"], 66.67, places=2)
        self.assertEqual(metrics["error_rate_by_type_pct"]["timeout"], 33.33)
        self.assertEqual(metrics["output_tokens_avg"], 10)
    
    @patch('llmflux.processors.batch.LLMClient')
    def test_completion_endpoint(self, mock_client_class):
        """Test handling the completions endpoint."""
        # Create a test JSONL with completions endpoint
        completions_jsonl = self.test_dir / "completions.jsonl"
        completion_entry = {
            "custom_id": "completion-1",
            "method": "POST",
            "url": "/v1/completions",
            "body": {
                "prompt": "Complete this sentence: The sky is",
                "temperature": 0.7,
                "max_tokens": 500
            }
        }
        
        with open(completions_jsonl, "w") as f:
            f.write(json.dumps(completion_entry) + "\n")
        
        # Mock client instance
        mock_client = MagicMock()
        mock_client.chat.return_value = ("blue", {})
        mock_client_class.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)

        # Run processor
        results = processor.run(completions_jsonl, self.output_path, "vllm")

        # Check that chat was called with expected parameters
        mock_client.chat.assert_called_with(
            model="test/test-model",
            engine="vllm",
            messages=[{"role": "user", "content": "Complete this sentence: The sky is"}],
            temperature=0.7,
            max_tokens=500,
            top_p=0.9,
            stop=None,
            return_usage=True
        )
        
        # Check output format
        self.assertEqual(len(results), 1)
        output = results[0].output
        self.assertEqual(output["object"], "text_completion")
        self.assertEqual(output["choices"][0]["text"], "blue")

    @patch('llmflux.processors.batch.LLMClient')
    def test_completion_endpoint_rejects_mismatched_model(self, mock_client_class):
        """_get_validated_model raises on mismatch for /v1/completions requests."""
        mock_client = MagicMock()
        mock_client.chat.return_value = "blue"
        mock_client_class.return_value = mock_client

        completions_jsonl = self.test_dir / "completions_mismatch.jsonl"
        entry = {
            "custom_id": "completion-mismatch",
            "method": "POST",
            "url": "/v1/completions",
            "body": {
                "model": "different/model",
                "prompt": "Complete this:",
                "temperature": 0.7,
                "max_tokens": 100,
            },
        }
        with open(completions_jsonl, "w") as f:
            f.write(json.dumps(entry) + "\n")

        processor = BatchProcessor(model_config=self.model_config)
        results = processor.run(completions_jsonl, self.output_path, "vllm")

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].output)
        self.assertIn("does not match requested model", results[0].error)
        self.assertTrue(results[0].metadata.get("error"))

    @patch('llmflux.processors.batch.LLMClient')
    def test_completion_endpoint_accepts_matching_model(self, mock_client_class):
        """_get_validated_model accepts matching model field on /v1/completions."""
        mock_client = MagicMock()
        # Tuple form matches the (content, usage) contract client.chat() returns
        # with return_usage=True, used once benchmarking's batch.py changes land.
        mock_client.chat.return_value = ("blue", {})
        mock_client_class.return_value = mock_client

        completions_jsonl = self.test_dir / "completions_match.jsonl"
        entry = {
            "custom_id": "completion-match",
            "method": "POST",
            "url": "/v1/completions",
            "body": {
                "model": "test/test-model",
                "prompt": "Complete this:",
                "temperature": 0.7,
                "max_tokens": 100,
            },
        }
        with open(completions_jsonl, "w") as f:
            f.write(json.dumps(entry) + "\n")

        processor = BatchProcessor(model_config=self.model_config)
        results = processor.run(completions_jsonl, self.output_path, "vllm")

        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].error)
        self.assertIsNotNone(results[0].output)

    @patch('llmflux.processors.batch.LLMClient')
    def test_get_validated_model_no_model_field_uses_config(self, mock_client_class):
        """_get_validated_model returns the config model name when body has no model key."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)
        result = processor._get_validated_model({})
        self.assertEqual(result, "test/test-model")

    @patch('llmflux.processors.batch.LLMClient')
    def test_get_validated_model_mismatch_raises(self, mock_client_class):
        """_get_validated_model raises ValueError when model field doesn't match config."""
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)
        with self.assertRaises(ValueError) as ctx:
            processor._get_validated_model({"model": "wrong/model"})
        self.assertIn("does not match requested model", str(ctx.exception))


if __name__ == "__main__":
    unittest.main() 
