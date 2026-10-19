"""Batch requests reach the engine intact, and its responses come back intact.

LLMFlux#154: only six body fields were sent (response_format, seed, guided_*
and the rest were dropped without a word), and every response was rebuilt with
finish_reason hard-coded to "stop", hiding truncation. These tests run the real
BatchProcessor and LLMClient against a local OpenAI-compatible endpoint and
assert on the JSON that crossed the wire in both directions.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmflux.core.config import ModelConfig, ModelParameters
from llmflux.processors.batch import BatchProcessor

from .fake_engine import FakeEngine, REPLY_WORDS

MODEL = "test/test-model"

#: OpenAI and vLLM request fields LLMFlux has no reason to touch.
PASSTHROUGH = {
    "response_format": {
        "type": "json_schema",
        "json_schema": {"name": "x", "schema": {"type": "object", "properties": {"a": {"type": ["string", "null"]}}}},
    },
    "seed": 1234,
    "n": 2,
    "logprobs": True,
    "top_logprobs": 2,
    "frequency_penalty": 0.5,
    "presence_penalty": -0.5,
    "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
    "structured_outputs": {"choice": ["yes", "no"]},
    # Not fields in the container's vLLM 0.15.0 (it ignores them with a
    # warning), but forwarding is not LLMFlux's call to make.
    "guided_choice": ["yes", "no"],
    "max_completion_tokens": 50,
    "repetition_penalty": 1.1,
}


class TestRequestPassthrough(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_config = ModelConfig(
            name="test:7b", hf_name=MODEL, engine="vllm",
            parameters=ModelParameters(temperature=0.7, max_tokens=500, top_p=0.9, top_k=40),
        )

    def _run(self, items, engine=None, **processor_kwargs):
        """Run items through BatchProcessor against a FakeEngine.

        Returns (results, the request bodies the engine received for the items).
        """
        input_path = self.root / "in.jsonl"
        with open(input_path, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")
        engine = engine or FakeEngine()
        with engine, patch.dict(os.environ, {"VLLM_HOST": engine.url}):
            processor = BatchProcessor(model_config=self.model_config, retry_delay=0, **processor_kwargs)
            results = processor.run(str(input_path), str(self.root / "out.json"), "vllm")
        self.assertEqual(len(engine.requests), len(items) + 1, "one warmup, then one per item")
        return results, engine.requests[1:]

    def test_remaining_body_fields_reach_the_engine(self):
        messages = [{"role": "user", "content": "extract"}]
        body = {"model": MODEL, "messages": messages, "max_tokens": 20, **PASSTHROUGH}
        [result], [sent] = self._run([{"custom_id": "a", "body": body}])
        self.assertIsNone(result.error)
        for key, value in PASSTHROUGH.items():
            self.assertEqual(sent.get(key), value, key)
        self.assertEqual(sent["model"], MODEL)
        self.assertEqual(sent["messages"], messages)
        self.assertEqual(sent["max_tokens"], 20)        # the body's own value
        self.assertEqual(sent["temperature"], 0.7)      # model-config fallback, as before
        self.assertIs(sent["stream"], False)

    def test_llmflux_keeps_control_of_stream(self):
        body = {"messages": [], "stream": True, "stream_options": {"include_usage": True}}
        [result], [sent] = self._run([{"custom_id": "a", "body": body}])
        self.assertIsNone(result.error)
        self.assertIs(sent["stream"], False)
        self.assertNotIn("stream_options", sent)

    def test_body_model_must_still_match(self):
        """Passing fields through does not pass `model` through unchecked."""
        body = {"model": "some/other-model", "messages": []}
        input_path = self.root / "in.jsonl"
        input_path.write_text(json.dumps({"custom_id": "a", "body": body}) + "\n")
        with FakeEngine() as engine, patch.dict(os.environ, {"VLLM_HOST": engine.url}):
            processor = BatchProcessor(model_config=self.model_config, retry_delay=0, max_retries=0)
            [result] = processor.run(str(input_path), str(self.root / "out.json"), "vllm")
        self.assertIn("does not match requested model", result.error)
        self.assertEqual(len(engine.requests), 1)  # the warmup only

    def test_truncated_reply_reports_length(self):
        [cut, whole], _ = self._run([
            {"custom_id": "cut", "body": {"messages": [], "max_tokens": 3}},
            {"custom_id": "whole", "body": {"messages": [], "max_tokens": 100}},
        ])
        self.assertEqual(cut.output["choices"][0]["finish_reason"], "length")
        self.assertEqual(cut.output["choices"][0]["message"]["content"], " ".join(REPLY_WORDS[:3]))
        self.assertEqual(whole.output["choices"][0]["finish_reason"], "stop")

    def test_response_is_written_as_the_engine_sent_it(self):
        engine = FakeEngine(extra_response_fields={"system_fingerprint": "fp-1", "prompt_logprobs": None})
        [result], [sent] = self._run([{"custom_id": "a", "body": {"messages": [], "n": 2}}], engine=engine)
        self.assertEqual(result.output, engine.respond(sent))
        self.assertEqual(len(result.output["choices"]), 2)  # n=2 used to come back as one
        self.assertEqual(result.output["system_fingerprint"], "fp-1")
        # And the saved file carries the same thing.
        saved = json.loads((self.root / "out.json").read_text())["results"][0]["output"]
        self.assertEqual(saved, result.output)

    def test_completions_endpoint_reports_length_and_forwards_fields(self):
        body = {"prompt": "complete", "max_tokens": 2, "seed": 7, "n": 2}
        [result], [sent] = self._run([{"custom_id": "a", "url": "/v1/completions", "body": body}])
        self.assertIsNone(result.error)
        self.assertEqual(sent["messages"], [{"role": "user", "content": "complete"}])
        self.assertNotIn("prompt", sent)
        self.assertEqual((sent["seed"], sent["n"]), (7, 2))
        output = result.output
        self.assertEqual(output["object"], "text_completion")
        self.assertEqual(
            [(c["index"], c["text"], c["finish_reason"]) for c in output["choices"]],
            [(0, "the quick", "length"), (1, "the quick", "length")],
        )
        self.assertEqual(output["usage"]["completion_tokens"], 4)

    def test_reply_without_choices_is_an_error_not_an_empty_success(self):
        class NoChoices(FakeEngine):
            def respond(self, body):
                if body.get("max_tokens") == 5:  # let setup()'s warmup through
                    return super().respond(body)
                return {"object": "error", "message": "boom"}

        input_path = self.root / "in.jsonl"
        input_path.write_text(json.dumps({"custom_id": "a", "body": {"messages": [], "max_tokens": 10}}) + "\n")
        with NoChoices() as engine, patch.dict(os.environ, {"VLLM_HOST": engine.url}):
            processor = BatchProcessor(model_config=self.model_config, retry_delay=0, max_retries=1)
            [result] = processor.run(str(input_path), str(self.root / "out.json"), "vllm")
        self.assertIsNone(result.output)
        self.assertIn("no choices", result.error)
        self.assertEqual(result.metadata["retry_count"], 1)
        self.assertEqual(len(engine.requests), 3)  # warmup + 2 attempts


if __name__ == "__main__":
    unittest.main()
