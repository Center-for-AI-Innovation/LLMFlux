"""The batch stage keeps batch_size requests in flight (LLMFlux#152).

It used to send them one at a time, so a vLLM server, built to batch many
concurrent requests on the GPU, served one. These tests run the real
BatchProcessor and LLMClient against a local endpoint that records how many
requests it is serving at once and over which connections, and can delay or
fail individual requests.
"""

import json
import logging
import os
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from llmflux.core.client import LLMClient
from llmflux.core.config import ModelConfig, ModelParameters
from llmflux.processors.batch import BatchProcessor

MODEL = "test/test-model"
WARMUP = "Hello, world!"  # BatchProcessor.setup()'s warmup prompt


class Endpoint:
    """A local /v1/chat/completions that answers each prompt with itself.

    `behave(prompt, attempt)` returns (delay_seconds, http_status) for each
    request; attempt counts from 0 per prompt. Requests are held open for the
    delay, so in-flight counts are real.
    """

    def __init__(self, behave=lambda prompt, attempt: (0.0, 200)):
        self.behave = behave
        self.in_flight = 0
        self.max_in_flight = 0
        self.attempts = {}
        self.client_ports = set()
        self._lock = threading.Lock()

    def __enter__(self):
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"  # keep-alive, so connection reuse is visible

            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                prompt = body["messages"][-1]["content"]
                with endpoint._lock:
                    attempt = endpoint.attempts.get(prompt, 0)
                    endpoint.attempts[prompt] = attempt + 1
                    endpoint.client_ports.add(self.client_address[1])
                    endpoint.in_flight += 1
                    endpoint.max_in_flight = max(endpoint.max_in_flight, endpoint.in_flight)
                delay, status = endpoint.behave(prompt, attempt)
                time.sleep(delay)
                with endpoint._lock:
                    endpoint.in_flight -= 1
                reply = {
                    "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": body["model"],
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": prompt},
                                 "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                }
                data = json.dumps(reply if status == 200 else {"error": "boom"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            request_queue_size = 128  # the default backlog of 5 stalls 32 simultaneous connects

        self._server = Server(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        return self

    def __exit__(self, *exc):
        self._server.shutdown()
        self._server.server_close()
        self._thread.join()


class TestConcurrentRequests(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.model_config = ModelConfig(
            name="test:7b", hf_name=MODEL, engine="vllm",
            parameters=ModelParameters(temperature=0.7, max_tokens=16, top_p=0.9, top_k=40),
        )

    def _run(self, endpoint, n_items, **processor_kwargs):
        prompts = [f"item-{i}" for i in range(n_items)]
        input_path = self.root / "in.jsonl"
        with open(input_path, "w") as f:
            for i, prompt in enumerate(prompts):
                f.write(json.dumps({"custom_id": f"c{i}", "body": {
                    "messages": [{"role": "user", "content": prompt}]}}) + "\n")
        processor_kwargs.setdefault("retry_delay", 0)
        with endpoint, patch.dict(os.environ, {"VLLM_HOST": endpoint.url}):
            processor = BatchProcessor(model_config=self.model_config, **processor_kwargs)
            results = processor.run(str(input_path), str(self.root / "out.json"), "vllm")
        return prompts, results

    def test_batch_size_requests_are_in_flight_at_once(self):
        # Each request is held for 0.2 s, so a sequential processor peaks at 1.
        endpoint = Endpoint(lambda prompt, attempt: (0.0 if prompt == WARMUP else 0.2, 200))
        _, results = self._run(endpoint, 24, batch_size=8)
        self.assertTrue(all(r.error is None for r in results))
        self.assertEqual(endpoint.max_in_flight, 8)

    def test_never_more_than_batch_size_in_flight(self):
        endpoint = Endpoint(lambda prompt, attempt: (0.05, 200))
        self._run(endpoint, 30, batch_size=3)
        self.assertEqual(endpoint.max_in_flight, 3)

    def test_results_keep_input_order_when_completion_order_differs(self):
        # Earlier items take longer, so they finish last.
        def behave(prompt, attempt):
            if prompt == WARMUP:
                return 0.0, 200
            return 0.02 * (12 - int(prompt.split("-")[1])), 200

        prompts, results = self._run(Endpoint(behave), 12, batch_size=12)
        self.assertEqual([r.input["custom_id"] for r in results], [f"c{i}" for i in range(12)])
        self.assertEqual([r.output["choices"][0]["message"]["content"] for r in results], prompts)
        saved = json.loads((self.root / "out.json").read_text())["results"]
        self.assertEqual([row["input"]["custom_id"] for row in saved], [f"c{i}" for i in range(12)])

    def test_retries_stay_per_request(self):
        # item-3 fails twice then succeeds; item-5 always fails. The others
        # must be unaffected, and each keeps its own retry count.
        def behave(prompt, attempt):
            if prompt == "item-3" and attempt < 2:
                return 0.0, 503
            if prompt == "item-5":
                return 0.0, 500
            return 0.01, 200

        endpoint = Endpoint(behave)
        _, results = self._run(endpoint, 8, batch_size=4, max_retries=2)
        by_id = {r.input["custom_id"]: r for r in results}
        self.assertIsNone(by_id["c3"].error)
        self.assertEqual(by_id["c3"].metadata["retry_count"], 2)
        self.assertIn("500", by_id["c5"].error)
        self.assertEqual(by_id["c5"].metadata["retry_count"], 2)
        for cid in ("c0", "c1", "c2", "c4", "c6", "c7"):
            self.assertIsNone(by_id[cid].error)
            self.assertEqual(by_id[cid].metadata["retry_count"], 0)
        self.assertEqual(endpoint.attempts["item-3"], 3)
        self.assertEqual(endpoint.attempts["item-5"], 3)

    def test_connection_pool_holds_every_worker(self):
        """requests keeps 10 connections per host. With 32 workers on one
        session, urllib3 discards the surplus with "Connection pool is full,
        discarding connection" (22 times in this run at the default size)."""
        endpoint = Endpoint(lambda prompt, attempt: (0.1, 200))
        discards = []

        class Recorder(logging.Handler):
            def emit(self, record):
                if "Connection pool is full" in record.getMessage():
                    discards.append(record)

        pool_logger = logging.getLogger("urllib3.connectionpool")
        recorder = Recorder(level=logging.WARNING)
        pool_logger.addHandler(recorder)
        self.addCleanup(pool_logger.removeHandler, recorder)
        self._run(endpoint, 160, batch_size=32)
        self.assertEqual(endpoint.max_in_flight, 32)  # the pool really was contended
        self.assertEqual(discards, [])

    def test_intermediate_saves_follow_save_frequency(self):
        """Saves used to trigger only at multiples of lcm(batch_size,
        save_frequency): at 4 and 5, never in a 12-item run."""
        saved_counts = []
        with patch.object(BatchProcessor, "_save_intermediate_results",
                          lambda self, results, path: saved_counts.append(len(results))):
            self._run(Endpoint(), 12, batch_size=4, save_frequency=5)
        self.assertEqual(saved_counts, [5, 10])

    def test_batch_size_below_one_is_rejected(self):
        for bad in (0, -1):
            with self.subTest(batch_size=bad):
                with self.assertRaises(ValueError):
                    BatchProcessor(model_config=self.model_config, batch_size=bad)


class TestClientPoolSize(unittest.TestCase):

    def test_pool_holds_at_least_pool_size_connections(self):
        for size, expected in ((32, 32), (4, 10)):
            with self.subTest(size=size):
                client = LLMClient(engine="vllm", host="localhost", port=1, pool_size=size)
                for prefix in ("http://", "https://"):
                    self.assertEqual(client.session.get_adapter(prefix)._pool_maxsize, expected)


if __name__ == "__main__":
    unittest.main()
