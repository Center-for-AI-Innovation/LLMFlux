"""Tests for the concurrency load tester.

The end-to-end tests run against a mock vLLM server with a fixed number of
decode slots, so saturation behavior (queueing, latency growth, throughput
plateau) is reproducible without a GPU.
"""

import json
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from llmflux.loadtest import (
    build_payload,
    format_report,
    run_ramp,
    send_request,
    warm_up,
)

API_KEY = "test-key"
SLOTS = 2
TOKENS = 5
TOKEN_DELAY = 0.005


class _MockVllmHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible SSE endpoint with a bounded concurrency."""

    protocol_version = "HTTP/1.1"
    semaphore = threading.BoundedSemaphore(SLOTS)
    state = {"running": 0, "waiting": 0}
    lock = threading.Lock()

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/metrics":
            with self.lock:
                body = (
                    f'vllm:num_requests_running{{model_name="mock"}} {self.state["running"]}\n'
                    f'vllm:num_requests_waiting{{model_name="mock"}} {self.state["waiting"]}\n'
                    f'vllm:kv_cache_usage_perc{{model_name="mock"}} '
                    f'{min(1.0, self.state["running"] / SLOTS):.3f}\n'
                ).encode()
            self._send(200, body)
        else:
            self._send(404, b"")

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))

        if self.headers.get("Authorization") != f"Bearer {API_KEY}":
            self._send(401, b'{"error":{"message":"Unauthorized"}}')
            return

        with self.lock:
            self.state["waiting"] += 1
        self.semaphore.acquire()
        with self.lock:
            self.state["waiting"] -= 1
            self.state["running"] += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for index in range(TOKENS):
                self._chunk(f"data: {json.dumps({'choices': [{'delta': {'content': f'tok{index} '}}]})}\n\n".encode())
                time.sleep(TOKEN_DELAY)
            self._chunk(f"data: {json.dumps({'choices': [], 'usage': {'completion_tokens': TOKENS}})}\n\n".encode())
            self._chunk(b"data: [DONE]\n\n")
            self._chunk(b"")
        finally:
            with self.lock:
                self.state["running"] -= 1
            self.semaphore.release()

    def _send(self, status, body):
        self.send_response(status)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _chunk(self, payload: bytes):
        self.wfile.write(f"{len(payload):X}\r\n".encode() + payload + b"\r\n")
        self.wfile.flush()


class _MockServer:
    def __enter__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _MockVllmHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        port = self.httpd.server_address[1]
        self.endpoint = f"http://127.0.0.1:{port}/v1"
        return self

    def __exit__(self, *exc):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=5)


class TestBuildPayload(unittest.TestCase):
    def test_text_payload_streams_with_usage(self):
        payload = build_payload("some/model")
        self.assertTrue(payload["stream"])
        self.assertTrue(payload["stream_options"]["include_usage"])
        self.assertIsInstance(payload["messages"][0]["content"], str)

    def test_image_payload_uses_content_array(self):
        payload = build_payload("some/model", ["data:image/png;base64,AAAA"])
        content = payload["messages"][0]["content"]
        self.assertEqual([part["type"] for part in content], ["text", "image_url"])
        self.assertEqual(content[1]["image_url"]["url"], "data:image/png;base64,AAAA")

    def test_multiple_images_all_attached(self):
        payload = build_payload("m", ["data:image/png;base64,A", "data:image/png;base64,B"])
        content = payload["messages"][0]["content"]
        self.assertEqual(sum(1 for part in content if part["type"] == "image_url"), 2)


class TestAgainstMockServer(unittest.TestCase):
    def test_warm_up_succeeds_with_key(self):
        with _MockServer() as server:
            sample = warm_up(server.endpoint, "mock", API_KEY)
        self.assertIsNone(sample.error)
        self.assertIsNotNone(sample.ttft_ms)
        self.assertEqual(sample.output_tokens, TOKENS)

    def test_missing_key_reports_401_instead_of_raising(self):
        with _MockServer() as server:
            sample = warm_up(server.endpoint, "mock", None)
        self.assertIsNotNone(sample.error)
        self.assertIn("401", sample.error)

    def test_unreachable_endpoint_becomes_an_error_sample(self):
        import requests
        session = requests.Session()
        sample = send_request(
            session, "http://127.0.0.1:1/v1/chat/completions", build_payload("m"), timeout=1
        )
        self.assertIsNotNone(sample.error)

    def test_ramp_reports_a_row_per_level(self):
        with _MockServer() as server:
            rows = run_ramp(
                endpoint=server.endpoint,
                model="mock",
                api_key=API_KEY,
                levels=[1, 2],
                phase_seconds=0.5,
                cooldown=0,
                metrics_interval=0.1,
            )

        self.assertEqual([row["concurrency"] for row in rows], [1, 2])
        for row in rows:
            self.assertEqual(row["errors"], 0, row["error_examples"])
            self.assertGreater(row["completed"], 0)
            self.assertIsNotNone(row["ttft_p50_ms"])
            self.assertIsNotNone(row["e2e_p95_ms"])

    def test_ramp_scrapes_server_metrics(self):
        with _MockServer() as server:
            rows = run_ramp(
                endpoint=server.endpoint,
                model="mock",
                api_key=API_KEY,
                levels=[2],
                phase_seconds=0.6,
                cooldown=0,
                metrics_interval=0.1,
            )
        self.assertIn("vllm_num_requests_waiting_max", rows[0]["server"])

    def test_queue_builds_past_saturation(self):
        """Beyond the server's slot count, requests wait rather than fail."""
        with _MockServer() as server:
            rows = run_ramp(
                endpoint=server.endpoint,
                model="mock",
                api_key=API_KEY,
                levels=[SLOTS * 4],
                phase_seconds=0.8,
                cooldown=0,
                metrics_interval=0.05,
            )
        self.assertEqual(rows[0]["errors"], 0, rows[0]["error_examples"])
        self.assertGreater(rows[0]["completed"], 0)


class TestFormatReport(unittest.TestCase):
    def _row(self, concurrency, e2e_p95, errors=0, tok_per_sec=100.0):
        return {
            "concurrency": concurrency,
            "completed": 10,
            "errors": errors,
            "error_examples": ["HTTP 500: boom"] if errors else [],
            "requests_per_sec": 2.0,
            "output_tokens_per_sec": tok_per_sec,
            "ttft_p50_ms": 50.0,
            "ttft_p95_ms": 60.0,
            "e2e_p50_ms": e2e_p95 / 2,
            "e2e_p95_ms": e2e_p95,
            "server": {},
        }

    def test_flags_first_level_over_slo(self):
        rows = [self._row(1, 500.0), self._row(4, 1200.0), self._row(8, 3000.0)]
        report = format_report(rows, slo_ms=1000.0)
        self.assertIn("First level over the 1000ms p95 budget: 4", report)

    def test_says_when_nothing_exceeded_the_budget(self):
        rows = [self._row(1, 100.0), self._row(2, 200.0)]
        report = format_report(rows, slo_ms=5000.0)
        self.assertIn("No level exceeded", report)

    def test_reports_where_errors_started(self):
        rows = [self._row(1, 100.0), self._row(8, 400.0, errors=3)]
        report = format_report(rows, slo_ms=None)
        self.assertIn("Errors first appeared at concurrency 8", report)

    def test_reports_peak_throughput_level(self):
        rows = [self._row(1, 100.0, tok_per_sec=50.0), self._row(4, 400.0, tok_per_sec=180.0)]
        report = format_report(rows, slo_ms=None)
        self.assertIn("first reached at concurrency 4", report)

    def test_knee_is_the_cheapest_level_on_the_plateau(self):
        """Throughput flat from 4 upward: the knee is 4, not the highest level."""
        rows = [
            self._row(1, 100.0, tok_per_sec=50.0),
            self._row(4, 400.0, tok_per_sec=174.0),
            self._row(16, 1800.0, tok_per_sec=175.0),
        ]
        report = format_report(rows, slo_ms=None)
        self.assertIn("first reached at concurrency 4", report)
        self.assertIn("sustainable users ≈ 4 ×", report)

    def test_handles_a_level_where_every_request_failed(self):
        empty = {
            "concurrency": 16, "completed": 0, "errors": 5,
            "error_examples": ["HTTP 503: overloaded"], "requests_per_sec": 0.0,
            "output_tokens_per_sec": 0.0, "ttft_p50_ms": None, "ttft_p95_ms": None,
            "e2e_p50_ms": None, "e2e_p95_ms": None, "server": {},
        }
        report = format_report([empty], slo_ms=1000.0)
        self.assertIn("errors", report)


if __name__ == "__main__":
    unittest.main()
