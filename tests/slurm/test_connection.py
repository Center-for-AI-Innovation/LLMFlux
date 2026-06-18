"""Tests for llmflux.slurm.connection helpers."""

import json
import tempfile
import unittest
import urllib.error
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from llmflux.slurm.connection import (
    _ping_endpoint,
    connect,
    read_connection_info,
    wait_for_connection_file,
)

SAMPLE_INFO = {
    "job_id": "99999",
    "node": "gpu-node-01",
    "port": 8000,
    "model": "meta-llama/Llama-3.2-3B-Instruct",
    "api_key": "llmflux-00000000000000000000000000000000",
    "engine": "vllm",
}


class TestReadConnectionInfo(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.home = Path(self.temp_dir.name)
        self.home_patcher = patch(
            "llmflux.slurm.connection.Path.home", return_value=self.home
        )
        self.home_patcher.start()

    def tearDown(self):
        self.home_patcher.stop()
        self.temp_dir.cleanup()

    def _write_connection_file(self, job_id: str, content: str) -> Path:
        path = self.home / ".llmflux" / "serve" / job_id / "connection.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def test_missing_file_returns_none(self):
        self.assertIsNone(read_connection_info("99999"))

    def test_valid_json_returns_dict(self):
        self._write_connection_file("99999", json.dumps(SAMPLE_INFO))
        result = read_connection_info("99999")
        self.assertEqual(result, SAMPLE_INFO)

    def test_corrupt_json_returns_none(self):
        self._write_connection_file("99999", "{not valid json")
        self.assertIsNone(read_connection_info("99999"))

    def test_os_error_on_read_returns_none(self):
        self._write_connection_file("99999", json.dumps(SAMPLE_INFO))
        with patch("pathlib.Path.read_text", side_effect=OSError("permission denied")):
            self.assertIsNone(read_connection_info("99999"))

    def test_different_job_ids_are_isolated(self):
        self._write_connection_file("11111", json.dumps({"node": "a"}))
        self.assertIsNone(read_connection_info("22222"))
        self.assertEqual(read_connection_info("11111"), {"node": "a"})


class TestWaitForConnectionFile(unittest.TestCase):
    @patch("llmflux.slurm.connection.time.sleep")
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_returns_immediately_when_file_already_present(self, mock_read, mock_sleep):
        result = wait_for_connection_file("99999", poll_interval=0, timeout=60)
        self.assertEqual(result, SAMPLE_INFO)
        mock_sleep.assert_not_called()

    @patch("llmflux.slurm.connection.time.sleep")
    @patch(
        "llmflux.slurm.connection.read_connection_info",
        side_effect=[None, None, SAMPLE_INFO],
    )
    def test_polls_until_file_appears(self, mock_read, mock_sleep):
        result = wait_for_connection_file("99999", poll_interval=0, timeout=3600)
        self.assertEqual(result, SAMPLE_INFO)
        self.assertEqual(mock_read.call_count, 3)

    @patch("llmflux.slurm.connection.time.sleep")
    @patch("llmflux.slurm.connection.read_connection_info", return_value=None)
    @patch(
        "llmflux.slurm.connection.time.monotonic",
        side_effect=[0, 0, 2],
    )
    def test_raises_timeout_error_when_deadline_passes(
        self, mock_monotonic, mock_read, mock_sleep
    ):
        with self.assertRaises(TimeoutError) as ctx:
            wait_for_connection_file("99999", poll_interval=0, timeout=1)
        self.assertIn("99999", str(ctx.exception))
        self.assertIn("llmflux logs", str(ctx.exception))

    @patch("llmflux.slurm.connection.time.sleep")
    @patch(
        "llmflux.slurm.connection.read_connection_info",
        side_effect=[None, None, SAMPLE_INFO],
    )
    def test_sleeps_with_configured_interval(self, mock_read, mock_sleep):
        wait_for_connection_file("99999", poll_interval=7, timeout=3600)
        for c in mock_sleep.call_args_list:
            self.assertEqual(c, call(7))

    @patch("llmflux.slurm.connection.time.sleep")
    @patch("llmflux.slurm.connection.read_connection_info", return_value=None)
    @patch(
        "llmflux.slurm.connection.time.monotonic",
        side_effect=[0, 0, 2],
    )
    def test_timeout_error_mentions_timeout_seconds(
        self, mock_monotonic, mock_read, mock_sleep
    ):
        with self.assertRaises(TimeoutError) as ctx:
            wait_for_connection_file("99999", poll_interval=0, timeout=1)
        self.assertIn("1s", str(ctx.exception))


class TestPingEndpoint(unittest.TestCase):
    def _make_response(self, status: int):
        mock_resp = MagicMock()
        mock_resp.status = status
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    @patch("llmflux.slurm.connection.urllib.request.urlopen")
    def test_vllm_uses_health_path(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response(200)
        _ping_endpoint("gpu-node-01", 8000, "vllm")
        url_used = mock_urlopen.call_args[0][0]
        self.assertIn("/health", url_used)
        self.assertNotIn("/api/version", url_used)

    @patch("llmflux.slurm.connection.urllib.request.urlopen")
    def test_ollama_uses_api_version_path(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response(200)
        _ping_endpoint("gpu-node-01", 11434, "ollama")
        url_used = mock_urlopen.call_args[0][0]
        self.assertIn("/api/version", url_used)
        self.assertNotIn("/health", url_used)

    @patch("llmflux.slurm.connection.urllib.request.urlopen")
    def test_returns_true_on_200(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response(200)
        self.assertTrue(_ping_endpoint("gpu-node-01", 8000, "vllm"))

    @patch("llmflux.slurm.connection.urllib.request.urlopen")
    def test_returns_false_on_non_200(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response(503)
        self.assertFalse(_ping_endpoint("gpu-node-01", 8000, "vllm"))

    @patch(
        "llmflux.slurm.connection.urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    )
    def test_returns_false_on_url_error(self, mock_urlopen):
        self.assertFalse(_ping_endpoint("gpu-node-01", 8000, "vllm"))

    @patch(
        "llmflux.slurm.connection.urllib.request.urlopen",
        side_effect=OSError("timeout"),
    )
    def test_returns_false_on_os_error(self, mock_urlopen):
        self.assertFalse(_ping_endpoint("gpu-node-01", 8000, "vllm"))

    @patch("llmflux.slurm.connection.urllib.request.urlopen")
    def test_url_contains_node_and_port(self, mock_urlopen):
        mock_urlopen.return_value = self._make_response(200)
        _ping_endpoint("gpu-node-07", 9999, "vllm")
        url_used = mock_urlopen.call_args[0][0]
        self.assertIn("gpu-node-07", url_used)
        self.assertIn("9999", url_used)


class TestConnect(unittest.TestCase):
    @patch("llmflux.slurm.connection._ping_endpoint", return_value=True)
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_success_returns_zero(self, mock_read, mock_ping):
        self.assertEqual(connect("99999"), 0)

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=True)
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_success_prints_endpoint(self, mock_read, mock_ping):
        with patch("builtins.print") as mock_print:
            connect("99999")
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("http://gpu-node-01:8000/v1", output)

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=True)
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_success_prints_api_key(self, mock_read, mock_ping):
        with patch("builtins.print") as mock_print:
            connect("99999")
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn("llmflux-00000000000000000000000000000000", output)

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=True)
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_success_prints_model_and_engine(self, mock_read, mock_ping):
        with patch("builtins.print") as mock_print:
            connect("99999")
        output = " ".join(str(c) for c in mock_print.call_args_list)
        self.assertIn(SAMPLE_INFO["model"], output)
        self.assertIn(SAMPLE_INFO["engine"], output)

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=False)
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_ping_failure_returns_one(self, mock_read, mock_ping):
        self.assertEqual(connect("99999"), 1)

    @patch(
        "llmflux.slurm.connection.wait_for_connection_file",
        side_effect=TimeoutError("did not load"),
    )
    @patch("llmflux.slurm.connection.read_connection_info", return_value=None)
    def test_timeout_waiting_returns_one(self, mock_read, mock_wait):
        self.assertEqual(connect("99999", wait_timeout=1), 1)

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=True)
    @patch(
        "llmflux.slurm.connection.wait_for_connection_file",
        return_value=SAMPLE_INFO,
    )
    @patch("llmflux.slurm.connection.read_connection_info", return_value=None)
    def test_waits_when_file_not_yet_present(self, mock_read, mock_wait, mock_ping):
        result = connect("99999", wait_timeout=120)
        mock_wait.assert_called_once_with("99999", timeout=120)
        self.assertEqual(result, 0)

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=True)
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_skips_wait_when_file_already_present(self, mock_read, mock_ping):
        with patch(
            "llmflux.slurm.connection.wait_for_connection_file"
        ) as mock_wait:
            connect("99999")
        mock_wait.assert_not_called()

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=True)
    @patch(
        "llmflux.slurm.connection.read_connection_info",
        return_value={**SAMPLE_INFO, "engine": "ollama"},
    )
    def test_pings_with_correct_engine(self, mock_read, mock_ping):
        connect("99999")
        _, kwargs = mock_ping.call_args if mock_ping.call_args.kwargs else (mock_ping.call_args.args, {})
        args = mock_ping.call_args.args
        self.assertEqual(args[2], "ollama")


if __name__ == "__main__":
    unittest.main()
