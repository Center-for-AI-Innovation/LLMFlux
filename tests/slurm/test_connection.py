"""Tests for llmflux.slurm.connection helpers."""

import json
import socket
import tempfile
import unittest
import urllib.error
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from llmflux.slurm.connection import (
    _ping_endpoint,
    _validate_node,
    _validate_port,
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


class TestValidateNode(unittest.TestCase):
    def test_accepts_cluster_hostnames(self):
        for node in ("gpu-node-01", "gpub073", "gpub073.delta.ncsa.illinois.edu"):
            _validate_node(node)  # must not raise

    def test_accepts_private_ipv4(self):
        _validate_node("10.1.2.3")

    def test_rejects_localhost(self):
        for node in ("localhost", "LOCALHOST", "localhost.localdomain", "evil.localhost"):
            with self.assertRaises(ValueError):
                _validate_node(node)

    def test_rejects_loopback_ip(self):
        for node in ("127.0.0.1", "127.1.2.3"):
            with self.assertRaises(ValueError):
                _validate_node(node)

    def test_rejects_link_local_metadata_ip(self):
        with self.assertRaises(ValueError):
            _validate_node("169.254.169.254")

    def test_rejects_ipv6_loopback(self):
        # Colons fail the hostname pattern, which also covers ::1
        with self.assertRaises(ValueError):
            _validate_node("::1")

    def test_rejects_encoded_loopback_ip(self):
        # Legacy IPv4 encodings that ipaddress.ip_address() rejects but the
        # OS resolver maps to 127.0.0.1 (getaddrinfo parses these locally).
        for node in ("2130706433", "0x7f000001", "0177.0.0.1", "127.1"):
            with self.assertRaises(ValueError):
                _validate_node(node)

    def test_rejects_encoded_metadata_ip(self):
        # Decimal/hex/short forms of the 169.254.169.254 cloud metadata IP.
        for node in ("2852039166", "0xA9FEA9FE", "169.254.43518"):
            with self.assertRaises(ValueError):
                _validate_node(node)

    @patch("llmflux.slurm.connection.socket.getaddrinfo")
    def test_rejects_hostname_resolving_to_internal_ip(self, mock_getaddrinfo):
        mock_getaddrinfo.return_value = [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("169.254.169.254", 0))
        ]
        with self.assertRaises(ValueError):
            _validate_node("innocent-looking-host")

    @patch("llmflux.slurm.connection.socket.getaddrinfo", side_effect=socket.gaierror)
    def test_allows_unresolvable_hostname(self, mock_getaddrinfo):
        # An unresolvable name can't be pinged, so there is no SSRF reach to
        # guard against; validation must not reject it.
        _validate_node("gpub073")  # must not raise

    def test_rejects_empty_and_malformed(self):
        for node in ("", "-leading-hyphen", "trailing-hyphen-.x", "has space", "a/b", "node:8000"):
            with self.assertRaises(ValueError):
                _validate_node(node)

    def test_rejects_ssh_option_injection(self):
        with self.assertRaises(ValueError):
            _validate_node("-oProxyCommand=evil")

    @patch.dict("os.environ", {"LLMFLUX_NODE_PATTERN": r"gpu-node-\d+"})
    def test_node_pattern_env_var_enforced(self):
        _validate_node("gpu-node-07")
        with self.assertRaises(ValueError):
            _validate_node("other-host")

    @patch.dict("os.environ", {"LLMFLUX_NODE_PATTERN": r"gpu-node-\d+"})
    def test_node_pattern_must_match_full_hostname(self):
        with self.assertRaises(ValueError):
            _validate_node("gpu-node-07.evil.example.com")

    @patch.dict("os.environ", {"LLMFLUX_NODE_PATTERN": "[invalid"})
    def test_invalid_node_pattern_raises_value_error(self):
        with self.assertRaises(ValueError):
            _validate_node("gpu-node-07")


class TestValidatePort(unittest.TestCase):
    def test_accepts_unprivileged_ports(self):
        for port in (1024, 8000, 11434, 65535):
            _validate_port(port)  # must not raise

    def test_rejects_privileged_and_out_of_range_ports(self):
        for port in (0, 22, 80, 443, 1023, 65536, -1):
            with self.assertRaises(ValueError):
                _validate_port(port)


class TestConnectRejectsTamperedFile(unittest.TestCase):
    @patch("llmflux.slurm.connection._ping_endpoint")
    @patch(
        "llmflux.slurm.connection.read_connection_info",
        return_value={**SAMPLE_INFO, "node": "169.254.169.254"},
    )
    def test_metadata_ip_returns_one_without_pinging(self, mock_read, mock_ping):
        self.assertEqual(connect("99999"), 1)
        mock_ping.assert_not_called()

    @patch("llmflux.slurm.connection._ping_endpoint")
    @patch(
        "llmflux.slurm.connection.read_connection_info",
        return_value={**SAMPLE_INFO, "node": "localhost"},
    )
    def test_localhost_returns_one_without_pinging(self, mock_read, mock_ping):
        self.assertEqual(connect("99999"), 1)
        mock_ping.assert_not_called()

    @patch("llmflux.slurm.connection._ping_endpoint")
    @patch(
        "llmflux.slurm.connection.read_connection_info",
        return_value={**SAMPLE_INFO, "port": 22},
    )
    def test_privileged_port_returns_one_without_pinging(self, mock_read, mock_ping):
        self.assertEqual(connect("99999"), 1)
        mock_ping.assert_not_called()

    @patch(
        "llmflux.slurm.connection.read_connection_info",
        return_value={**SAMPLE_INFO, "node": "127.0.0.1"},
    )
    def test_error_message_mentions_tampering(self, mock_read):
        stderr = StringIO()
        with patch("sys.stderr", stderr):
            connect("99999")
        self.assertIn("tampered", stderr.getvalue())


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
    def test_ping_failure_still_returns_zero(self, mock_read, mock_ping):
        # Unreachable ping is a warning, not a hard failure — endpoint info is still shown
        self.assertEqual(connect("99999"), 0)

    @patch("llmflux.slurm.connection._ping_endpoint", return_value=False)
    @patch("llmflux.slurm.connection.read_connection_info", return_value=SAMPLE_INFO)
    def test_ping_failure_prints_tunnel_hint(self, mock_read, mock_ping):
        with patch("sys.stdout") as mock_stdout:
            connect("99999")
        output = "".join(call.args[0] for call in mock_stdout.write.call_args_list)
        self.assertIn("ssh -N -L", output)
        self.assertIn("localhost:", output)

    def _tunnel_target(self, node):
        info = {**SAMPLE_INFO, "node": node}
        with patch("llmflux.slurm.connection._ping_endpoint", return_value=False), \
             patch("llmflux.slurm.connection.read_connection_info", return_value=info), \
             patch("sys.stdout") as mock_stdout:
            connect("99999")
        output = "".join(call.args[0] for call in mock_stdout.write.call_args_list)
        line = next(l for l in output.splitlines() if "ssh -N -L" in l)
        return line.split()[-2]

    def test_tunnel_target_strips_a_domain_suffix(self):
        self.assertEqual(self._tunnel_target("gpu-node-04.some.domain"), "gpu-node-04")
        self.assertEqual(self._tunnel_target("gpu-node-04"), "gpu-node-04")

    def test_tunnel_target_keeps_an_ip_address_whole(self):
        """A multi-node serve job advertises the fabric IPv4 rank 0 bound.

        Splitting that on "." yields the first octet, and glibc parses a bare
        integer as a packed address rather than rejecting it — so `ssh ... 172`
        fails in a way that looks like a network problem.
        """
        self.assertEqual(self._tunnel_target("172.28.80.96"), "172.28.80.96")
        # IPv6 never reaches this path: _validate_node rejects ":" outright.

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
