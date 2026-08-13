"""Every outbound request from LLMClient must carry a timeout.

Without one, `requests` blocks on a socket read indefinitely. If the engine dies
after the job's readiness check has already passed, the batch processor waits
for the remainder of the job's walltime — the job is billed in full and produces
nothing. That is the same silent-waste failure class as an idle allocated node,
and it gets more likely as more nodes are added, since any one of them can die.

A timeout converts it into a per-item failure that BatchProcessor's existing
retry and error accounting can see.
"""

import os
import unittest
from unittest.mock import MagicMock, patch

from llmflux.core.client import LLMClient


class TestClientTimeouts(unittest.TestCase):
    def _client(self, **env):
        with patch.dict(os.environ, {"VLLM_HOST": "http://localhost:8000", **env}):
            return LLMClient(engine="vllm")

    def test_timeout_is_a_connect_read_pair(self):
        c = self._client()
        self.assertIsInstance(c.timeout, tuple)
        self.assertEqual(len(c.timeout), 2)
        for v in c.timeout:
            self.assertGreater(v, 0)

    def test_timeouts_are_configurable(self):
        c = self._client(LLMFLUX_CONNECT_TIMEOUT="3", LLMFLUX_READ_TIMEOUT="45")
        self.assertEqual(c.timeout, (3.0, 45.0))

    def test_read_budget_is_generous_by_default(self):
        """Long generations are legitimate; the budget only has to be finite."""
        c = self._client()
        self.assertGreaterEqual(c.timeout[1], 60)

    def test_get_passes_timeout(self):
        c = self._client()
        with patch.object(c.session, "get") as get:
            get.return_value = MagicMock(
                status_code=200, json=lambda: {"models": []}, raise_for_status=lambda: None
            )
            c.list_models()
            self.assertIn("timeout", get.call_args.kwargs)
            self.assertEqual(get.call_args.kwargs["timeout"], c.timeout)

    def test_no_session_call_site_omits_timeout(self):
        """Source-level guard: a new call site added without a timeout fails here."""
        import inspect
        import llmflux.core.client as mod

        src = inspect.getsource(mod)
        offenders = [
            line.strip()
            for line in src.splitlines()
            if "self.session." in line
            and "timeout=" not in line
            and not line.strip().startswith("#")
        ]
        self.assertEqual(
            offenders, [], f"session call sites without a timeout: {offenders}"
        )


if __name__ == "__main__":
    unittest.main()
