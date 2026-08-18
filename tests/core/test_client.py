"""Tests for LLMClient."""

import os
import unittest
from unittest.mock import MagicMock, patch

from llmflux.core.client import LLMClient


class TestLLMClientInit(unittest.TestCase):
    def test_default_ollama_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OLLAMA_HOST", None)
            os.environ.pop("OLLAMA_PORT", None)
            client = LLMClient(engine="ollama")
        self.assertEqual(client.base_url, "http://localhost:11434")

    def test_default_vllm_url(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("VLLM_HOST", None)
            os.environ.pop("VLLM_PORT", None)
            client = LLMClient(engine="vllm")
        self.assertEqual(client.base_url, "http://localhost:11434")

    def test_full_url_in_host_used_directly(self):
        client = LLMClient(engine="ollama", host="http://myserver:8080")
        self.assertEqual(client.base_url, "http://myserver:8080")

    def test_host_and_port_combined(self):
        client = LLMClient(engine="ollama", host="myserver", port=9999)
        self.assertEqual(client.base_url, "http://myserver:9999")

    def test_ollama_host_env_var(self):
        with patch.dict(os.environ, {"OLLAMA_HOST": "http://env-host:1234"}):
            client = LLMClient(engine="ollama")
        self.assertEqual(client.base_url, "http://env-host:1234")

    def test_vllm_host_env_var(self):
        with patch.dict(os.environ, {"VLLM_HOST": "http://vllm-host:5678"}):
            client = LLMClient(engine="vllm")
        self.assertEqual(client.base_url, "http://vllm-host:5678")

    def test_custom_port_env_var(self):
        with patch.dict(os.environ, {"OLLAMA_PORT": "9000"}, clear=False):
            os.environ.pop("OLLAMA_HOST", None)
            client = LLMClient(engine="ollama")
        self.assertEqual(client.base_url, "http://localhost:9000")


class TestLLMClientAuth(unittest.TestCase):
    """A `serve` endpoint runs vLLM with --api-key, so requests need a bearer token."""

    def test_no_auth_header_without_key(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_API_KEY", None)
            client = LLMClient(engine="vllm", host="localhost", port=8000)
        self.assertIsNone(client.api_key)
        self.assertNotIn("Authorization", client.session.headers)

    def test_api_key_argument_sets_bearer_header(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_API_KEY", None)
            client = LLMClient(engine="vllm", host="localhost", port=8000, api_key="llmflux-abc")
        self.assertEqual(client.session.headers["Authorization"], "Bearer llmflux-abc")

    def test_api_key_env_var_fallback(self):
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": "llmflux-from-env"}):
            client = LLMClient(engine="vllm", host="localhost", port=8000)
        self.assertEqual(client.api_key, "llmflux-from-env")
        self.assertEqual(client.session.headers["Authorization"], "Bearer llmflux-from-env")

    def test_argument_wins_over_env_var(self):
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": "llmflux-from-env"}):
            client = LLMClient(engine="vllm", host="localhost", port=8000, api_key="llmflux-explicit")
        self.assertEqual(client.api_key, "llmflux-explicit")

    def test_blank_env_var_treated_as_unset(self):
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": ""}):
            client = LLMClient(engine="vllm", host="localhost", port=8000)
        self.assertIsNone(client.api_key)
        self.assertNotIn("Authorization", client.session.headers)

    def test_whitespace_only_env_var_treated_as_unset(self):
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": "   "}):
            client = LLMClient(engine="vllm", host="localhost", port=8000)
        self.assertIsNone(client.api_key)
        self.assertNotIn("Authorization", client.session.headers)

    def test_surrounding_whitespace_stripped_from_env_var(self):
        # A quoted .env value keeps its trailing space; vLLM compares the header
        # exactly, so an unstripped "Bearer llmflux-abc " would 401.
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": " llmflux-abc "}):
            client = LLMClient(engine="vllm", host="localhost", port=8000)
        self.assertEqual(client.api_key, "llmflux-abc")
        self.assertEqual(client.session.headers["Authorization"], "Bearer llmflux-abc")

    def test_trailing_newline_stripped_from_env_var(self):
        # requests raises ValueError on a header value containing a newline.
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": "llmflux-abc\n"}):
            client = LLMClient(engine="vllm", host="localhost", port=8000)
        self.assertEqual(client.session.headers["Authorization"], "Bearer llmflux-abc")

    def test_surrounding_whitespace_stripped_from_argument(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_API_KEY", None)
            client = LLMClient(
                engine="vllm", host="localhost", port=8000, api_key=" llmflux-abc\n"
            )
        self.assertEqual(client.api_key, "llmflux-abc")
        self.assertEqual(client.session.headers["Authorization"], "Bearer llmflux-abc")

    def test_blank_argument_falls_back_to_env_var(self):
        # A blank argument is not a key, so it does not shadow a real env var.
        for blank in ("", "   ", "\n"):
            with self.subTest(api_key=repr(blank)):
                with patch.dict(os.environ, {"LLMFLUX_API_KEY": "llmflux-from-env"}):
                    client = LLMClient(
                        engine="vllm", host="localhost", port=8000, api_key=blank
                    )
                self.assertEqual(client.api_key, "llmflux-from-env")

    def test_blank_argument_and_blank_env_var_leave_key_unset(self):
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": "  "}):
            client = LLMClient(
                engine="vllm", host="localhost", port=8000, api_key="  "
            )
        self.assertIsNone(client.api_key)
        self.assertNotIn("Authorization", client.session.headers)


class TestListModels(unittest.TestCase):
    def _make_client(self):
        client = LLMClient(engine="ollama", host="localhost", port=11434)
        client.session = MagicMock()
        return client

    def test_returns_model_names(self):
        client = self._make_client()
        resp = MagicMock()
        resp.json.return_value = {"models": [{"name": "llama3.2:3b"}, {"name": "mistral:7b"}]}
        client.session.get.return_value = resp
        models = client.list_models()
        self.assertEqual(models, ["llama3.2:3b", "mistral:7b"])

    def test_returns_empty_on_unexpected_format(self):
        client = self._make_client()
        resp = MagicMock()
        resp.json.return_value = {"unexpected": "data"}
        client.session.get.return_value = resp
        models = client.list_models()
        self.assertEqual(models, [])

    def test_returns_empty_on_request_error(self):
        import requests
        client = self._make_client()
        client.session.get.side_effect = requests.exceptions.ConnectionError("refused")
        models = client.list_models()
        self.assertEqual(models, [])


class TestModelExists(unittest.TestCase):
    def _make_client(self, engine="ollama"):
        client = LLMClient(engine=engine, host="localhost", port=11434)
        client.session = MagicMock()
        return client

    def test_vllm_always_returns_true(self):
        client = self._make_client(engine="vllm")
        self.assertTrue(client.model_exists("any-model"))
        client.session.get.assert_not_called()

    def test_ollama_model_found(self):
        client = self._make_client()
        resp = MagicMock()
        resp.json.return_value = {"models": [{"name": "llama3.2:3b"}]}
        client.session.get.return_value = resp
        self.assertTrue(client.model_exists("llama3.2:3b"))

    def test_ollama_model_not_found(self):
        client = self._make_client()
        resp = MagicMock()
        resp.json.return_value = {"models": [{"name": "mistral:7b"}]}
        client.session.get.return_value = resp
        self.assertFalse(client.model_exists("llama3.2:3b"))


class TestPullModel(unittest.TestCase):
    def _make_client(self):
        client = LLMClient(engine="ollama", host="localhost", port=11434)
        client.session = MagicMock()
        return client

    def test_successful_pull(self):
        client = self._make_client()
        resp = MagicMock()
        client.session.post.return_value = resp
        self.assertTrue(client.pull_model("llama3.2:3b"))

    def test_failed_pull_returns_false(self):
        import requests
        client = self._make_client()
        client.session.post.side_effect = requests.exceptions.ConnectionError("refused")
        self.assertFalse(client.pull_model("llama3.2:3b"))


class TestChat(unittest.TestCase):
    def _make_client(self, engine="vllm"):
        client = LLMClient(engine=engine, host="localhost", port=11434)
        client.session = MagicMock()
        return client

    def _mock_response(self, client, content="hello"):
        resp = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {"role": "assistant", "content": content}}]
        }
        client.session.post.return_value = resp
        return resp

    def test_returns_content_string(self):
        client = self._make_client()
        self._mock_response(client, "The answer is 42.")
        result = client.chat(
            model="test/model",
            engine="vllm",
            messages=[{"role": "user", "content": "What is the answer?"}],
        )
        self.assertEqual(result, "The answer is 42.")

    def test_optional_params_forwarded(self):
        client = self._make_client()
        self._mock_response(client)
        client.chat(
            model="test/model",
            engine="vllm",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.5,
            max_tokens=128,
            top_p=0.9,
        )
        call_payload = client.session.post.call_args[1]["json"]
        self.assertEqual(call_payload["temperature"], 0.5)
        self.assertEqual(call_payload["max_tokens"], 128)
        self.assertEqual(call_payload["top_p"], 0.9)

    def test_unexpected_response_returns_empty_string(self):
        client = self._make_client()
        resp = MagicMock()
        resp.json.return_value = {"no_choices": True}
        client.session.post.return_value = resp
        result = client.chat(
            model="test/model",
            engine="vllm",
            messages=[{"role": "user", "content": "hi"}],
        )
        self.assertEqual(result, "")

    def test_auth_failure_without_key_logs_actionable_hint(self):
        import requests
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_API_KEY", None)
            client = self._make_client()
        error = requests.exceptions.HTTPError("401 Client Error")
        error.response = MagicMock(status_code=401)
        client.session.post.side_effect = error

        with self.assertLogs("llmflux.core.client", level="ERROR") as logs:
            with self.assertRaises(requests.exceptions.RequestException):
                client.chat(
                    model="test/model",
                    engine="vllm",
                    messages=[{"role": "user", "content": "hi"}],
                )

        self.assertIn("LLMFLUX_API_KEY", "\n".join(logs.output))

    def test_forbidden_without_key_logs_actionable_hint(self):
        import requests
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_API_KEY", None)
            client = self._make_client()
        error = requests.exceptions.HTTPError("403 Client Error")
        error.response = MagicMock(status_code=403)
        client.session.post.side_effect = error

        with self.assertLogs("llmflux.core.client", level="ERROR") as logs:
            with self.assertRaises(requests.exceptions.RequestException):
                client.chat(
                    model="test/model",
                    engine="vllm",
                    messages=[{"role": "user", "content": "hi"}],
                )

        self.assertIn("LLMFLUX_API_KEY", "\n".join(logs.output))

    def test_auth_failure_with_key_does_not_claim_key_is_missing(self):
        # A stale or wrong key should not be told "no API key is set".
        import requests
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_API_KEY", None)
            client = LLMClient(
                engine="vllm", host="localhost", port=11434, api_key="stale-key"
            )
        client.session = MagicMock()
        error = requests.exceptions.HTTPError("401 Client Error")
        error.response = MagicMock(status_code=401)
        client.session.post.side_effect = error

        with self.assertLogs("llmflux.core.client", level="ERROR") as logs:
            with self.assertRaises(requests.exceptions.RequestException):
                client.chat(
                    model="test/model",
                    engine="vllm",
                    messages=[{"role": "user", "content": "hi"}],
                )

        output = "\n".join(logs.output)
        self.assertNotIn("LLMFLUX_API_KEY", output)
        self.assertNotIn("stale-key", output)

    def test_non_auth_status_does_not_log_auth_hint(self):
        import requests
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_API_KEY", None)
            client = self._make_client()
        error = requests.exceptions.HTTPError("500 Server Error")
        error.response = MagicMock(status_code=500)
        client.session.post.side_effect = error

        with self.assertLogs("llmflux.core.client", level="ERROR") as logs:
            with self.assertRaises(requests.exceptions.RequestException):
                client.chat(
                    model="test/model",
                    engine="vllm",
                    messages=[{"role": "user", "content": "hi"}],
                )

        self.assertNotIn("LLMFLUX_API_KEY", "\n".join(logs.output))

    def test_raises_on_request_error(self):
        import requests
        client = self._make_client()
        client.session.post.side_effect = requests.exceptions.ConnectionError("down")
        with self.assertRaises(requests.exceptions.RequestException):
            client.chat(
                model="test/model",
                engine="vllm",
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_ollama_engine_checks_model_availability(self):
        client = self._make_client(engine="ollama")
        self._mock_response(client)
        # Patch model_exists and ensure_model_available so no real network call
        with patch.object(client, "ensure_model_available", return_value=True) as mock_ensure:
            client.chat(
                model="llama3.2:3b",
                engine="ollama",
                messages=[{"role": "user", "content": "hi"}],
            )
        mock_ensure.assert_called_once_with("llama3.2:3b")

    def test_ollama_raises_when_model_unavailable(self):
        client = self._make_client(engine="ollama")
        with patch.object(client, "ensure_model_available", return_value=False):
            with self.assertRaises(ValueError):
                client.chat(
                    model="missing-model",
                    engine="ollama",
                    messages=[{"role": "user", "content": "hi"}],
                )


if __name__ == "__main__":
    unittest.main()
