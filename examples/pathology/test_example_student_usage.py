"""Tests for example_student_usage.py's error-handling branches.

The happy paths are exercised end-to-end by running the script itself against
a live server (see STUDENT_GUIDE.md); this covers the failure modes a student
is actually likely to hit (wrong endpoint, stale API key, server not up yet)
so the friendly SystemExit messages don't regress silently.
"""

import os
import sys
import unittest
from unittest import mock

import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import example_student_usage as student  # noqa: E402


def _response(status_code=200, json_body=None, text=""):
    resp = mock.MagicMock()
    resp.status_code = status_code
    resp.ok = status_code < 400
    resp.text = text
    resp.json.return_value = json_body or {}
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.exceptions.HTTPError(f"{status_code} error")
    else:
        resp.raise_for_status.side_effect = None
    return resp


class TestCheckHealth(unittest.TestCase):
    def test_success_prints_model_and_device(self):
        ok = _response(200, {"model": "conch", "device": "cuda"})
        with mock.patch("requests.get", return_value=ok) as get:
            student.check_health()
        get.assert_called_once_with(f"{student.ENDPOINT}/health", timeout=10)

    def test_connection_error_raises_friendly_system_exit(self):
        with mock.patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            with self.assertRaises(SystemExit) as ctx:
                student.check_health()
        self.assertIn("Could not reach", str(ctx.exception))

    def test_http_error_status_raises_friendly_system_exit(self):
        bad = _response(500, text="internal error")
        with mock.patch("requests.get", return_value=bad):
            with self.assertRaises(SystemExit) as ctx:
                student.check_health()
        self.assertIn("500", str(ctx.exception))


class TestEmbedAndClassify(unittest.TestCase):
    def test_embed_returns_embeddings_on_success(self):
        ok = _response(200, {"embeddings": [[0.1, 0.2]]})
        with mock.patch("requests.post", return_value=ok):
            result = student.embed(["b64img"])
        self.assertEqual(result, [[0.1, 0.2]])

    def test_embed_401_raises_friendly_system_exit(self):
        unauthorized = _response(401)
        with mock.patch("requests.post", return_value=unauthorized):
            with self.assertRaises(SystemExit) as ctx:
                student.embed(["b64img"])
        self.assertIn("API_KEY", str(ctx.exception))

    def test_classify_returns_results_on_success(self):
        ok = _response(200, {"results": [{"predicted_label": "tumor", "score": 0.9}]})
        with mock.patch("requests.post", return_value=ok):
            result = student.classify(["b64img"], {"tumor": "a tumor"})
        self.assertEqual(result, [{"predicted_label": "tumor", "score": 0.9}])

    def test_classify_401_raises_friendly_system_exit(self):
        unauthorized = _response(401)
        with mock.patch("requests.post", return_value=unauthorized):
            with self.assertRaises(SystemExit) as ctx:
                student.classify(["b64img"], {"tumor": "a tumor"})
        self.assertIn("API_KEY", str(ctx.exception))

    def test_other_http_error_propagates(self):
        server_error = _response(500)
        with mock.patch("requests.post", return_value=server_error):
            with self.assertRaises(requests.exceptions.HTTPError):
                student.embed(["b64img"])


class TestMakePlaceholderTiles(unittest.TestCase):
    def test_returns_three_distinct_base64_tiles(self):
        tiles = student.make_placeholder_tiles_b64()
        self.assertEqual(len(tiles), 3)
        self.assertEqual(len(set(tiles)), 3)


if __name__ == "__main__":
    unittest.main()
