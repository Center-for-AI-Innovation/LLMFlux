"""Tests for OutputResult and JSONOutputHandler."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

from llmflux.io.base import OutputResult
from llmflux.io.output.json_output import JSONOutputHandler


class TestOutputResultToDict(unittest.TestCase):
    def test_always_includes_input_and_metadata(self):
        r = OutputResult(input={"messages": []})
        d = r.to_dict()
        self.assertIn("input", d)
        self.assertIn("metadata", d)

    def test_output_included_when_set(self):
        r = OutputResult(input={}, output="hello")
        self.assertEqual(r.to_dict()["output"], "hello")

    def test_output_omitted_when_none(self):
        r = OutputResult(input={})
        self.assertNotIn("output", r.to_dict())

    def test_error_included_when_set(self):
        r = OutputResult(input={}, error="something failed")
        self.assertEqual(r.to_dict()["error"], "something failed")

    def test_error_omitted_when_none(self):
        r = OutputResult(input={})
        self.assertNotIn("error", r.to_dict())

    def test_metadata_defaults_to_empty_dict(self):
        r = OutputResult(input={})
        self.assertEqual(r.to_dict()["metadata"], {})

    def test_metadata_preserved(self):
        r = OutputResult(input={}, metadata={"model": "llama3", "tokens": 42})
        self.assertEqual(r.to_dict()["metadata"]["model"], "llama3")

    def test_both_output_and_error_included(self):
        r = OutputResult(input={}, output="partial", error="timeout")
        d = r.to_dict()
        self.assertEqual(d["output"], "partial")
        self.assertEqual(d["error"], "timeout")


class TestOutputResultFromDict(unittest.TestCase):
    def test_round_trips_full_result(self):
        original = OutputResult(
            input={"messages": [{"role": "user", "content": "hi"}]},
            output="hello",
            error=None,
            metadata={"model": "llama3"},
        )
        restored = OutputResult.from_dict(original.to_dict())
        self.assertEqual(restored.output, "hello")
        self.assertEqual(restored.metadata["model"], "llama3")
        self.assertIsNone(restored.error)

    def test_missing_input_defaults_to_empty_dict(self):
        r = OutputResult.from_dict({})
        self.assertEqual(r.input, {})

    def test_missing_output_is_none(self):
        r = OutputResult.from_dict({"input": {}})
        self.assertIsNone(r.output)

    def test_missing_error_is_none(self):
        r = OutputResult.from_dict({"input": {}})
        self.assertIsNone(r.error)

    def test_missing_metadata_defaults_to_empty_dict(self):
        r = OutputResult.from_dict({"input": {}})
        self.assertEqual(r.metadata, {})

    def test_error_round_trips(self):
        original = OutputResult(input={}, error="timed out")
        restored = OutputResult.from_dict(original.to_dict())
        self.assertEqual(restored.error, "timed out")


class TestJSONOutputHandlerSave(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        self.handler = JSONOutputHandler()

    def tearDown(self):
        self.tmp.cleanup()

    def _make_results(self, n=2):
        return [
            OutputResult(input={"id": i}, output=f"response {i}", metadata={"idx": i})
            for i in range(n)
        ]

    def test_writes_valid_json_file(self):
        out = str(self.tmp_dir / "out.json")
        self.handler.save(self._make_results(), out)
        with open(out) as f:
            data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertEqual(len(data), 2)

    def test_output_content_correct(self):
        out = str(self.tmp_dir / "out.json")
        self.handler.save(self._make_results(1), out)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(data[0]["output"], "response 0")

    def test_creates_parent_directories(self):
        out = str(self.tmp_dir / "nested" / "deep" / "out.json")
        self.handler.save(self._make_results(1), out)
        self.assertTrue(Path(out).exists())

    def test_empty_results_writes_empty_list(self):
        out = str(self.tmp_dir / "empty.json")
        self.handler.save([], out)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(data, [])

    def test_custom_indent(self):
        out = str(self.tmp_dir / "out.json")
        self.handler.save(self._make_results(1), out, indent=4)
        raw = Path(out).read_text()
        self.assertIn("    ", raw)

    def test_reraises_on_write_error(self):
        with patch("builtins.open", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.handler.save(self._make_results(1), "/nonexistent/path/out.json")

    def test_multiple_results_all_saved(self):
        results = self._make_results(5)
        out = str(self.tmp_dir / "multi.json")
        self.handler.save(results, out)
        with open(out) as f:
            data = json.load(f)
        self.assertEqual(len(data), 5)


if __name__ == "__main__":
    unittest.main()