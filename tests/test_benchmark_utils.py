"""Tests for benchmark_utils."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmflux.benchmark_utils import (
    ensure_benchmark_data_dir,
    extract_prompts_from_jsonl,
    generate_synthetic_prompts,
    save_prompts_to_jsonl,
)


class TestGenerateSyntheticPrompts(unittest.TestCase):
    def test_returns_correct_count(self):
        prompts = generate_synthetic_prompts(num_prompts=10)
        self.assertEqual(len(prompts), 10)

    def test_reproducible_with_same_seed(self):
        p1 = generate_synthetic_prompts(num_prompts=5, seed=1)
        p2 = generate_synthetic_prompts(num_prompts=5, seed=1)
        self.assertEqual(p1, p2)

    def test_different_seeds_differ(self):
        p1 = generate_synthetic_prompts(num_prompts=5, seed=1)
        p2 = generate_synthetic_prompts(num_prompts=5, seed=99)
        self.assertNotEqual(p1, p2)

    def test_entry_structure(self):
        prompts = generate_synthetic_prompts(num_prompts=1)
        entry = prompts[0]
        self.assertIn("custom_id", entry)
        self.assertIn("method", entry)
        self.assertEqual(entry["method"], "POST")
        self.assertEqual(entry["url"], "/v1/chat/completions")
        self.assertIn("body", entry)
        self.assertIn("messages", entry["body"])
        self.assertGreater(len(entry["body"]["messages"]), 0)

    def test_custom_model_name(self):
        prompts = generate_synthetic_prompts(num_prompts=1, model="my-model:latest")
        self.assertEqual(prompts[0]["body"]["model"], "my-model:latest")

    def test_custom_id_format(self):
        prompts = generate_synthetic_prompts(num_prompts=3)
        ids = [p["custom_id"] for p in prompts]
        self.assertEqual(ids[0], "bench-0000")
        self.assertEqual(ids[1], "bench-0001")
        self.assertEqual(ids[2], "bench-0002")


class TestExtractPromptsFromJsonl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)
        self.jsonl_path = self.tmp_dir / "data.jsonl"
        self.entries = [{"id": i, "text": f"prompt {i}"} for i in range(10)]
        with open(self.jsonl_path, "w") as f:
            for entry in self.entries:
                f.write(json.dumps(entry) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_zero_for_num_prompts_zero(self):
        result = extract_prompts_from_jsonl(self.jsonl_path, num_prompts=0)
        self.assertEqual(result, [])

    def test_returns_all_for_negative_num_prompts(self):
        result = extract_prompts_from_jsonl(self.jsonl_path, num_prompts=-1)
        self.assertEqual(len(result), 10)

    def test_returns_subset(self):
        result = extract_prompts_from_jsonl(self.jsonl_path, num_prompts=4)
        self.assertEqual(len(result), 4)

    def test_returns_all_when_num_exceeds_total(self):
        result = extract_prompts_from_jsonl(self.jsonl_path, num_prompts=100)
        self.assertEqual(len(result), 10)

    def test_skips_blank_lines(self):
        path = self.tmp_dir / "blanks.jsonl"
        with open(path, "w") as f:
            f.write('{"a": 1}\n\n{"b": 2}\n')
        result = extract_prompts_from_jsonl(path, num_prompts=-1)
        self.assertEqual(len(result), 2)


class TestSavePromptsToJsonl(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_writes_valid_jsonl(self):
        prompts = [{"id": 0, "text": "hello"}, {"id": 1, "text": "world"}]
        out = self.tmp_dir / "out.jsonl"
        save_prompts_to_jsonl(prompts, out)
        self.assertTrue(out.exists())
        with open(out) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(lines, prompts)

    def test_creates_parent_directories(self):
        out = self.tmp_dir / "nested" / "dir" / "out.jsonl"
        save_prompts_to_jsonl([{"x": 1}], out)
        self.assertTrue(out.exists())

    def test_empty_list_creates_empty_file(self):
        out = self.tmp_dir / "empty.jsonl"
        save_prompts_to_jsonl([], out)
        self.assertTrue(out.exists())
        self.assertEqual(out.read_text().strip(), "")


class TestEnsureBenchmarkDataDir(unittest.TestCase):
    def test_returns_path_object(self):
        result = ensure_benchmark_data_dir()
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())


if __name__ == "__main__":
    unittest.main()