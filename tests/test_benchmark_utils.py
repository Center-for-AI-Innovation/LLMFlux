"""Tests for benchmark_utils."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmflux.benchmark_utils import (
    create_test_prompts_file,
    download_prompts_data,
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


class TestCreateTestPromptsFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_category_file(self, benchmark_dir: Path, category: str, n: int = 5):
        path = benchmark_dir / f"{category}.jsonl"
        prompts = [
            [{"role": "user", "content": f"{category} question {i}"}]
            for i in range(n)
        ]
        with open(path, "w") as f:
            for p in prompts:
                f.write(json.dumps(p) + "\n")

    def test_returns_path_string(self):
        import llmflux.benchmark_utils as bu
        categories = ["data_analysis", "language", "math", "reasoning", "instruction_following", "coding"]
        bench_dir = self.tmp_dir / "benchmark_data"
        bench_dir.mkdir()
        for cat in categories:
            self._write_category_file(bench_dir, cat)
        with patch.object(bu, "BENCHMARK_DATA_DIR", bench_dir):
            result = create_test_prompts_file(num_prompts=6)
        self.assertIsInstance(result, str)
        self.assertTrue(Path(result).exists())

    def test_output_is_valid_jsonl(self):
        import llmflux.benchmark_utils as bu
        categories = ["data_analysis", "language", "math", "reasoning", "instruction_following", "coding"]
        bench_dir = self.tmp_dir / "benchmark_data"
        bench_dir.mkdir()
        for cat in categories:
            self._write_category_file(bench_dir, cat, n=5)
        with patch.object(bu, "BENCHMARK_DATA_DIR", bench_dir):
            result_path = create_test_prompts_file(num_prompts=6, model="test-model")
        with open(result_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertGreater(len(lines), 0)
        for entry in lines:
            self.assertIn("custom_id", entry)
            self.assertIn("method", entry)
            self.assertIn("body", entry)
            self.assertIn("messages", entry["body"])
            self.assertIn("model", entry["body"])

    def test_calls_download_when_files_missing(self):
        import llmflux.benchmark_utils as bu
        bench_dir = self.tmp_dir / "benchmark_data_empty"
        bench_dir.mkdir()
        categories = ["data_analysis", "language", "math", "reasoning", "instruction_following", "coding"]
        with patch.object(bu, "BENCHMARK_DATA_DIR", bench_dir):
            with patch.object(bu, "download_prompts_data") as mock_dl:
                # After download_prompts_data is called, write the files
                def side_effect():
                    for cat in categories:
                        self._write_category_file(bench_dir, cat, n=3)
                mock_dl.side_effect = side_effect
                create_test_prompts_file(num_prompts=6)
            mock_dl.assert_called_once()

    def test_skips_download_when_files_present(self):
        import llmflux.benchmark_utils as bu
        categories = ["data_analysis", "language", "math", "reasoning", "instruction_following", "coding"]
        bench_dir = self.tmp_dir / "benchmark_data_full"
        bench_dir.mkdir()
        for cat in categories:
            self._write_category_file(bench_dir, cat, n=5)
        with patch.object(bu, "BENCHMARK_DATA_DIR", bench_dir):
            with patch.object(bu, "download_prompts_data") as mock_dl:
                create_test_prompts_file(num_prompts=6)
            mock_dl.assert_not_called()


class TestEnsureBenchmarkDataDir(unittest.TestCase):
    def test_returns_path_object(self):
        result = ensure_benchmark_data_dir()
        self.assertIsInstance(result, Path)
        self.assertTrue(result.exists())


class TestDownloadPromptsData(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_raises_import_error_when_deps_missing(self):
        import builtins
        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name in ("datasets", "pandas"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with self.assertRaises(ImportError) as ctx:
                download_prompts_data()
        self.assertIn("datasets", str(ctx.exception))

    def test_downloads_all_categories(self):
        import sys
        import llmflux.benchmark_utils as bu
        bench_dir = self.tmp_dir / "benchmark_data"
        bench_dir.mkdir()

        mock_row = MagicMock()
        mock_row.__getitem__ = lambda self, key: ["What is ML?"] if key == "turns" else None

        mock_df = MagicMock()
        mock_df.iterrows.return_value = iter([(0, mock_row)])

        mock_dataset = MagicMock()
        mock_dataset.to_pandas.return_value = mock_df

        mock_datasets_mod = MagicMock()
        mock_datasets_mod.load_dataset.return_value = mock_dataset

        categories = ["coding", "data_analysis", "instruction_following", "math", "reasoning", "language"]

        with patch.object(bu, "BENCHMARK_DATA_DIR", bench_dir):
            with patch.dict(sys.modules, {"datasets": mock_datasets_mod}):
                download_prompts_data()

        written = list(bench_dir.glob("*.jsonl"))
        self.assertEqual(len(written), len(categories))


if __name__ == "__main__":
    unittest.main()