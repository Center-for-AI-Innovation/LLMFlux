"""Integration tests that exercise multiple components working together."""

import json
import os
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import MagicMock, patch

from llmflux.converters.json import json_to_jsonl
from llmflux.converters.utils import merge_jsonl_files, jsonl_to_json, validate_jsonl
from llmflux.core.config import ModelConfig, ModelParameters
from llmflux.core.registry import JobRegistry
from llmflux.processors.batch import BatchProcessor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_model_config():
    return ModelConfig(
        name="test:7b",
        hf_name="test/test-model",
        parameters=ModelParameters(
            temperature=0.7, max_tokens=500, top_p=0.9, top_k=40, stop_sequences=None
        ),
    )


def _write_jsonl(path, entries):
    with open(path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")


CHAT_ENTRY = {
    "custom_id": "req-1",
    "method": "POST",
    "url": "/v1/chat/completions",
    "body": {
        "model": "test:7b",
        "messages": [{"role": "user", "content": "Hello"}],
        "temperature": 0.7,
        "max_tokens": 500,
    },
}

COMPLETIONS_ENTRY = {
    "custom_id": "req-comp",
    "method": "POST",
    "url": "/v1/completions",
    "body": {"prompt": "The sky is", "temperature": 0.7, "max_tokens": 50},
}

UNSUPPORTED_ENTRY = {
    "custom_id": "req-bad",
    "method": "POST",
    "url": "/v1/embeddings",
    "body": {"input": "text"},
}


# ===========================================================================
# 1. BatchProcessor end-to-end pipeline
# ===========================================================================

class TestBatchProcessorPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.model_config = _make_model_config()
        self.output = self.d / "output.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, mock_cls, entries, response="answer", batch_size=4, save_frequency=50, output=None):
        """Wire the mock class so setup() gets a properly configured client."""
        mock_client = MagicMock()
        mock_client.chat.return_value = response
        mock_cls.return_value = mock_client
        input_path = self.d / "input.jsonl"
        _write_jsonl(input_path, entries)
        processor = BatchProcessor(
            model_config=self.model_config,
            batch_size=batch_size,
            save_frequency=save_frequency,
        )
        out = str(output or self.output)
        results = processor.run(str(input_path), out, "vllm")
        return results, processor, mock_client

    @patch("llmflux.processors.batch.LLMClient")
    def test_full_run_writes_output_file(self, mock_cls):
        """Full pipeline: JSONL in → JSON file out with correct structure."""
        results, _, _ = self._run(mock_cls, [CHAT_ENTRY], response="Hi there!")

        self.assertEqual(len(results), 1)
        self.assertTrue(self.output.exists())
        saved = json.loads(self.output.read_text())
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["input"]["custom_id"], "req-1")
        self.assertIn("Hi there!", str(saved[0]["output"]))

    @patch("llmflux.processors.batch.LLMClient")
    def test_batch_chunking_triggers_intermediate_save(self, mock_cls):
        """With batch_size=1 and save_frequency=1, intermediate file is written each batch."""
        entries = [dict(CHAT_ENTRY, custom_id=f"req-{i}") for i in range(3)]
        results, processor, _ = self._run(mock_cls, entries, batch_size=1, save_frequency=1)

        self.assertEqual(len(results), 3)
        self.assertTrue(self.output.exists())
        self.assertFalse(Path(processor.temp_file).exists())

    @patch("llmflux.processors.batch.LLMClient")
    def test_missing_input_raises(self, mock_cls):
        """FileNotFoundError is raised before any processing if input is missing."""
        mock_cls.return_value = MagicMock()
        processor = BatchProcessor(model_config=self.model_config)
        with self.assertRaises(FileNotFoundError):
            processor.run(str(self.d / "nonexistent.jsonl"), str(self.output), "vllm")

    @patch("llmflux.processors.batch.LLMClient")
    def test_completions_url_routes_correctly(self, mock_cls):
        """Items with /v1/completions URL are handled and produce text_completion output."""
        results, _, _ = self._run(mock_cls, [COMPLETIONS_ENTRY], response="blue")

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].output["object"], "text_completion")

    @patch("llmflux.processors.batch.LLMClient")
    def test_unsupported_url_records_error_and_continues(self, mock_cls):
        """Unsupported URL produces an error result; processing of other items continues."""
        results, _, _ = self._run(mock_cls, [UNSUPPORTED_ENTRY, CHAT_ENTRY], response="ok")

        self.assertEqual(len(results), 2)
        self.assertIsNone(results[0].output)
        self.assertIsNotNone(results[0].error)
        self.assertIsNotNone(results[1].output)

    @patch("llmflux.processors.batch.LLMClient")
    def test_output_preserves_custom_ids(self, mock_cls):
        """custom_id flows through unchanged from JSONL to saved output."""
        entries = [dict(CHAT_ENTRY, custom_id="alpha"), dict(CHAT_ENTRY, custom_id="beta")]
        self._run(mock_cls, entries, response="response")

        saved = json.loads(self.output.read_text())
        ids = [r["input"]["custom_id"] for r in saved]
        self.assertEqual(ids, ["alpha", "beta"])


# ===========================================================================
# 2. CLI jobs + status + cancel commands with real JobRegistry
# ===========================================================================

class TestJobsCommandIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry_file = Path(self.tmp.name) / "jobs.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _registry_with_jobs(self, jobs: dict):
        r = JobRegistry(registry_file=str(self.registry_file))
        for job_id, meta in jobs.items():
            r.create_job(job_id, meta)
        return r

    def _args(self, **kwargs):
        args = MagicMock()
        args.all = False
        args.state = []
        for k, v in kwargs.items():
            setattr(args, k, v)
        return args

    @patch("llmflux.cli.JobRegistry")
    def test_empty_registry_prints_message(self, mock_registry_cls, capsys=None):
        from llmflux.cli import _jobs_command
        mock_registry_cls.return_value.get_all_jobs.return_value = {}
        args = self._args()
        with patch("sys.stdout", new=StringIO()) as out:
            result = _jobs_command(args)
        self.assertEqual(result, 0)

    @patch("llmflux.cli.get_list_of_jobs_details")
    @patch("llmflux.cli.JobRegistry")
    def test_all_flag_renders_table(self, mock_registry_cls, mock_get_list):
        from llmflux.cli import _jobs_command
        mock_registry_cls.return_value.get_all_jobs.return_value = {
            "111": {"job_name": "flux_job", "model": "llama3", "engine": "vllm", "submitted_at": None}
        }
        mock_get_list.return_value = {
            "111": {"job_id": 111, "job_state": "COMPLETED"}
        }
        args = self._args(all=True)
        with patch("sys.stdout", new=StringIO()) as out:
            result = _jobs_command(args)
            output = out.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("111", output)
        self.assertIn("COMPLETED", output)

    @patch("llmflux.cli.get_active_job_details")
    @patch("llmflux.cli.JobRegistry")
    def test_default_shows_only_active_jobs(self, mock_registry_cls, mock_get_active):
        from llmflux.cli import _jobs_command
        mock_registry_cls.return_value.get_all_jobs.return_value = {
            "222": {"job_name": "active_job", "model": "llama3", "engine": "vllm", "submitted_at": None}
        }
        mock_get_active.return_value = {
            "222": {"job_id": 222, "job_state": "RUNNING"}
        }
        args = self._args(all=False)
        with patch("sys.stdout", new=StringIO()) as out:
            result = _jobs_command(args)
            output = out.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("222", output)
        self.assertIn("RUNNING", output)

    @patch("llmflux.cli.get_list_of_jobs_details")
    @patch("llmflux.cli.JobRegistry")
    def test_state_filter_excludes_non_matching(self, mock_registry_cls, mock_get_list):
        from llmflux.cli import _jobs_command
        mock_registry_cls.return_value.get_all_jobs.return_value = {
            "333": {"job_name": "done", "model": "m", "engine": "e", "submitted_at": None},
            "444": {"job_name": "running", "model": "m", "engine": "e", "submitted_at": None},
        }
        mock_get_list.return_value = {
            "333": {"job_id": 333, "job_state": "COMPLETED"},
            "444": {"job_id": 444, "job_state": "RUNNING"},
        }
        args = self._args(all=True, state=["COMPLETED"])
        with patch("sys.stdout", new=StringIO()) as out:
            result = _jobs_command(args)
            output = out.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("333", output)
        self.assertNotIn("444", output)

    @patch("llmflux.cli.get_active_job_details")
    @patch("llmflux.cli.JobRegistry")
    def test_active_job_not_in_registry_excluded(self, mock_registry_cls, mock_get_active):
        """Slurm reports a job as active but it's not in LLMFlux registry — excluded."""
        from llmflux.cli import _jobs_command
        mock_registry_cls.return_value.get_all_jobs.return_value = {}
        mock_get_active.return_value = {
            "999": {"job_id": 999, "job_state": "RUNNING"}
        }
        args = self._args(all=False)
        with patch("sys.stdout", new=StringIO()) as out:
            result = _jobs_command(args)
        self.assertEqual(result, 0)


class TestStatusCommandIntegration(unittest.TestCase):
    def _args(self, job_id="123"):
        args = MagicMock()
        args.job_id = job_id
        return args

    @patch("llmflux.cli.get_job_details")
    @patch("llmflux.cli.JobRegistry")
    def test_job_not_found_anywhere_returns_error(self, mock_registry_cls, mock_get_details):
        from llmflux.cli import _status_command
        mock_registry_cls.return_value.get_job.return_value = None
        mock_get_details.return_value = {}
        with patch("sys.stderr", new=StringIO()):
            result = _status_command(self._args())
        self.assertEqual(result, 1)

    @patch("llmflux.cli.get_job_details")
    @patch("llmflux.cli.JobRegistry")
    def test_slurm_error_returns_error(self, mock_registry_cls, mock_get_details):
        from llmflux.cli import _status_command
        from llmflux.slurm.commands import SlurmCommandError
        mock_registry_cls.return_value.get_job.return_value = None
        mock_get_details.side_effect = SlurmCommandError("sacct failed")
        with patch("sys.stderr", new=StringIO()):
            result = _status_command(self._args())
        self.assertEqual(result, 1)

    @patch("llmflux.cli.get_job_log_paths")
    @patch("llmflux.cli.get_job_details")
    @patch("llmflux.cli.JobRegistry")
    def test_job_in_slurm_only_shows_details(self, mock_registry_cls, mock_get_details, mock_log_paths):
        from llmflux.cli import _status_command
        mock_registry_cls.return_value.get_job.return_value = None
        mock_get_details.return_value = {
            "job_id": 123,
            "job_state": "COMPLETED",
            "partition": "gpuA100",
        }
        mock_log_paths.return_value = ("/tmp/123.out", "/tmp/123.err")
        with patch("sys.stdout", new=StringIO()) as out:
            result = _status_command(self._args())
            output = out.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("123", output)
        self.assertIn("COMPLETED", output)

    @patch("llmflux.cli.get_job_log_paths")
    @patch("llmflux.cli.get_job_details")
    @patch("llmflux.cli.JobRegistry")
    def test_job_in_both_merges_registry_and_slurm(self, mock_registry_cls, mock_get_details, mock_log_paths):
        from llmflux.cli import _status_command
        mock_registry_cls.return_value.get_job.return_value = {
            "job_name": "my_job",
            "model": "llama3",
            "engine": "vllm",
            "submitted_at": None,
            "workspace": "/work",
            "input": "data.jsonl",
            "output": "out.json",
            "logs_dir": "/logs",
        }
        mock_get_details.return_value = {
            "job_id": 123,
            "job_state": "RUNNING",
            "partition": "gpuA100",
        }
        mock_log_paths.return_value = ("/logs/123.out", "/logs/123.err")
        with patch("sys.stdout", new=StringIO()) as out:
            result = _status_command(self._args())
            output = out.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("llama3", output)
        self.assertIn("vllm", output)
        self.assertIn("RUNNING", output)


class TestCancelCommandIntegration(unittest.TestCase):
    def _args(self, job_id="123", force=False):
        args = MagicMock()
        args.job_id = job_id
        args.force = force
        return args

    @patch("llmflux.cli.JobRegistry")
    def test_job_not_in_registry_returns_error(self, mock_registry_cls):
        from llmflux.cli import _cancel_command
        mock_registry_cls.return_value.get_job.return_value = None
        with patch("sys.stderr", new=StringIO()):
            result = _cancel_command(self._args())
        self.assertEqual(result, 1)

    @patch("llmflux.cli.cancel_job")
    @patch("llmflux.cli.JobRegistry")
    def test_cancel_success(self, mock_registry_cls, mock_cancel):
        from llmflux.cli import _cancel_command
        mock_registry_cls.return_value.get_job.return_value = {"job_name": "job"}
        mock_cancel.return_value = None
        with patch("sys.stdout", new=StringIO()) as out:
            result = _cancel_command(self._args())
            output = out.getvalue()
        self.assertEqual(result, 0)
        self.assertIn("cancelled", output)
        mock_cancel.assert_called_once_with("123", force=False)

    @patch("llmflux.cli.cancel_job")
    @patch("llmflux.cli.JobRegistry")
    def test_cancel_slurm_error_returns_error(self, mock_registry_cls, mock_cancel):
        from llmflux.cli import _cancel_command
        from llmflux.slurm.commands import SlurmCommandError
        mock_registry_cls.return_value.get_job.return_value = {"job_name": "job"}
        mock_cancel.side_effect = SlurmCommandError("permission denied")
        with patch("sys.stderr", new=StringIO()):
            result = _cancel_command(self._args())
        self.assertEqual(result, 1)

    @patch("llmflux.cli.cancel_job")
    @patch("llmflux.cli.JobRegistry")
    def test_force_flag_forwarded(self, mock_registry_cls, mock_cancel):
        from llmflux.cli import _cancel_command
        mock_registry_cls.return_value.get_job.return_value = {"job_name": "job"}
        mock_cancel.return_value = None
        with patch("sys.stdout", new=StringIO()):
            _cancel_command(self._args(force=True))
        mock_cancel.assert_called_once_with("123", force=True)


# ===========================================================================
# 3. Converter utilities — cross-file operations
# ===========================================================================

class TestConverterUtilitiesIntegration(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write(self, filename, lines):
        p = self.d / filename
        p.write_text("\n".join(lines) + "\n")
        return str(p)

    def test_merge_skips_missing_file(self):
        """merge_jsonl_files silently skips a file that does not exist."""
        real = self._write("real.jsonl", ['{"a": 1}', '{"a": 2}'])
        missing = str(self.d / "ghost.jsonl")
        out = str(self.d / "merged.jsonl")
        merge_jsonl_files([real, missing], out)
        lines = [l for l in Path(out).read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)

    def test_merge_skips_invalid_json_lines(self):
        """merge_jsonl_files skips malformed lines, keeps valid ones."""
        f1 = self._write("f1.jsonl", ['{"ok": 1}', "{bad json}", '{"ok": 2}'])
        out = str(self.d / "merged.jsonl")
        merge_jsonl_files([f1], out)
        lines = [l for l in Path(out).read_text().splitlines() if l.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0]), {"ok": 1})
        self.assertEqual(json.loads(lines[1]), {"ok": 2})

    def test_jsonl_to_json_round_trip(self):
        """JSONL written then converted to JSON preserves all entries."""
        entries = [{"id": i, "v": f"val{i}"} for i in range(5)]
        jsonl_path = self.d / "data.jsonl"
        _write_jsonl(str(jsonl_path), entries)
        json_path = str(self.d / "data.json")
        jsonl_to_json(str(jsonl_path), json_path)
        loaded = json.loads(Path(json_path).read_text())
        self.assertEqual(loaded, entries)

    def test_validate_jsonl_with_one_bad_line(self):
        """validate_jsonl returns False when any line is malformed."""
        path = self._write("mixed.jsonl", ['{"good": true}', "not-json", '{"also": "good"}'])
        self.assertFalse(validate_jsonl(path))

    def test_validate_jsonl_all_valid(self):
        entries = [{"id": i} for i in range(4)]
        path = self.d / "valid.jsonl"
        _write_jsonl(str(path), entries)
        self.assertTrue(validate_jsonl(str(path)))


# ===========================================================================
# 4. JSON → JSONL → BatchProcessor full pipeline
# ===========================================================================

class TestJsonToJsonlToBatchProcessorPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.model_config = _make_model_config()

    def tearDown(self):
        self.tmp.cleanup()

    @patch("llmflux.processors.batch.LLMClient")
    def test_json_array_converted_and_processed(self, mock_cls):
        """Raw JSON → json_to_jsonl → BatchProcessor produces one result per entry."""
        raw_data = [
            {"messages": [{"role": "user", "content": f"Question {i}"}]}
            for i in range(3)
        ]
        json_path = self.d / "input.json"
        json_path.write_text(json.dumps(raw_data))
        jsonl_path = self.d / "converted.jsonl"

        convert_result = json_to_jsonl(json_path, jsonl_path, model="test:7b")
        self.assertTrue(convert_result["success"])
        self.assertEqual(convert_result["successful_conversions"], 3)

        mock_client = MagicMock()
        mock_client.chat.return_value = "answer"
        mock_cls.return_value = mock_client

        output_path = str(self.d / "results.json")
        processor = BatchProcessor(model_config=self.model_config)
        results = processor.run(str(jsonl_path), output_path, "vllm")

        self.assertEqual(len(results), 3)
        saved = json.loads(Path(output_path).read_text())
        self.assertEqual(len(saved), 3)

    @patch("llmflux.processors.batch.LLMClient")
    def test_model_field_survives_pipeline(self, mock_cls):
        """model set during json_to_jsonl conversion is used by BatchProcessor."""
        raw_data = [{"messages": [{"role": "user", "content": "hi"}]}]
        json_path = self.d / "input.json"
        json_path.write_text(json.dumps(raw_data))
        jsonl_path = self.d / "converted.jsonl"

        json_to_jsonl(json_path, jsonl_path, model="special-model")

        entry = json.loads(jsonl_path.read_text().strip())
        self.assertEqual(entry["body"]["model"], "special-model")

        mock_client = MagicMock()
        mock_client.chat.return_value = "ok"
        mock_cls.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)
        processor.run(str(jsonl_path), str(self.d / "out.json"), "vllm")

        call_kwargs = mock_client.chat.call_args[1]
        self.assertEqual(call_kwargs["model"], "special-model")

    @patch("llmflux.processors.batch.LLMClient")
    def test_already_batch_format_passthrough(self, mock_cls):
        """Items already in batch format pass through json_to_jsonl unchanged."""
        raw_data = [
            {
                "custom_id": "pre-formatted",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "llama3",
                    "messages": [{"role": "user", "content": "pre-built"}],
                    "temperature": 0.5,
                    "max_tokens": 100,
                },
            }
        ]
        json_path = self.d / "batch_input.json"
        json_path.write_text(json.dumps(raw_data))
        jsonl_path = self.d / "converted.jsonl"

        json_to_jsonl(json_path, jsonl_path)
        entry = json.loads(jsonl_path.read_text().strip())
        self.assertEqual(entry["custom_id"], "pre-formatted")
        self.assertEqual(entry["body"]["model"], "llama3")
        self.assertEqual(entry["body"]["temperature"], 0.5)

        mock_client = MagicMock()
        mock_client.chat.return_value = "response"
        mock_cls.return_value = mock_client

        processor = BatchProcessor(model_config=self.model_config)
        results = processor.run(str(jsonl_path), str(self.d / "out.json"), "vllm")
        self.assertEqual(len(results), 1)
        self.assertIsNotNone(results[0].output)


if __name__ == "__main__":
    unittest.main()