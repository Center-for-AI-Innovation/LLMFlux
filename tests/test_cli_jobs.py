import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmflux import cli
from llmflux.slurm.commands import SlurmCommandError


class _FakeRegistry:
    def __init__(self, jobs=None):
        self._jobs = jobs or {}

    def get_all_jobs(self):
        return self._jobs

    def get_job(self, job_id):
        return self._jobs.get(str(job_id))


class TestCliJobs(unittest.TestCase):
    @patch("llmflux.cli.get_active_job_details")
    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_jobs_defaults_to_active_states(
        self, mock_stdout, mock_registry_cls, mock_get_active_job_details
    ):
        tracked_jobs = {
            "100": {"job_name": "llmflux_a_ollama", "model": "a", "engine": "ollama", "submitted_at": "2026-01-01T00:00:00"},
            "200": {"job_name": "llmflux_b_vllm", "model": "b", "engine": "vllm", "submitted_at": "2026-01-02T00:00:00"},
        }
        mock_registry_cls.return_value = _FakeRegistry(tracked_jobs)
        # squeue only returns job 100 (active); job 200 is absent — it should not appear in output.
        # State is derived by extract_state() directly from this squeue data, not from get_job_state.
        mock_get_active_job_details.return_value = {"100": {"job_state": "RUNNING"}}

        exit_code = cli.main(["jobs"])
        self.assertEqual(exit_code, 0)
        output = mock_stdout.getvalue()
        self.assertIn("100", output)
        self.assertIn("RUNNING", output)
        self.assertNotIn("200", output)

    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_logs_requires_tracked_job(self, mock_stderr, mock_registry_cls):
        mock_registry_cls.return_value = _FakeRegistry({})
        exit_code = cli.main(["logs", "12345"])
        self.assertEqual(exit_code, 1)
        self.assertIn("not tracked by LLMFlux registry", mock_stderr.getvalue())

    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_cancel_requires_tracked_job(self, mock_stderr, mock_registry_cls):
        mock_registry_cls.return_value = _FakeRegistry({})
        exit_code = cli.main(["cancel", "12345"])
        self.assertEqual(exit_code, 1)
        self.assertIn("not tracked by LLMFlux registry", mock_stderr.getvalue())

    @patch("llmflux.cli.get_job_log_paths")
    @patch("llmflux.cli.get_job_details")
    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_status_prints_details_for_tracked_job(
        self, mock_stdout, mock_registry_cls, mock_get_job_details, mock_get_job_log_paths
    ):
        tracked_jobs = {
            "300": {
                "job_name": "llmflux_model_ollama",
                "model": "llama3.2:3b",
                "engine": "ollama",
                "input": "/tmp/input.jsonl",
                "output": "/tmp/output.json",
                "workspace": "/tmp/ws",
                "logs_dir": "/tmp/ws/logs",
                "submitted_at": "2026-01-03T00:00:00",
            }
        }
        mock_registry_cls.return_value = _FakeRegistry(tracked_jobs)
        mock_get_job_details.return_value = {"job_state": "RUNNING", "partition": "gpu"}
        mock_get_job_log_paths.return_value = ("/tmp/ws/logs/300.out", "/tmp/ws/logs/300.err")

        exit_code = cli.main(["status", "300"])
        self.assertEqual(exit_code, 0)
        output = mock_stdout.getvalue()
        self.assertIn("Job ID:", output)
        self.assertIn("llmflux_model_ollama", output)
        self.assertIn("Tip: Run `llmflux logs 300`", output)

    @patch("llmflux.cli.delete")
    @patch("llmflux.cli.get_active_job_details")
    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stderr", new_callable=io.StringIO)
    def test_clean_refuses_while_a_job_is_running(
        self, mock_stderr, mock_registry_cls, mock_get_active_job_details, mock_delete
    ):
        mock_registry_cls.return_value = _FakeRegistry({"100": {"job_name": "llmflux_a_vllm"}})
        mock_get_active_job_details.return_value = {"100": {"job_state": "RUNNING"}}

        exit_code = cli.main(["clean"])

        self.assertEqual(exit_code, 1)
        self.assertIn("still running", mock_stderr.getvalue())
        self.assertIn("llmflux cancel --all", mock_stderr.getvalue())
        mock_delete.assert_not_called()

    @patch("llmflux.cli.delete", return_value=([], [], []))
    @patch("llmflux.cli.get_active_job_details", return_value={})
    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_remove_deletes_when_nothing_is_running(
        self, mock_stdout, mock_registry_cls, _mock_active, mock_delete
    ):
        mock_registry_cls.return_value = _FakeRegistry({"100": {}})
        # Keep the real path list away from real dirs.
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"LLMFLUX_WORKSPACE": tmp}), \
                 patch("llmflux.core.cleanup.Path.home", return_value=Path(tmp)):
                exit_code = cli.main(["remove"])

        self.assertEqual(exit_code, 0)
        mock_delete.assert_called_once()

    @patch("llmflux.cli.cancel_jobs")
    @patch("llmflux.cli.get_active_job_details")
    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cancel_all_cancels_every_running_job(
        self, mock_stdout, mock_registry_cls, mock_get_active_job_details, mock_cancel_jobs
    ):
        mock_registry_cls.return_value = _FakeRegistry({"100": {}, "200": {}})
        mock_get_active_job_details.return_value = {
            "100": {"job_state": "RUNNING"},
            "200": {"job_state": "PENDING"},
        }

        exit_code = cli.main(["cancel", "--all"])

        self.assertEqual(exit_code, 0)
        mock_cancel_jobs.assert_called_once_with(["100", "200"], force=False)

    def _run(self, argv, active=None, active_error=None):
        """Run the CLI with one tracked job (100) and the given sacct result."""
        with patch("llmflux.cli.JobRegistry", return_value=_FakeRegistry({"100": {}})), \
             patch("llmflux.cli.get_active_job_details",
                   return_value=active or {}, side_effect=active_error), \
             patch("sys.stdout", new_callable=io.StringIO), \
             patch("sys.stderr", new_callable=io.StringIO):
            return cli.main(argv)

    def test_cancel_needs_exactly_one_of_job_id_or_all(self):
        self.assertEqual(self._run(["cancel"]), 2)
        self.assertEqual(self._run(["cancel", "123", "--all"]), 2)

    @patch("llmflux.cli.cancel_jobs")
    def test_cancel_all_with_nothing_active(self, mock_cancel_jobs):
        self.assertEqual(self._run(["cancel", "--all"]), 0)
        mock_cancel_jobs.assert_not_called()

    def test_cancel_all_when_slurm_query_fails(self):
        self.assertEqual(self._run(["cancel", "--all"], active_error=SlurmCommandError("down")), 1)

    @patch("llmflux.cli.cancel_jobs", side_effect=SlurmCommandError("scancel failed"))
    def test_cancel_all_when_scancel_fails(self, _mock_cancel_jobs):
        self.assertEqual(self._run(["cancel", "--all"], active={"100": {"job_state": "RUNNING"}}), 1)

    @patch("llmflux.cli.cancel_jobs")
    def test_cancel_all_forwards_force(self, mock_cancel_jobs):
        self._run(["cancel", "--all", "--force"], active={"100": {"job_state": "RUNNING"}})
        mock_cancel_jobs.assert_called_once_with(["100"], force=True)

    @patch("llmflux.cli.delete")
    def test_clean_when_slurm_query_fails(self, mock_delete):
        self.assertEqual(self._run(["clean"], active_error=SlurmCommandError("down")), 1)
        mock_delete.assert_not_called()

    @patch("llmflux.cli.delete", return_value=([], [], ["logs/1.out: Permission denied"]))
    def test_clean_exits_1_when_a_delete_fails(self, _mock_delete):
        self.assertEqual(self._run(["clean"]), 1)

    @patch("llmflux.cli.Path.exists", return_value=False)
    @patch("llmflux.cli.get_job_state", return_value="PENDING")
    @patch("llmflux.cli.get_job_log_paths")
    @patch("llmflux.cli.JobRegistry")
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_logs_informs_user_when_pending_and_logs_missing(
        self,
        mock_stdout,
        mock_registry_cls,
        mock_get_job_log_paths,
        _mock_get_job_state,
        _mock_path_exists,
    ):
        mock_registry_cls.return_value = _FakeRegistry({"400": {"logs_dir": "/tmp/llmflux-logs"}})
        mock_get_job_log_paths.return_value = ("/tmp/llmflux-logs/400.out", "/tmp/llmflux-logs/400.err")

        exit_code = cli.main(["logs", "400"])
        self.assertEqual(exit_code, 0)
        self.assertIn("is PENDING", mock_stdout.getvalue())
