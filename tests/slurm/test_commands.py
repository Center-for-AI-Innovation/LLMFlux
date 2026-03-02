import subprocess
import unittest
from unittest.mock import patch

from llmflux.slurm.commands import (
    SlurmCommandError,
    cancel_job,
    get_active_jobs,
    get_historical_jobs,
    get_job_log_paths,
    normalize_state,
)


def _completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestSlurmCommands(unittest.TestCase):
    @patch("llmflux.slurm.commands.subprocess.run")
    def test_get_active_jobs_parses_json(self, mock_run):
        mock_run.return_value = _completed('{"jobs":[{"job_id":"12345","job_state":"RUNNING"}]}')
        jobs = get_active_jobs(user="tester")
        self.assertIn("12345", jobs)
        self.assertEqual(jobs["12345"]["job_state"], "RUNNING")

    @patch("llmflux.slurm.commands.subprocess.run")
    def test_get_historical_jobs_filters_batch_rows(self, mock_run):
        mock_run.return_value = _completed(
            '{"jobs":[{"job_id":"11111","state":"COMPLETED"},{"job_id":"11111.batch","state":"COMPLETED"}]}'
        )
        jobs = get_historical_jobs(user="tester")
        self.assertIn("11111", jobs)
        self.assertTrue(all(".batch" not in job_id for job_id in jobs))

    @patch("llmflux.slurm.commands.subprocess.run")
    def test_cancel_job_raises_on_failure(self, mock_run):
        mock_run.return_value = _completed("", returncode=1, stderr="permission denied")
        with self.assertRaises(SlurmCommandError):
            cancel_job("999")

    @patch("llmflux.slurm.commands.get_job_details")
    def test_get_job_log_paths_falls_back_to_logs_dir(self, mock_get_job_details):
        mock_get_job_details.return_value = {}
        stdout_path, stderr_path = get_job_log_paths("123", "/tmp/logs")
        self.assertEqual(stdout_path, "/tmp/logs/123.out")
        self.assertEqual(stderr_path, "/tmp/logs/123.err")

    def test_normalize_state(self):
        self.assertEqual(normalize_state({"job_state": "running"}), "RUNNING")
