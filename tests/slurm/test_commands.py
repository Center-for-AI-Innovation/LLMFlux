import subprocess
import unittest
from unittest.mock import patch

from llmflux.slurm.commands import (
    SlurmCommandError,
    cancel_job,
    extract_state,
    get_active_job_details,
    get_job_details,
    get_job_log_paths,
)


def _completed(stdout: str, returncode: int = 0, stderr: str = ""):
    return subprocess.CompletedProcess(args=[], returncode=returncode, stdout=stdout, stderr=stderr)


class TestSlurmCommands(unittest.TestCase):
    @patch("llmflux.slurm.commands.subprocess.run")
    def test_get_active_job_details_parses_json(self, mock_run):
        # job_id must be an integer — _build_job_index skips string job_ids
        mock_run.return_value = _completed('{"jobs":[{"job_id":12345,"job_state":"RUNNING"}]}')
        jobs = get_active_job_details(user="tester")
        self.assertIn("12345", jobs)
        self.assertEqual(jobs["12345"]["job_state"], "RUNNING")

    @patch("llmflux.slurm.commands.subprocess.run")
    def test_get_job_details_filters_batch_step(self, mock_run):
        # First call is squeue (empty), second is sacct with base + batch entries
        mock_run.side_effect = [
            _completed('{"jobs":[]}'),
            _completed(
                '{"jobs":['
                '{"job_id":11111,"state":"COMPLETED"},'
                '{"job_id":"11111.batch","state":"COMPLETED"}'
                ']}'
            ),
        ]
        job = get_job_details("11111")
        # Only the base job (integer job_id) should be returned
        self.assertEqual(job.get("job_id"), 11111)

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

    def test_extract_state_plain_string(self):
        self.assertEqual(extract_state({"job_state": "running"}), "RUNNING")

    def test_extract_state_nested_dict(self):
        # Slurm 25+ sacct JSON schema: state is an object with a "current" list
        self.assertEqual(
            extract_state({"state": {"current": ["COMPLETED"], "reason": "None"}}),
            "COMPLETED",
        )
