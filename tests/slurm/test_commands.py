import subprocess
import unittest
from unittest.mock import patch

from llmflux.slurm.commands import (
    SlurmCommandError,
    _extract_jobs,
    _parse_state_value,
    _run_json_command,
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
        mock_run.return_value = _completed(
            '{"jobs":['
            '{"job_id":11111,"state":"COMPLETED"},'
            '{"job_id":"11111.batch","state":"COMPLETED"}'
            ']}'
        )
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

    def test_extract_state_no_known_key(self):
        self.assertEqual(extract_state({}), "UNKNOWN")


class TestRunJsonCommand(unittest.TestCase):
    @patch("llmflux.slurm.commands.subprocess.run")
    def test_raises_on_nonzero_returncode(self, mock_run):
        mock_run.return_value = _completed("", returncode=1, stderr="oops")
        with self.assertRaises(SlurmCommandError):
            _run_json_command(["sacct", "--json"])

    @patch("llmflux.slurm.commands.subprocess.run")
    def test_empty_stdout_returns_empty_dict(self, mock_run):
        mock_run.return_value = _completed("")
        result = _run_json_command(["sacct", "--json"])
        self.assertEqual(result, {})

    @patch("llmflux.slurm.commands.subprocess.run")
    def test_invalid_json_raises(self, mock_run):
        mock_run.return_value = _completed("not json at all")
        with self.assertRaises(SlurmCommandError):
            _run_json_command(["sacct", "--json"])

    @patch("llmflux.slurm.commands.subprocess.run")
    def test_valid_json_returned(self, mock_run):
        mock_run.return_value = _completed('{"jobs": []}')
        result = _run_json_command(["sacct", "--json"])
        self.assertEqual(result, {"jobs": []})


class TestExtractJobs(unittest.TestCase):
    def test_non_dict_payload_returns_empty(self):
        self.assertEqual(_extract_jobs([]), [])
        self.assertEqual(_extract_jobs("string"), [])
        self.assertEqual(_extract_jobs(None), [])

    def test_jobs_not_a_list_returns_empty(self):
        self.assertEqual(_extract_jobs({"jobs": "oops"}), [])

    def test_filters_non_dict_entries(self):
        payload = {"jobs": [{"job_id": 1}, "not-a-dict", {"job_id": 2}]}
        result = _extract_jobs(payload)
        self.assertEqual(len(result), 2)

    def test_empty_jobs_list(self):
        self.assertEqual(_extract_jobs({"jobs": []}), [])


class TestParseStateValue(unittest.TestCase):
    def test_dict_with_current_list(self):
        self.assertEqual(_parse_state_value({"current": ["RUNNING"]}), "RUNNING")

    def test_dict_with_current_string(self):
        self.assertEqual(_parse_state_value({"current": "pending"}), "PENDING")

    def test_dict_with_CURRENT_uppercase_key(self):
        self.assertEqual(_parse_state_value({"CURRENT": ["FAILED"]}), "FAILED")

    def test_dict_with_empty_current_list(self):
        self.assertIsNone(_parse_state_value({"current": []}))

    def test_dict_missing_current_key(self):
        self.assertIsNone(_parse_state_value({"reason": "None"}))

    def test_list_single_element(self):
        self.assertEqual(_parse_state_value(["COMPLETED"]), "COMPLETED")

    def test_list_empty(self):
        self.assertIsNone(_parse_state_value([]))

    def test_plain_string(self):
        self.assertEqual(_parse_state_value("running"), "RUNNING")

    def test_blank_string(self):
        self.assertIsNone(_parse_state_value("   "))

    def test_none_input(self):
        self.assertIsNone(_parse_state_value(None))


class TestCancelJob(unittest.TestCase):
    @patch("llmflux.slurm.commands.subprocess.run")
    def test_force_flag_adds_signal(self, mock_run):
        mock_run.return_value = _completed("", returncode=0)
        cancel_job("123", force=True)
        cmd = mock_run.call_args[0][0]
        self.assertIn("--signal=KILL", cmd)

    @patch("llmflux.slurm.commands.subprocess.run")
    def test_no_force_omits_signal(self, mock_run):
        mock_run.return_value = _completed("", returncode=0)
        cancel_job("123", force=False)
        cmd = mock_run.call_args[0][0]
        self.assertNotIn("--signal=KILL", cmd)
