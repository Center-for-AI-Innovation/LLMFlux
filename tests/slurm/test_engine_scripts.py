"""Tests for SLURM batch script generation (vllm and ollama engines)."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock


def _make_slurm_config(extra_args=None):
    cfg = MagicMock()
    cfg.extra_sbatch_args = extra_args
    return cfg


class TestCreateVllmBatchScript(unittest.TestCase):
    def _call(self, extra_sbatch_args=None, **kwargs):
        from llmflux.slurm.engine.vllm import create_vllm_batch_script

        defaults = dict(
            account="myaccount",
            partition="gpu",
            nodes="1",
            gpus_per_node="2",
            time="01:00:00",
            memory="64G",
            cpus_per_task="8",
            logs_dir=Path("/logs"),
            input_file=Path("/data/in.jsonl"),
            output_file=Path("/data/out.json"),
            job_name="test-job",
            slurm_config=_make_slurm_config(extra_sbatch_args),
        )
        defaults.update(kwargs)
        return create_vllm_batch_script(**defaults)

    def test_returns_list(self):
        script = self._call()
        self.assertIsInstance(script, list)
        self.assertGreater(len(script), 0)

    def test_shebang_first_line(self):
        self.assertEqual(self._call()[0], "#!/bin/bash")

    def test_sbatch_job_name(self):
        script = self._call()
        self.assertIn("#SBATCH --job-name=test-job", script)

    def test_sbatch_account(self):
        self.assertIn("#SBATCH --account=myaccount", self._call())

    def test_sbatch_partition(self):
        self.assertIn("#SBATCH --partition=gpu", self._call())

    def test_sbatch_gpus_per_node(self):
        self.assertIn("#SBATCH --gpus-per-node=2", self._call())

    def test_sbatch_time(self):
        self.assertIn("#SBATCH --time=01:00:00", self._call())

    def test_sbatch_memory(self):
        self.assertIn("#SBATCH --mem=64G", self._call())

    def test_sbatch_cpus_per_task(self):
        self.assertIn("#SBATCH --cpus-per-task=8", self._call())

    def test_log_paths_use_logs_dir(self):
        script = self._call()
        self.assertTrue(any("/logs/%j.out" in line for line in script))
        self.assertTrue(any("/logs/%j.err" in line for line in script))

    def test_input_file_embedded_in_script(self):
        script = self._call()
        full = "\n".join(script)
        self.assertIn("/data/in.jsonl", full)

    def test_output_file_embedded_in_script(self):
        script = self._call()
        full = "\n".join(script)
        self.assertIn("/data/out.json", full)

    def test_extra_sbatch_args_included(self):
        script = self._call(extra_sbatch_args={"reservation": "myres", "qos": "high"})
        self.assertIn("#SBATCH --reservation=myres", script)
        self.assertIn("#SBATCH --qos=high", script)

    def test_no_extra_sbatch_args_when_none(self):
        script = self._call(extra_sbatch_args=None)
        self.assertFalse(any("reservation" in line for line in script))


class TestCreateOllamaBatchScript(unittest.TestCase):
    def _call(self, extra_sbatch_args=None, **kwargs):
        from llmflux.slurm.engine.ollama import create_ollama_batch_script

        defaults = dict(
            account="myaccount",
            partition="gpu",
            nodes="1",
            gpus_per_node="1",
            time="00:30:00",
            memory="32G",
            cpus_per_task="4",
            logs_dir=Path("/logs"),
            input_file=Path("/data/in.jsonl"),
            output_file=Path("/data/out.json"),
            job_name="ollama-job",
            slurm_config=_make_slurm_config(extra_sbatch_args),
        )
        defaults.update(kwargs)
        return create_ollama_batch_script(**defaults)

    def test_returns_list(self):
        script = self._call()
        self.assertIsInstance(script, list)
        self.assertGreater(len(script), 0)

    def test_shebang_first_line(self):
        self.assertEqual(self._call()[0], "#!/bin/bash")

    def test_sbatch_job_name(self):
        self.assertIn("#SBATCH --job-name=ollama-job", self._call())

    def test_sbatch_account(self):
        self.assertIn("#SBATCH --account=myaccount", self._call())

    def test_sbatch_partition(self):
        self.assertIn("#SBATCH --partition=gpu", self._call())

    def test_log_paths_use_logs_dir(self):
        script = self._call()
        self.assertTrue(any("/logs/%j.out" in line for line in script))
        self.assertTrue(any("/logs/%j.err" in line for line in script))

    def test_input_file_embedded_in_script(self):
        full = "\n".join(self._call())
        self.assertIn("/data/in.jsonl", full)

    def test_output_file_embedded_in_script(self):
        full = "\n".join(self._call())
        self.assertIn("/data/out.json", full)

    def test_extra_sbatch_args_included(self):
        script = self._call(extra_sbatch_args={"reservation": "myres"})
        self.assertIn("#SBATCH --reservation=myres", script)

    def test_no_extra_sbatch_args_when_none(self):
        script = self._call(extra_sbatch_args=None)
        self.assertFalse(any("reservation" in line for line in script))

    def test_ollama_serve_present(self):
        full = "\n".join(self._call())
        self.assertIn("ollama serve", full)


if __name__ == "__main__":
    unittest.main()