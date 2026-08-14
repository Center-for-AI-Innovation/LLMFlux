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

class TestGpuVisibilityParity(unittest.TestCase):
    """Both engines must pass SLURM's real device list into the container.

    `apptainer --cleanenv` strips CUDA_VISIBLE_DEVICES, so each engine script
    has to re-export it. vllm.py always did; ollama.py did not, so Ollama
    received the list synthesised from the *requested* GPU count
    (runner.py: `range(gpus_per_node)`) rather than the devices actually
    granted. That is only correct when granted indices happen to start at 0 and
    be contiguous, and Ollama itself warns about it:

        WARN "user overrode visible devices" CUDA_VISIBLE_DEVICES=0,1,2,3
        WARN "if GPUs are not correctly discovered, unset and try again"
    """

    EXPECTED = "export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"

    def _script(self, engine, mode="batch"):
        from pathlib import Path
        if engine == "vllm":
            from llmflux.slurm.engine.vllm import create_vllm_batch_script as maker
        else:
            from llmflux.slurm.engine.ollama import create_ollama_batch_script as maker
        kwargs = dict(
            account="a", partition="p", nodes="1", gpus_per_node="4",
            time="01:00:00", memory="64G", cpus_per_task="8",
            logs_dir=Path("/logs"), input_file=Path("/in.jsonl"),
            output_file=Path("/out.json"), job_name="j",
            slurm_config=_make_slurm_config(), mode=mode,
        )
        if mode == "serve":
            kwargs["email"] = "x@example.edu"
        return maker(**kwargs)

    def test_both_engines_export_the_real_device_list(self):
        for engine in ("vllm", "ollama"):
            for mode in ("batch", "serve"):
                with self.subTest(engine=engine, mode=mode):
                    self.assertIn(self.EXPECTED, self._script(engine, mode))

    def test_export_precedes_the_container_exec(self):
        """An export after the exec would have no effect on it."""
        for engine in ("vllm", "ollama"):
            with self.subTest(engine=engine):
                script = self._script(engine)
                export_at = script.index(self.EXPECTED)
                exec_at = next(
                    i for i, line in enumerate(script) if "apptainer exec" in line
                )
                self.assertLess(export_at, exec_at)


class TestServeTeardownParity(unittest.TestCase):
    """Both engines must kill their own server by PID, not only by pattern.

    `pkill -f "<engine> serve"` is not job-scoped: it matches the same user's
    other jobs on a shared node. For ollama it was the ONLY thing stopping the
    server ($OLLAMA_PID was never killed at all); for vLLM it matched nothing at
    all on the single-node path, which launches
    `python3 -m vllm.entrypoints.openai.api_server`.

    The two engine scripts drifting apart is a recurring defect here, so this is
    written as parity rather than as a per-engine detail.
    """

    CASES = [("vllm", "VLLM_PID"), ("ollama", "OLLAMA_PID")]

    def _trap(self, engine):
        from tests.slurm.helpers import build_text
        script = build_text(engine, "serve", nodes="1")
        return next(l for l in script.splitlines() if l.startswith("trap "))

    def test_trap_kills_the_server_pid(self):
        for engine, pid in self.CASES:
            with self.subTest(engine=engine):
                trap = self._trap(engine)
                self.assertIn(f'kill "${pid}"', trap)
                self.assertIn("$CONNECTION_FILE", trap, "the key-bearing file must still go")

    def test_trap_guards_the_pid_for_an_early_exit(self):
        """The trap is installed before the launch, so it can fire with the
        variable genuinely unset — a failed container build, say."""
        for engine, pid in self.CASES:
            with self.subTest(engine=engine):
                self.assertIn(f'[ -n "${{{pid}:-}}" ]', self._trap(engine))

    def test_trap_fires_with_the_pid_set(self):
        """The trap body is single-quoted, so ${PID:-} expands when the trap
        FIRES, not when it is installed. Asserted by running the real body."""
        import subprocess
        for engine, pid in self.CASES:
            with self.subTest(engine=engine):
                body = self._trap(engine)
                body = body[len("trap '"):body.rindex("'")]
                body = body.replace("$CONNECTION_FILE", "/dev/null")
                script = (
                    "pkill() { echo PKILL; }\n"
                    f"kill() {{ echo KILLED $1; }}\n"
                    f"trap '{body}' EXIT\n"
                    f"{pid}=4242\n"
                )
                out = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
                self.assertIn("KILLED 4242", out.stdout)


class TestServeExitStatus(unittest.TestCase):
    """A serve job whose deployment collapses must not report success.

    Without this the script's last command is `kill -9 ... || true`, so the exit
    status is unconditionally 0: sacct says COMPLETED, `#SBATCH --mail-type=FAIL`
    never fires, and `llmflux connect` only reports that the file is gone. At
    nodes>1 `srun --kill-on-bad-exit=1` makes this status the only in-band signal
    that a remote rank died.
    """

    def _script(self, nodes):
        from tests.slurm.helpers import build_text
        return build_text("vllm", "serve", nodes=nodes)

    def test_serve_exits_with_the_server_status(self):
        for nodes in ("1", "2"):
            with self.subTest(nodes=nodes):
                lines = [l for l in self._script(nodes).splitlines() if l.strip()]
                self.assertEqual(lines[-1], "exit ${LLMFLUX_SERVE_RC:-0}")

    def test_the_status_is_captured_immediately_after_wait(self):
        """Anything between `wait` and the capture overwrites $?."""
        for nodes in ("1", "2"):
            with self.subTest(nodes=nodes):
                lines = self._script(nodes).splitlines()
                i = next(i for i, l in enumerate(lines) if l.startswith("wait $VLLM_PID"))
                self.assertEqual(lines[i + 1], "LLMFLUX_SERVE_RC=$?")

    def test_a_cancelled_job_is_not_a_failure(self):
        """scancel and Ctrl-C are how a serve job normally ends. Run the real
        emitted case statement rather than asserting its text."""
        import subprocess
        case_line = next(
            l for l in self._script("2").splitlines()
            if l.startswith('case "$LLMFLUX_SERVE_RC"')
        )
        for rc, expected in (("0", "0"), ("130", "0"), ("143", "0"), ("1", "1"), ("137", "137")):
            with self.subTest(rc=rc):
                out = subprocess.run(
                    ["bash", "-c", f'LLMFLUX_SERVE_RC={rc}\n{case_line}\necho "$LLMFLUX_SERVE_RC"'],
                    capture_output=True, text=True,
                )
                self.assertEqual(out.stdout.strip(), expected)


if __name__ == "__main__":
    unittest.main()
