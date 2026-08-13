"""Execute the generated SLURM scripts under stubbed external commands.

Every other test in this suite asserts on the *text* of the generated script.
Text assertions prove a line is present; they cannot prove the script's control
flow reaches it, exits non-zero, or terminates at all. A backgrounded watchdog
that references a variable set after the fork, an `|| exit 1` attached to a
command that cannot fail, a readiness loop whose bound exceeds the allocation —
all of those pass a text assertion and hang or silently succeed on a node.

So: put fakes for every external command on PATH, run the script with bash, and
assert on exit status, elapsed time and output markers.

SAFETY. The generated script calls `pkill -f "vllm serve"` and `rm -rf` on
directories from its environment. Run unsandboxed on a shared login node that
would kill other users' processes. Everything here is confined:

  * every external command is a stub on a PATH prepended to the environment;
    `pkill` in particular is a stub that records its arguments and kills nothing
  * every directory the script is told to use lives under one temp dir
  * `sleep` is a no-op stub, so a 300-iteration readiness loop runs instantly
    (stubs that must stay alive use /bin/sleep by absolute path to bypass it)

Stub behaviour is driven by environment variables so a test can choose a failure
mode: STUB_CURL_READY_AFTER, STUB_SERVER_MODE, STUB_PYTHON_EXIT.
"""

import os
import shutil
import signal
import stat
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock

BASH = shutil.which("bash")

# Stubs. Each logs its invocation to $STUB_LOG so tests can assert on calls.
_STUBS = {
    # Records the call and kills nothing. Never let the real pkill run.
    "pkill": """#!/bin/bash
echo "pkill $*" >> "$STUB_LOG"
exit 0
""",
    # Fails STUB_CURL_READY_AFTER times, then succeeds — models a server that
    # takes a while to come up.
    "curl": """#!/bin/bash
echo "curl $*" >> "$STUB_LOG"
n=$(cat "$STUB_STATE/curl_calls" 2>/dev/null || echo 0)
n=$((n + 1)); echo "$n" > "$STUB_STATE/curl_calls"
[ "$n" -gt "${STUB_CURL_READY_AFTER:-0}" ] && exit 0
exit 7
""",
    # Backgrounded as the inference server. 'stay' lives until killed; 'die'
    # exits immediately, which is the server-died path.
    #
    # The redirect matters: this is backgrounded by the script and inherits the
    # harness's stdout pipe. Holding that pipe open keeps the parent's read()
    # blocked long after bash exits, so the test hangs for the stub's whole
    # lifetime instead of finishing when the script does.
    "apptainer": """#!/bin/bash
echo "apptainer $*" >> "$STUB_LOG"
if [ "${STUB_SERVER_MODE:-stay}" = "die" ]; then exit 1; fi
exec /bin/sleep 120 >/dev/null 2>&1 </dev/null
""",
    # The inline batch processor.
    "python3": """#!/bin/bash
echo "python3 $*" >> "$STUB_LOG"
exit "${STUB_PYTHON_EXIT:-0}"
""",
    # No-op so readiness loops run instantly.
    "sleep": """#!/bin/bash
exit 0
""",
    "ss": """#!/bin/bash
echo "ss $*" >> "$STUB_LOG"
exit 0
""",
    "mail": """#!/bin/bash
echo "mail $*" >> "$STUB_LOG"
exit 0
""",
    "nvidia-smi": """#!/bin/bash
echo "nvidia-smi $*" >> "$STUB_LOG"
exit 0
""",
}


def _slurm_config():
    cfg = MagicMock()
    cfg.extra_sbatch_args = None
    return cfg


def _build(engine, mode, **over):
    if engine == "vllm":
        from llmflux.slurm.engine.vllm import create_vllm_batch_script as maker
    else:
        from llmflux.slurm.engine.ollama import create_ollama_batch_script as maker

    kwargs = dict(
        account="acct", partition="gpu", nodes="1", gpus_per_node="2",
        time="01:00:00", memory="64G", cpus_per_task="8",
        logs_dir=Path("/logs"), input_file=Path("/data/in.jsonl"),
        output_file=Path("/data/out.json"), job_name=f"exec-{engine}-{mode}",
        slurm_config=_slurm_config(), mode=mode,
    )
    if mode == "serve":
        kwargs["email"] = "someone@example.edu"
    kwargs.update(over)
    return "\n".join(maker(**kwargs)) + "\n"


class Sandbox:
    """Temp dir holding stub binaries, script, and every directory the script uses."""

    def __enter__(self):
        self.root = Path(tempfile.mkdtemp(prefix="llmflux-exec-"))
        binp = self.root / "bin"
        binp.mkdir()
        for name, body in _STUBS.items():
            p = binp / name
            p.write_text(body)
            p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

        state = self.root / "state"
        state.mkdir()
        self.log = self.root / "stub.log"
        self.log.touch()

        work = self.root / "work"
        for sub in ("in", "out", "models", "logs", "containers", "tmp",
                    "cache", "hf", "vllm", "ollama", "xdg"):
            (work / sub).mkdir(parents=True, exist_ok=True)
        # The script only builds the container when the .sif is absent.
        (work / "containers" / "llm_processor.sif").write_text("fake sif\n")

        self.env = dict(os.environ)
        self.env.update({
            "PATH": f"{binp}:{os.environ['PATH']}",
            "STUB_LOG": str(self.log),
            "STUB_STATE": str(state),
            # Directories the generated script mkdir's, binds and rm -rf's.
            "DATA_INPUT_DIR": str(work / "in"),
            "DATA_OUTPUT_DIR": str(work / "out"),
            "MODELS_DIR": str(work / "models"),
            "LOGS_DIR": str(work / "logs"),
            "CONTAINERS_DIR": str(work / "containers"),
            "CONTAINER_DEF": str(work / "container.def"),
            "APPTAINER_TMPDIR": str(work / "tmp"),
            "APPTAINER_CACHEDIR": str(work / "cache"),
            "HF_HOME": str(work / "hf"),
            "VLLM_HOME": str(work / "vllm"),
            "VLLM_MODELS": str(work / "vllm"),
            "OLLAMA_HOME": str(work / "ollama"),
            "OLLAMA_MODELS": str(work / "ollama"),
            "XDG_CACHE_HOME": str(work / "xdg"),
            "FLASHINFER_WORKSPACE_BASE": str(work / "xdg"),
            "PROJECT_ROOT": str(work),
            "VLLM_MODEL_NAME": "fake/model",
            "VLLM_HOST": "0.0.0.0",
            "VLLM_PORT": "8000",
            "VLLM_ENGINE_ARGS": "",
            "OLLAMA_MODEL_NAME": "fake:1b",
            "OLLAMA_PORT": "8000",
            "LLMFLUX_API_KEY": "test-key",
            "LLMFLUX_FORCE_REBUILD": "0",
            "SLURM_JOB_ID": "12345",
        })
        return self

    def __exit__(self, *exc):
        shutil.rmtree(self.root, ignore_errors=True)

    def run(self, script_text, timeout=60, **env_over):
        """Run the script in its own process group; kill the whole group after.

        The script backgrounds a long-lived "server". Without the group kill a
        failed or early-exiting script leaks that process, which then outlives
        the test run — observed leaking /bin/sleep onto a shared login node.
        """
        path = self.root / "job.sh"
        path.write_text(script_text)
        env = dict(self.env)
        env.update({k: str(v) for k, v in env_over.items()})

        t0 = time.monotonic()
        proc = subprocess.Popen(
            [BASH, str(path)], env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        pgid = os.getpgid(proc.pid)
        try:
            out, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._kill_group(pgid)
            out, _ = proc.communicate()
            raise
        finally:
            self._kill_group(pgid)
        return proc.returncode, out.decode(errors="replace"), time.monotonic() - t0

    @staticmethod
    def _kill_group(pgid):
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass

    def calls(self):
        return self.log.read_text().splitlines()


@unittest.skipIf(BASH is None, "bash not available")
class TestVllmBatchExecution(unittest.TestCase):
    """Characterizes today's control flow, so changes to it are visible."""

    def test_happy_path_exits_zero_and_runs_processor(self):
        with Sandbox() as sb:
            rc, out, _ = sb.run(_build("vllm", "batch"), STUB_CURL_READY_AFTER=2)
            self.assertEqual(rc, 0, out[-2000:])
            self.assertIn("Server has started!", out)
            self.assertTrue(
                any(c.startswith("python3") for c in sb.calls()),
                "batch processor was never invoked",
            )

    def test_server_death_during_load_fails_fast(self):
        """Server exits before ready -> non-zero, and quickly."""
        with Sandbox() as sb:
            rc, out, elapsed = sb.run(
                _build("vllm", "batch"),
                STUB_SERVER_MODE="die",
                STUB_CURL_READY_AFTER=10**9,  # never ready: force the loop to the ps check
            )
            self.assertNotEqual(rc, 0, "a dead server must not exit 0")
            self.assertIn("VLLM server died", out)
            self.assertLess(elapsed, 30, "should fail fast, not spin the full loop")
            self.assertFalse(
                any(c.startswith("python3") for c in sb.calls()),
                "processor must not run against a dead server",
            )

    def test_readiness_timeout_is_bounded_and_fails(self):
        """Server never becomes ready -> the loop must end, non-zero."""
        with Sandbox() as sb:
            rc, out, _ = sb.run(
                _build("vllm", "batch"), STUB_CURL_READY_AFTER=10**9, timeout=120
            )
            self.assertNotEqual(rc, 0, "an unready server must not exit 0")
            self.assertIn("Server failed to load!", out)

    def test_processor_failure_is_currently_swallowed(self):
        """Characterizes a known defect: a failing processor still exits 0.

        The inline `python3 -c` exit status is not checked, so a run whose every
        item failed produces a complete-looking output file and a successful
        job. This assertion documents today's behaviour; the fix flips it, and
        this test flips with it.
        """
        with Sandbox() as sb:
            rc, _, _ = sb.run(
                _build("vllm", "batch"), STUB_CURL_READY_AFTER=1, STUB_PYTHON_EXIT=1
            )
            self.assertEqual(
                rc, 0, "if this now fails, the processor-exit-code fix has landed — "
                "update this test to assert non-zero"
            )


@unittest.skipIf(BASH is None, "bash not available")
class TestOllamaBatchExecution(unittest.TestCase):
    def test_happy_path_exits_zero(self):
        with Sandbox() as sb:
            rc, out, _ = sb.run(_build("ollama", "batch"), STUB_CURL_READY_AFTER=1)
            self.assertEqual(rc, 0, out[-2000:])
            self.assertTrue(any(c.startswith("python3") for c in sb.calls()))

    def test_server_death_fails_fast(self):
        with Sandbox() as sb:
            rc, out, elapsed = sb.run(
                _build("ollama", "batch"),
                STUB_SERVER_MODE="die",
                STUB_CURL_READY_AFTER=10**9,
            )
            self.assertNotEqual(rc, 0)
            self.assertIn("Ollama server died", out)
            self.assertLess(elapsed, 30)


@unittest.skipIf(BASH is None, "bash not available")
class TestSandboxSafety(unittest.TestCase):
    """The harness must not be able to touch anything outside its temp dir."""

    def test_pkill_is_stubbed_and_kills_nothing(self):
        with Sandbox() as sb:
            sb.run(_build("vllm", "batch"), STUB_CURL_READY_AFTER=1)
            pkills = [c for c in sb.calls() if c.startswith("pkill")]
            self.assertTrue(pkills, "expected the script to attempt a pkill")
            self.assertEqual(
                shutil.which("pkill", path=str(sb.root / "bin")),
                str(sb.root / "bin" / "pkill"),
                "pkill must resolve to the stub, never the system binary",
            )

    def test_script_only_removes_paths_inside_the_sandbox(self):
        with Sandbox() as sb:
            work = Path(sb.env["APPTAINER_TMPDIR"])
            self.assertTrue(work.exists())
            sb.run(_build("vllm", "batch"), STUB_CURL_READY_AFTER=1)
            # The script rm -rf's its tmpdir; that must be inside the sandbox.
            self.assertTrue(str(work).startswith(str(sb.root)))
            self.assertTrue(sb.root.exists(), "sandbox root itself must survive")

    def test_stub_path_precedes_system_path(self):
        with Sandbox() as sb:
            first = sb.env["PATH"].split(os.pathsep)[0]
            self.assertEqual(first, str(sb.root / "bin"))


if __name__ == "__main__":
    unittest.main()
