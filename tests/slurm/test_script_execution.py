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

from .helpers import build_text

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
    # Emulates SPMD placement: runs the rank script once per node with
    # SLURM_PROCID set. STUB_SRUN_MODE selects the failure being tested.
    "srun": """#!/bin/bash
echo "srun $*" >> "$STUB_LOG"
script="${@: -1}"
case "${STUB_SRUN_MODE:-ok}" in
  die)      exit 1 ;;
  partial)  n=1 ;;
  none)     exec /bin/sleep 120 >/dev/null 2>&1 </dev/null ;;
  *)        n="${LLMFLUX_NNODES:-1}" ;;
esac
i=0
while [ "$i" -lt "$n" ]; do
  SLURM_PROCID="$i" "$script" &
  i=$((i + 1))
done
wait
""",
    "ip": """#!/bin/bash
echo "ip $*" >> "$STUB_LOG"
echo "2: hsn0    inet ${STUB_NODE_IP:-172.28.86.64}/21 brd x scope global hsn0"
echo "3: hsn0.561    inet 141.142.254.64/21 brd x scope global hsn0.561"
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
            rc, out, _ = sb.run(build_text("vllm", "batch"), STUB_CURL_READY_AFTER=2)
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
                build_text("vllm", "batch"),
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
                build_text("vllm", "batch"), STUB_CURL_READY_AFTER=10**9, timeout=120
            )
            self.assertNotEqual(rc, 0, "an unready server must not exit 0")
            self.assertIn("Server failed to load!", out)

    def test_processor_failure_propagates_to_the_job(self):
        """A failing batch processor must fail the job.

        Previously the inline `python3 -c` status was discarded and the script
        exited with whatever cleanup returned, so a run whose every item failed
        still reported success — a complete-looking output file and a green job.
        """
        with Sandbox() as sb:
            rc, _, _ = sb.run(
                build_text("vllm", "batch"), STUB_CURL_READY_AFTER=1, STUB_PYTHON_EXIT=1
            )
            self.assertEqual(rc, 1, "processor failure must not be swallowed")

    def test_processor_success_still_exits_zero(self):
        with Sandbox() as sb:
            rc, _, _ = sb.run(
                build_text("vllm", "batch"), STUB_CURL_READY_AFTER=1, STUB_PYTHON_EXIT=0
            )
            self.assertEqual(rc, 0)

    def test_processor_status_survives_cleanup(self):
        """Cleanup runs between the processor and the exit, and must not mask it.

        pkill/rm/kill all overwrite $?, which is exactly how the status was lost.
        """
        with Sandbox() as sb:
            rc, out, _ = sb.run(
                build_text("vllm", "batch"), STUB_CURL_READY_AFTER=1, STUB_PYTHON_EXIT=3
            )
            self.assertIn("Cleaning up", out, "cleanup must still run on failure")
            self.assertEqual(rc, 3, "the processor's exact status must survive")


@unittest.skipIf(BASH is None, "bash not available")
class TestOllamaBatchExecution(unittest.TestCase):
    def test_happy_path_exits_zero(self):
        with Sandbox() as sb:
            rc, out, _ = sb.run(build_text("ollama", "batch"), STUB_CURL_READY_AFTER=1)
            self.assertEqual(rc, 0, out[-2000:])
            self.assertTrue(any(c.startswith("python3") for c in sb.calls()))

    def test_server_death_fails_fast(self):
        with Sandbox() as sb:
            rc, out, elapsed = sb.run(
                build_text("ollama", "batch"),
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
            sb.run(build_text("vllm", "batch"), STUB_CURL_READY_AFTER=1)
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
            sb.run(build_text("vllm", "batch"), STUB_CURL_READY_AFTER=1)
            # The script rm -rf's its tmpdir; that must be inside the sandbox.
            self.assertTrue(str(work).startswith(str(sb.root)))
            self.assertTrue(sb.root.exists(), "sandbox root itself must survive")

    def test_stub_path_precedes_system_path(self):
        with Sandbox() as sb:
            first = sb.env["PATH"].split(os.pathsep)[0]
            self.assertEqual(first, str(sb.root / "bin"))


if __name__ == "__main__":
    unittest.main()


@unittest.skipIf(BASH is None, "bash not available")
class TestMultiNodeLauncher(unittest.TestCase):
    """Execute the multi-node script; the launcher is bash that must actually run."""

    def _run(self, sb, nodes="2", **env):
        env.setdefault("SLURM_JOB_NUM_NODES", nodes)
        return sb.run(build_text("vllm", "batch", nodes=nodes), **env)

    def test_all_ranks_launch_and_the_barrier_passes(self):
        with Sandbox() as sb:
            rc, out, _ = self._run(sb, STUB_CURL_READY_AFTER=2)
            self.assertIn("LLMFLUX-STAGE-A: all 2 ranks launched", out)
            self.assertIn("LLMFLUX-TOPOLOGY:", out)
            self.assertEqual(rc, 0, out[-2500:])

    def test_rank_zero_serves_and_others_are_headless(self):
        with Sandbox() as sb:
            self._run(sb, STUB_CURL_READY_AFTER=1)
            calls = "\n".join(sb.calls())
            self.assertIn("--node-rank 0", calls)
            self.assertIn("--headless", calls, "ranks > 0 must be headless workers")
            self.assertIn("--nnodes 2", calls)

    def test_master_addr_is_the_fabric_ip_never_a_hostname(self):
        """FQDNs resolve to IPv6 link-local here; hostnames can round-robin."""
        with Sandbox() as sb:
            _, out, _ = self._run(sb, STUB_CURL_READY_AFTER=1, STUB_NODE_IP="172.28.86.64")
            topo = next(l for l in out.splitlines() if "LLMFLUX-TOPOLOGY:" in l)
            self.assertIn("master=172.28.86.64:", topo)
            self.assertNotIn("141.142.254", topo, "VLAN child must not be selected")

    def test_srun_failure_fails_fast(self):
        with Sandbox() as sb:
            rc, out, elapsed = self._run(sb, STUB_SRUN_MODE="die", STUB_CURL_READY_AFTER=10**9)
            self.assertNotEqual(rc, 0)
            self.assertIn("LLMFLUX-ERROR", out)
            self.assertLess(elapsed, 30, "must not spin the full readiness budget")

    def test_partial_rank_start_fails_fast(self):
        """1 of 2 ranks up must fail, not hang until walltime."""
        with Sandbox() as sb:
            rc, out, elapsed = self._run(
                sb, STUB_SRUN_MODE="partial", LLMFLUX_RANK_START_TIMEOUT=3,
                STUB_CURL_READY_AFTER=10**9,
            )
            self.assertNotEqual(rc, 0)
            self.assertIn("of 2 ranks started", out)
            self.assertIn("LLMFLUX-DIAG", out, "a failure must dump diagnostics")
            self.assertLess(elapsed, 40)

    def test_node_count_mismatch_is_caught(self):
        """--sbatch-arg can append a second --nodes; sbatch honours the last."""
        with Sandbox() as sb:
            rc, out, _ = self._run(sb, SLURM_JOB_NUM_NODES="1", STUB_CURL_READY_AFTER=10**9)
            self.assertNotEqual(rc, 0)
            self.assertIn("expected 2", out)

    def test_requeue_does_not_pass_the_barrier_on_stale_files(self):
        """A requeued job reuses SLURM_JOB_ID; stale rendezvous files must not count."""
        with Sandbox() as sb:
            self._run(sb, STUB_CURL_READY_AFTER=1)                      # attempt 1
            rc, out, _ = self._run(
                sb, STUB_SRUN_MODE="none", SLURM_RESTART_COUNT="1",
                LLMFLUX_RANK_START_TIMEOUT=3, STUB_CURL_READY_AFTER=10**9,
            )
            self.assertNotEqual(rc, 0, "stale files must not satisfy the barrier")
            self.assertIn("0 of 2 ranks started", out)
