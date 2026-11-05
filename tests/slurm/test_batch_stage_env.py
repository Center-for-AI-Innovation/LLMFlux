"""The batch stage must receive every tuning value `llmflux run` was given.

LLMFlux#144: SlurmRunner.run() exported the values as APPTAINERENV_* while the
generated batch stage, which runs on the host where that prefix is never
stripped, read the bare names. Each half looked right on its own, so a string
assertion on either the env or the script cannot catch it. These tests drive
both halves together: submit through SlurmRunner.run() with sbatch mocked,
let bash expand the stage's quoted `python -c` payload exactly as the job
would, then execute that payload against the captured sbatch environment with
BatchProcessor stubbed, and assert what it was constructed and run with.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from llmflux.core.config import Config, EngineConfig, SlurmConfig
from llmflux.core.config_manager import ConfigManager
from llmflux.slurm.runner import SlurmRunner

#: Every tuning flag, set to a value no default matches.
TUNING = dict(
    batch_size=16,
    save_frequency=7,
    max_retries=9,
    retry_delay=2.5,
    max_tokens=32,
    temperature=0.2,
    top_p=0.85,
    top_k=5,
)
EXPECTED_INIT = dict(batch_size=16, save_frequency=7, max_retries=9, retry_delay=2.5)
EXPECTED_RUN = dict(max_tokens=32, temperature=0.2, top_p=0.85, top_k=5)
#: The bare names, which a .env file or the submitting shell may set.
BARE_NAMES = [key.upper() for key in TUNING]


def _stage_payload(script_text):
    """Return the stage's Python source as bash hands it to the interpreter.

    The stage is emitted as `<interpreter> -c "<source>"`. Swapping the
    interpreter for printf and running that one command through bash applies
    bash's own double-quote expansion, so `$` or `\\` in the source would show
    up here exactly as it would in the job.
    """
    lines = script_text.splitlines()
    exe = re.escape(sys.executable)
    start = next(i for i, l in enumerate(lines) if re.fullmatch(rf"'?{exe}'? -c \"", l))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == '"')
    command = "printf '%s' \"" + "\n".join(lines[start + 1:end]) + '\n"'
    return subprocess.run(
        ["bash", "-c", command], check=True, capture_output=True, text=True
    ).stdout


class TestBatchStageReceivesTuningValues(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        for sub in ("data", "models", "logs", "containers"):
            (root / sub).mkdir()
        self.input = root / "data" / "in.jsonl"
        self.input.write_text('{"custom_id": "a", "body": {"messages": []}}\n')
        self.output = root / "data" / "out.json"
        self.root = root

    def _submit(self, engine, nodes=1, host_env=None, **kwargs):
        """Run SlurmRunner.run(); return (script text, env handed to sbatch)."""
        cfg = Config(
            data_dir=str(self.root / "data"),
            models_dir=str(self.root / "models"),
            logs_dir=str(self.root / "logs"),
            containers_dir=str(self.root / "containers"),
            slurm=SlurmConfig(
                partition="gpu", nodes=nodes, gpus_per_node=4,
                time="01:00:00", memory="16G", cpus_per_task=4, account="acct",
            ),
        )
        captured = {}

        def fake_sbatch(cmd, env=None, **_):
            captured["script"] = Path(cmd[1]).read_text()
            captured["env"] = dict(env)
            return MagicMock(stdout="Submitted batch job 4242")

        manager = MagicMock()
        manager.get_config.return_value = cfg
        # The real priority logic: it decides what reaches the env.
        manager.get_parameter.side_effect = ConfigManager.get_parameter
        with patch("llmflux.slurm.runner.ConfigManager", return_value=manager), \
             patch("llmflux.slurm.runner.JobRegistry"), \
             patch.dict(os.environ), \
             patch("llmflux.slurm.runner.subprocess.run", side_effect=fake_sbatch) as sbatch:
            # Whatever the developer's shell exports must not leak in.
            for name in BARE_NAMES:
                os.environ.pop(name, None)
            os.environ.update(host_env or {})
            self.sbatch = sbatch
            runner = SlurmRunner(config=cfg.slurm, engine_config=EngineConfig(engine=engine))
            runner.workspace = self.root
            job_id = runner.run(
                input_path=str(self.input), output_path=str(self.output),
                model="Llama-3.2-3B-Instruct", **kwargs,
            )
        self.assertEqual(job_id, "4242")
        return captured["script"], captured["env"]

    def _execute_stage(self, script, env):
        """Execute the stage against env; return (init kwargs, run kwargs)."""
        source = _stage_payload(script)
        self.assertIn("BatchProcessor(", source)  # the payload really was found
        processor_cls = MagicMock()
        with patch.dict(os.environ, env, clear=True), \
             patch("llmflux.processors.BatchProcessor", processor_cls), \
             patch("llmflux.core.config.Config"):
            exec(compile(source, "<batch stage>", "exec"), {"__name__": "__stage__"})
        processor_cls.assert_called_once()
        init = {k: v for k, v in processor_cls.call_args.kwargs.items() if k != "model_config"}
        run_kwargs = processor_cls.return_value.run.call_args.kwargs
        return init, run_kwargs

    def _assert_stage_gets(self, engine, nodes=1, host_env=None):
        script, env = self._submit(engine, nodes=nodes, host_env=host_env, **TUNING)
        init, run_kwargs = self._execute_stage(script, env)
        self.assertEqual(init, EXPECTED_INIT)
        self.assertEqual(run_kwargs, EXPECTED_RUN)
        # Typed, not the env's strings: max_tokens="32" would reach the engine
        # as a JSON string.
        for key, value in run_kwargs.items():
            self.assertIs(type(value), type(EXPECTED_RUN[key]), key)

    def test_vllm_stage_receives_every_tuning_value(self):
        self._assert_stage_gets("vllm")

    def test_vllm_multinode_stage_receives_every_tuning_value(self):
        self._assert_stage_gets("vllm", nodes=2)

    def test_ollama_stage_receives_every_tuning_value(self):
        self._assert_stage_gets("ollama")

    def test_flags_beat_bare_names_exported_in_submitting_shell(self):
        """The old workaround (`export BATCH_SIZE=8`) must not override a flag.

        sbatch inherits the submitting shell's environment, so a stage that read
        the bare names let a stale export beat the flag the user just typed.
        """
        stale = {"BATCH_SIZE": "8", "SAVE_FREQUENCY": "1", "MAX_RETRIES": "0",
                 "RETRY_DELAY": "0", "MAX_TOKENS": "999", "TEMPERATURE": "1.0",
                 "TOP_P": "1.0", "TOP_K": "1"}
        for engine in ("vllm", "ollama"):
            with self.subTest(engine=engine):
                self._assert_stage_gets(engine, host_env=stale)

    def test_environment_applies_when_flag_is_not_given(self):
        """docs/CONFIGURATION.md: code value, then environment (.env), then
        default. `BATCH_SIZE=8` in .env must still work once the stage stops
        reading the bare name."""
        script, env = self._submit(
            "vllm", host_env={"BATCH_SIZE": "8", "MAX_TOKENS": "64", "TOP_K": "3"},
        )
        init, run_kwargs = self._execute_stage(script, env)
        self.assertEqual(init, dict(batch_size=8, save_frequency=50, max_retries=3, retry_delay=1.0))
        self.assertEqual(run_kwargs, dict(max_tokens=64, top_k=3))

    def test_malformed_value_fails_before_submitting(self):
        for host_env, kwargs in (({"BATCH_SIZE": "eight"}, {}), ({}, {"temperature": "warm"})):
            with self.subTest(host_env=host_env, kwargs=kwargs):
                with self.assertRaises(ValueError) as ctx:
                    self._submit("vllm", host_env=host_env, **kwargs)
                self.assertRegex(str(ctx.exception), "batch_size|temperature")
                self.sbatch.assert_not_called()

    def test_unset_sampling_flags_send_nothing(self):
        """Without the flags, run() gets no sampling defaults at all, so the
        request body and then the model config decide, as before."""
        script, env = self._submit("vllm")
        init, run_kwargs = self._execute_stage(script, env)
        self.assertEqual(init, dict(batch_size=4, save_frequency=50, max_retries=3, retry_delay=1.0))
        self.assertEqual(run_kwargs, {})


if __name__ == "__main__":
    unittest.main()
