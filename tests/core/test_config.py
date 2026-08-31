"""Tests for Config, SlurmConfig, ValidationConfig, ModelConfig, and helpers."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml
from pydantic import ValidationError

from llmflux.core.config import (
    ConfigDirectoryError,
    Config,
    ModelConfig,
    ModelParameters,
    SlurmConfig,
    ValidationConfig,
    _parse_extra_sbatch_args,
    parse_gpu_memory,
)


class TestParseGpuMemory(unittest.TestCase):
    def test_valid_string(self):
        self.assertEqual(parse_gpu_memory("16GB"), 16)

    def test_large_value(self):
        self.assertEqual(parse_gpu_memory("80GB"), 80)

    def test_invalid_format_raises(self):
        with self.assertRaises(ValueError):
            parse_gpu_memory("16gb")

    def test_no_unit_raises(self):
        with self.assertRaises(ValueError):
            parse_gpu_memory("16")

    def test_float_raises(self):
        with self.assertRaises(ValueError):
            parse_gpu_memory("16.5GB")


class TestParseExtraSbatchArgs(unittest.TestCase):
    def test_returns_none_when_env_not_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_EXTRA_ARGS", None)
            self.assertIsNone(_parse_extra_sbatch_args())

    def test_json_format(self):
        val = json.dumps({"reservation": "myres", "qos": "high"})
        with patch.dict(os.environ, {"SLURM_EXTRA_ARGS": val}):
            result = _parse_extra_sbatch_args()
        self.assertEqual(result, {"reservation": "myres", "qos": "high"})

    def test_key_value_format(self):
        with patch.dict(os.environ, {"SLURM_EXTRA_ARGS": "reservation=myres,qos=high"}):
            result = _parse_extra_sbatch_args()
        self.assertEqual(result, {"reservation": "myres", "qos": "high"})

    def test_single_key_value(self):
        with patch.dict(os.environ, {"SLURM_EXTRA_ARGS": "reservation=myres"}):
            result = _parse_extra_sbatch_args()
        self.assertEqual(result, {"reservation": "myres"})

    def test_invalid_json_falls_back_to_kv(self):
        with patch.dict(os.environ, {"SLURM_EXTRA_ARGS": "{bad json}"}):
            result = _parse_extra_sbatch_args()
        self.assertIsNone(result)

    def test_empty_string_returns_none(self):
        with patch.dict(os.environ, {"SLURM_EXTRA_ARGS": ""}):
            result = _parse_extra_sbatch_args()
        self.assertIsNone(result)


class TestValidationConfig(unittest.TestCase):
    def _valid_kwargs(self):
        return dict(
            temperature_range=[0.0, 1.0],
            max_tokens_limit=4096,
            batch_size_range=[1, 32],
            concurrent_range=[1, 8],
        )

    def test_valid_config(self):
        cfg = ValidationConfig(**self._valid_kwargs())
        self.assertEqual(cfg.max_tokens_limit, 4096)

    def test_temperature_range_out_of_order_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["temperature_range"] = [1.0, 0.0]
        with self.assertRaises(ValidationError):
            ValidationConfig(**kwargs)

    def test_temperature_range_out_of_bounds_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["temperature_range"] = [0.0, 1.5]
        with self.assertRaises(ValidationError):
            ValidationConfig(**kwargs)

    def test_batch_size_range_out_of_order_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["batch_size_range"] = [32, 1]
        with self.assertRaises(ValidationError):
            ValidationConfig(**kwargs)

    def test_wrong_list_length_raises(self):
        kwargs = self._valid_kwargs()
        kwargs["temperature_range"] = [0.0]
        with self.assertRaises(ValidationError):
            ValidationConfig(**kwargs)


class TestConfigDirectoryResolution(unittest.TestCase):
    def test_explicit_data_dir_used(self):
        cfg = Config(data_dir="/tmp/mydata")
        self.assertEqual(cfg.data_dir, str(Path("/tmp/mydata").resolve()))

    def test_env_var_data_dir_used(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_DIR": "/tmp/envdata"}):
            cfg = Config()
        self.assertEqual(cfg.data_dir, str(Path("/tmp/envdata").resolve()))

    def test_default_data_dir_is_cwd_relative(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_DATA_DIR", None)
            cfg = Config()
        self.assertTrue(cfg.data_dir.endswith("data"))

    def test_explicit_logs_dir(self):
        cfg = Config(logs_dir="/tmp/mylogs")
        self.assertEqual(cfg.logs_dir, str(Path("/tmp/mylogs").resolve()))

    def test_code_param_beats_env_var(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_DIR": "/tmp/envdata"}):
            cfg = Config(data_dir="/tmp/explicit")
        self.assertEqual(cfg.data_dir, str(Path("/tmp/explicit").resolve()))


class TestConfigWorkspace(unittest.TestCase):
    def test_default_workspace_is_cwd(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_WORKSPACE", None)
            cfg = Config()
        self.assertEqual(cfg.workspace, Path.cwd())

    def test_explicit_workspace_used(self):
        cfg = Config(workspace="/tmp/myworkspace")
        self.assertEqual(cfg.workspace, Path("/tmp/myworkspace").resolve())

    def test_env_var_workspace_used(self):
        with patch.dict(os.environ, {"LLMFLUX_WORKSPACE": "/tmp/envworkspace"}):
            cfg = Config()
        self.assertEqual(cfg.workspace, Path("/tmp/envworkspace").resolve())

    def test_code_param_beats_env_var(self):
        with patch.dict(os.environ, {"LLMFLUX_WORKSPACE": "/tmp/envworkspace"}):
            cfg = Config(workspace="/tmp/explicit")
        self.assertEqual(cfg.workspace, Path("/tmp/explicit").resolve())

    def test_directories_derive_from_workspace(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in ("LLMFLUX_DATA_DIR", "LLMFLUX_MODELS_DIR", "LLMFLUX_LOGS_DIR", "LLMFLUX_CONTAINERS_DIR"):
                os.environ.pop(var, None)
            cfg = Config(workspace="/tmp/myworkspace")
        workspace = str(Path("/tmp/myworkspace").resolve())
        self.assertEqual(cfg.data_dir, f"{workspace}/data")
        self.assertEqual(cfg.models_dir, f"{workspace}/models")
        self.assertEqual(cfg.logs_dir, f"{workspace}/logs")
        self.assertEqual(cfg.containers_dir, f"{workspace}/containers")

    def test_explicit_dir_beats_workspace_derived_default(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_DIR": "/tmp/envdata"}):
            cfg = Config(workspace="/tmp/myworkspace")
        self.assertEqual(cfg.data_dir, str(Path("/tmp/envdata").resolve()))


class TestConfigInputOutputDirs(unittest.TestCase):
    def test_default_derive_from_data_dir(self):
        cfg = Config(data_dir="/tmp/mydata")
        data_dir = str(Path("/tmp/mydata").resolve())
        self.assertEqual(cfg.data_input_dir, f"{data_dir}/input")
        self.assertEqual(cfg.data_output_dir, f"{data_dir}/output")

    def test_explicit_separate_input_output(self):
        cfg = Config(data_input_dir="/tmp/projects/prompts", data_output_dir="/tmp/scratch/results")
        self.assertEqual(cfg.data_input_dir, str(Path("/tmp/projects/prompts").resolve()))
        self.assertEqual(cfg.data_output_dir, str(Path("/tmp/scratch/results").resolve()))

    def test_env_var_input_output(self):
        with patch.dict(os.environ, {
            "LLMFLUX_DATA_INPUT_DIR": "/tmp/env/inputs",
            "LLMFLUX_DATA_OUTPUT_DIR": "/tmp/env/outputs",
        }):
            cfg = Config()
        self.assertEqual(cfg.data_input_dir, str(Path("/tmp/env/inputs").resolve()))
        self.assertEqual(cfg.data_output_dir, str(Path("/tmp/env/outputs").resolve()))

    def test_code_param_beats_env_var(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_INPUT_DIR": "/tmp/env/inputs"}):
            cfg = Config(data_input_dir="/tmp/explicit/inputs")
        self.assertEqual(cfg.data_input_dir, str(Path("/tmp/explicit/inputs").resolve()))

    def test_legacy_unprefixed_env_vars_are_ignored(self):
        # DATA_INPUT_DIR/DATA_OUTPUT_DIR are no longer read by any config code;
        # only LLMFLUX_-prefixed vars and explicit params resolve directories
        with patch.dict(os.environ, {
            "DATA_INPUT_DIR": "/legacy/inputs",
            "DATA_OUTPUT_DIR": "/legacy/outputs",
        }):
            os.environ.pop("LLMFLUX_DATA_INPUT_DIR", None)
            os.environ.pop("LLMFLUX_DATA_OUTPUT_DIR", None)
            cfg = Config(data_dir="/tmp/mydata")
        data_dir = str(Path("/tmp/mydata").resolve())
        self.assertEqual(cfg.data_input_dir, f"{data_dir}/input")
        self.assertEqual(cfg.data_output_dir, f"{data_dir}/output")


class TestGetSlurmConfigMemory(unittest.TestCase):
    def test_memory_override_is_applied(self):
        cfg = Config()
        slurm = cfg.get_slurm_config({"memory": "64G"})
        self.assertEqual(slurm.memory, "64G")

    def test_mem_field_stays_removed(self):
        # get_slurm_config routes overrides through hasattr, so reintroducing
        # a duplicate 'mem' field would silently swallow memory overrides
        # while job scripts read 'memory' (#119)
        self.assertNotIn("mem", SlurmConfig.model_fields)

    def test_memory_defaults_from_slurm_mem_env(self):
        with patch.dict(os.environ, {"SLURM_MEM": "48G"}):
            self.assertEqual(SlurmConfig().memory, "48G")


class TestLoadModelConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _write_custom_yaml(self, data: dict, filename="models.yaml") -> str:
        path = self.tmp_dir / filename
        with open(path, "w") as f:
            yaml.dump(data, f)
        return str(path)

    def test_loads_from_bundled_models_yaml(self):
        cfg = Config()
        model = cfg.load_model_config("gemma-3-1b-it")
        self.assertIsNotNone(model)
        self.assertIsInstance(model, ModelConfig)

    def test_returns_none_for_missing_model_key(self):
        cfg = Config()
        result = cfg.load_model_config("nonexistent-model-xyz")
        self.assertIsNone(result)

    def test_custom_config_with_models_key(self):
        custom = {
            "models": {
                "my-model": {
                    "name": "my-model:latest",
                    "hf_name": "org/my-model",
                    "parameters": {"temperature": 0.5, "max_tokens": 1024,
                                   "top_p": 0.9, "top_k": 40},
                }
            }
        }
        path = self._write_custom_yaml(custom)
        cfg = Config()
        model = cfg.load_model_config("my-model", custom_config_path=path)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "my-model:latest")

    def test_custom_config_missing_key_returns_none(self):
        custom = {"models": {"other-model": {"name": "other:latest"}}}
        path = self._write_custom_yaml(custom)
        cfg = Config()
        result = cfg.load_model_config("my-model", custom_config_path=path)
        self.assertIsNone(result)

    def test_custom_config_flat_mapping(self):
        custom = {
            "my-model": {
                "name": "flat:latest",
                "hf_name": "org/flat",
                "parameters": {"temperature": 0.7, "max_tokens": 512,
                               "top_p": 0.9, "top_k": 40},
            }
        }
        path = self._write_custom_yaml(custom)
        cfg = Config()
        model = cfg.load_model_config("my-model", custom_config_path=path)
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "flat:latest")


class TestModelConfigGetModelNameForEngine(unittest.TestCase):
    def test_vllm_returns_hf_name(self):
        model = ModelConfig(name="mymodel:7b", hf_name="org/mymodel-7b", engine="vllm")
        self.assertEqual(model.get_model_name_for_engine(), "org/mymodel-7b")

    def test_vllm_without_hf_name_raises(self):
        model = ModelConfig(name="mymodel:7b", engine="vllm")
        with self.assertRaises(ValueError):
            model.get_model_name_for_engine()

    def test_ollama_returns_name(self):
        model = ModelConfig(name="mymodel:7b", engine="ollama")
        self.assertEqual(model.get_model_name_for_engine(), "mymodel:7b")




class _ReadOnlyDirMixin:
    """Helper for building directories the running user cannot write to.

    Skips rather than silently passing where the mode does not take effect —
    root ignores the write bit, and some networked filesystems apply ACLs that
    override it. A test that cannot create the condition it is testing must say
    so, not report success.
    """

    def make_dir(self, name, mode=0o755):
        path = Path(self.tmp_dir) / name
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(mode)
        self.addCleanup(path.chmod, 0o755)
        return path

    def make_unwritable_dir(self, name):
        path = self.make_dir(name, mode=0o555)
        if os.access(path, os.W_OK):
            self.skipTest(
                f"cannot make {path} unwritable (running as root, or the "
                f"filesystem overrides the mode) — the condition under test "
                f"cannot be created here"
            )
        return path

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp_dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        # Config reads every directory from the environment when the
        # corresponding argument is None, so a stray LLMFLUX_* in the caller's
        # environment would silently retarget whatever this test did not set.
        env_patch = patch.dict(os.environ, {})
        env_patch.start()
        self.addCleanup(env_patch.stop)
        for var in (
            "LLMFLUX_WORKSPACE", "LLMFLUX_DATA_DIR", "LLMFLUX_DATA_INPUT_DIR",
            "LLMFLUX_DATA_OUTPUT_DIR", "LLMFLUX_MODELS_DIR", "LLMFLUX_LOGS_DIR",
            "LLMFLUX_CONTAINERS_DIR",
        ):
            os.environ.pop(var, None)

    def base_kwargs(self, **overrides):
        """A fully-specified, writable Config so only the override is at issue."""
        kwargs = {
            "workspace": str(self.make_dir("ws")),
            "data_dir": str(self.make_dir("data")),
            "data_input_dir": str(self.make_dir("data/input")),
            "data_output_dir": str(self.make_dir("data/output")),
            "models_dir": str(self.make_dir("models")),
            "logs_dir": str(self.make_dir("logs")),
            "containers_dir": str(self.make_dir("containers")),
        }
        kwargs.update(overrides)
        return kwargs


class TestConfigDirectoryAccessRequirements(_ReadOnlyDirMixin, unittest.TestCase):
    """Only the directories LLMFlux writes into may require write access.

    Requiring it on all seven is what made LLMFlux undeployable as a module:
    a site stages the container image in a versioned, service-account-owned
    directory and points LLMFLUX_CONTAINERS_DIR at it, and `llmflux run` then
    died in the Config constructor before doing anything. The same check also
    blocked the read-only input directory that the input/output split exists to
    support.
    """

    #: Directories LLMFlux only ever reads. Each has a legitimate read-only
    #: deployment, so an unwritable one must construct fine.
    READ_ONLY_OK = ("containers_dir", "models_dir", "data_input_dir")

    #: Directories LLMFlux writes into. An unwritable one is a real
    #: misconfiguration and must still fail early, at construction, rather than
    #: part-way through a job.
    MUST_BE_WRITABLE = ("workspace", "data_dir", "data_output_dir", "logs_dir")

    def test_every_directory_is_classified(self):
        """Guards the two lists against a new directory being added untested."""
        cfg = Config(**self.base_kwargs())
        configured = {
            name for name in
            ("workspace", "data_dir", "data_input_dir", "data_output_dir",
             "models_dir", "logs_dir", "containers_dir")
            if getattr(cfg, name, None) is not None
        }
        self.assertEqual(
            configured,
            set(self.READ_ONLY_OK) | set(self.MUST_BE_WRITABLE),
            "a configured directory is in neither list, so its access "
            "requirement is untested",
        )

    def test_read_only_directories_are_accepted(self):
        for name in self.READ_ONLY_OK:
            with self.subTest(directory=name):
                ro = self.make_unwritable_dir(f"ro-{name}")
                cfg = Config(**self.base_kwargs(**{name: str(ro)}))
                self.assertEqual(
                    str(getattr(cfg, name)), str(ro),
                    "the directory was accepted but not the one we asked for",
                )

    def test_written_directories_still_reject_unwritable(self):
        for name in self.MUST_BE_WRITABLE:
            with self.subTest(directory=name):
                ro = self.make_unwritable_dir(f"rw-{name}")
                with self.assertRaises(OSError) as ctx:
                    Config(**self.base_kwargs(**{name: str(ro)}))
                self.assertIn(str(ro), str(ctx.exception))

    def test_read_only_directories_must_still_be_readable(self):
        """Relaxing the write requirement is not the same as no requirement.

        A containers_dir that cannot even be listed is unusable — the image
        cannot be read out of it — and must fail at construction rather than as
        an obscure error inside apptainer later.
        """
        for name in self.READ_ONLY_OK:
            with self.subTest(directory=name):
                unreadable = self.make_dir(f"nx-{name}", mode=0o000)
                if os.access(unreadable, os.R_OK | os.X_OK):
                    self.skipTest("cannot make a directory unreadable here")
                with self.assertRaises(OSError) as ctx:
                    Config(**self.base_kwargs(**{name: str(unreadable)}))
                self.assertIn(str(unreadable), str(ctx.exception))

    def test_missing_read_only_directory_is_still_created(self):
        """The relaxed directories keep workspace-relative defaults that
        LLMFlux is expected to create on first run, so mkdir must still happen.
        """
        for name in self.READ_ONLY_OK:
            with self.subTest(directory=name):
                target = Path(self.tmp_dir) / f"new-{name}" / "nested"
                self.assertFalse(target.exists())
                Config(**self.base_kwargs(**{name: str(target)}))
                self.assertTrue(
                    target.is_dir(),
                    "a not-yet-existing directory must still be created",
                )


class TestConfigDirectoryErrorReporting(_ReadOnlyDirMixin, unittest.TestCase):
    """An unusable directory must be reported as a configuration mistake."""

    def test_error_names_the_setting_and_its_env_var(self):
        ro = self.make_unwritable_dir("ro-logs")
        with self.assertRaises(OSError) as ctx:
            Config(**self.base_kwargs(logs_dir=str(ro)))
        message = str(ctx.exception)
        self.assertIn(str(ro), message)
        self.assertIn("logs_dir", message)
        self.assertIn(
            "LLMFLUX_LOGS_DIR", message,
            "the message must name the variable to change, not just the path",
        )

    def test_error_is_an_oserror(self):
        """ConfigDirectoryError narrows the type without breaking callers that
        already catch OSError."""
        self.assertTrue(issubclass(ConfigDirectoryError, OSError))
        ro = self.make_unwritable_dir("ro-data")
        with self.assertRaises(ConfigDirectoryError):
            Config(**self.base_kwargs(data_dir=str(ro)))
if __name__ == "__main__":
    unittest.main()