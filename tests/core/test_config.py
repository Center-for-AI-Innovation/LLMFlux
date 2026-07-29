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
        self.assertEqual(cfg.data_dir, "/tmp/mydata")

    def test_env_var_data_dir_used(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_DIR": "/tmp/envdata"}):
            cfg = Config()
        self.assertEqual(cfg.data_dir, "/tmp/envdata")

    def test_default_data_dir_is_cwd_relative(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_DATA_DIR", None)
            cfg = Config()
        self.assertTrue(cfg.data_dir.endswith("data"))

    def test_explicit_logs_dir(self):
        cfg = Config(logs_dir="/tmp/mylogs")
        self.assertEqual(cfg.logs_dir, "/tmp/mylogs")

    def test_code_param_beats_env_var(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_DIR": "/tmp/envdata"}):
            cfg = Config(data_dir="/tmp/explicit")
        self.assertEqual(cfg.data_dir, "/tmp/explicit")


class TestConfigWorkspace(unittest.TestCase):
    def test_default_workspace_is_cwd(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LLMFLUX_WORKSPACE", None)
            cfg = Config()
        self.assertEqual(cfg.workspace, Path.cwd())

    def test_explicit_workspace_used(self):
        cfg = Config(workspace="/tmp/myworkspace")
        self.assertEqual(cfg.workspace, Path("/tmp/myworkspace"))

    def test_env_var_workspace_used(self):
        with patch.dict(os.environ, {"LLMFLUX_WORKSPACE": "/tmp/envworkspace"}):
            cfg = Config()
        self.assertEqual(cfg.workspace, Path("/tmp/envworkspace"))

    def test_code_param_beats_env_var(self):
        with patch.dict(os.environ, {"LLMFLUX_WORKSPACE": "/tmp/envworkspace"}):
            cfg = Config(workspace="/tmp/explicit")
        self.assertEqual(cfg.workspace, Path("/tmp/explicit"))

    def test_directories_derive_from_workspace(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in ("LLMFLUX_DATA_DIR", "LLMFLUX_MODELS_DIR", "LLMFLUX_LOGS_DIR", "LLMFLUX_CONTAINERS_DIR"):
                os.environ.pop(var, None)
            cfg = Config(workspace="/tmp/myworkspace")
        self.assertEqual(cfg.data_dir, "/tmp/myworkspace/data")
        self.assertEqual(cfg.models_dir, "/tmp/myworkspace/models")
        self.assertEqual(cfg.logs_dir, "/tmp/myworkspace/logs")
        self.assertEqual(cfg.containers_dir, "/tmp/myworkspace/containers")
        self.assertEqual(cfg.default_paths["APPTAINER_TMPDIR"], Path("/tmp/myworkspace/tmp"))

    def test_explicit_dir_beats_workspace_derived_default(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_DIR": "/tmp/envdata"}):
            cfg = Config(workspace="/tmp/myworkspace")
        self.assertEqual(cfg.data_dir, "/tmp/envdata")


class TestConfigInputOutputDirs(unittest.TestCase):
    def test_default_derive_from_data_dir(self):
        cfg = Config(data_dir="/tmp/mydata")
        self.assertEqual(cfg.data_input_dir, "/tmp/mydata/input")
        self.assertEqual(cfg.data_output_dir, "/tmp/mydata/output")
        self.assertEqual(cfg.default_paths["DATA_INPUT_DIR"], Path("/tmp/mydata/input"))
        self.assertEqual(cfg.default_paths["DATA_OUTPUT_DIR"], Path("/tmp/mydata/output"))

    def test_explicit_separate_input_output(self):
        cfg = Config(data_input_dir="/projects/prompts", data_output_dir="/scratch/results")
        self.assertEqual(cfg.data_input_dir, "/projects/prompts")
        self.assertEqual(cfg.data_output_dir, "/scratch/results")
        self.assertEqual(cfg.default_paths["DATA_INPUT_DIR"], Path("/projects/prompts"))
        self.assertEqual(cfg.default_paths["DATA_OUTPUT_DIR"], Path("/scratch/results"))

    def test_env_var_input_output(self):
        with patch.dict(os.environ, {
            "LLMFLUX_DATA_INPUT_DIR": "/env/inputs",
            "LLMFLUX_DATA_OUTPUT_DIR": "/env/outputs",
        }):
            cfg = Config()
        self.assertEqual(cfg.data_input_dir, "/env/inputs")
        self.assertEqual(cfg.data_output_dir, "/env/outputs")

    def test_code_param_beats_env_var(self):
        with patch.dict(os.environ, {"LLMFLUX_DATA_INPUT_DIR": "/env/inputs"}):
            cfg = Config(data_input_dir="/explicit/inputs")
        self.assertEqual(cfg.data_input_dir, "/explicit/inputs")

    def test_get_path_reflects_explicit_dirs(self):
        cfg = Config(data_input_dir="/projects/prompts", data_output_dir="/scratch/results")
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATA_INPUT_DIR", None)
            os.environ.pop("DATA_OUTPUT_DIR", None)
            self.assertEqual(cfg.get_path("DATA_INPUT_DIR"), Path("/projects/prompts"))
            self.assertEqual(cfg.get_path("DATA_OUTPUT_DIR"), Path("/scratch/results"))


class TestConfigGetPath(unittest.TestCase):
    def setUp(self):
        self.cfg = Config(data_dir="/tmp/data", logs_dir="/tmp/logs")

    def test_code_path_highest_priority(self):
        result = self.cfg.get_path("DATA_INPUT_DIR", code_path="/override/path")
        self.assertEqual(result, Path("/override/path"))

    def test_env_var_used_when_no_code_path(self):
        with patch.dict(os.environ, {"DATA_INPUT_DIR": "/env/input"}):
            result = self.cfg.get_path("DATA_INPUT_DIR")
        self.assertEqual(result, Path("/env/input"))

    def test_default_path_used_as_fallback(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DATA_INPUT_DIR", None)
            result = self.cfg.get_path("DATA_INPUT_DIR")
        self.assertIn("input", str(result))

    def test_unknown_path_falls_back_to_workspace(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("UNKNOWN_KEY", None)
            result = self.cfg.get_path("UNKNOWN_KEY")
        self.assertIn("unknown_key", str(result))


class TestConfigGetSetting(unittest.TestCase):
    def setUp(self):
        self.cfg = Config()

    def test_code_value_beats_all(self):
        result = self.cfg.get_setting("SLURM_PARTITION", code_value="my-partition")
        self.assertEqual(result, "my-partition")

    def test_env_var_beats_default(self):
        with patch.dict(os.environ, {"SLURM_PARTITION": "env-partition"}):
            result = self.cfg.get_setting("SLURM_PARTITION")
        self.assertEqual(result, "env-partition")

    def test_default_returned_when_nothing_set(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SLURM_PARTITION", None)
            result = self.cfg.get_setting("SLURM_PARTITION")
        self.assertIsNotNone(result)

    def test_none_returned_for_unknown_setting(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TOTALLY_UNKNOWN", None)
            result = self.cfg.get_setting("TOTALLY_UNKNOWN")
        self.assertIsNone(result)


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


if __name__ == "__main__":
    unittest.main()