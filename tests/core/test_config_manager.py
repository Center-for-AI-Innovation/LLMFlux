"""Tests for ConfigManager singleton and parameter priority logic."""

import os
import unittest

import llmflux.core.config_manager as cm_module
from llmflux.core.config_manager import ConfigManager


class TestConfigManagerSingleton(unittest.TestCase):
    def tearDown(self):
        # Reset singleton after every test so state doesn't leak
        cm_module._config_instance = None

    def test_get_config_returns_instance(self):
        config = ConfigManager.get_config()
        self.assertIsNotNone(config)

    def test_get_config_same_instance_on_repeat_calls(self):
        c1 = ConfigManager.get_config()
        c2 = ConfigManager.get_config()
        self.assertIs(c1, c2)

    def test_reset_config_replaces_instance(self):
        c1 = ConfigManager.get_config()
        c2 = ConfigManager.reset_config()
        self.assertIsNot(c1, c2)
        self.assertIs(ConfigManager.get_config(), c2)

    def test_reset_config_with_custom_dirs(self):
        config = ConfigManager.reset_config(data_dir="/tmp/mydata", logs_dir="/tmp/mylogs")
        self.assertEqual(config.data_dir, "/tmp/mydata")
        self.assertEqual(config.logs_dir, "/tmp/mylogs")


class TestGetParameter(unittest.TestCase):
    def test_code_value_highest_priority(self):
        result = ConfigManager.get_parameter(
            "param",
            code_value="explicit",
            obj=None,
            env_var=None,
            default="fallback",
        )
        self.assertEqual(result, "explicit")

    def test_obj_attribute_over_env_and_default(self):
        class Obj:
            param = "from_obj"

        result = ConfigManager.get_parameter(
            "param",
            code_value=None,
            obj=Obj(),
            env_var="SOME_ENV_VAR_THAT_DOESNT_EXIST",
            default="fallback",
        )
        self.assertEqual(result, "from_obj")

    def test_env_var_over_default(self):
        with unittest.mock.patch.dict(os.environ, {"MY_TEST_VAR": "from_env"}):
            result = ConfigManager.get_parameter(
                "missing_attr",
                code_value=None,
                obj=None,
                env_var="MY_TEST_VAR",
                default="fallback",
            )
        self.assertEqual(result, "from_env")

    def test_default_when_nothing_else_set(self):
        result = ConfigManager.get_parameter(
            "nonexistent",
            code_value=None,
            obj=None,
            env_var=None,
            default="the_default",
        )
        self.assertEqual(result, "the_default")

    def test_none_code_value_falls_through(self):
        result = ConfigManager.get_parameter(
            "x",
            code_value=None,
            obj=None,
            env_var=None,
            default="default_val",
        )
        self.assertEqual(result, "default_val")

    def test_nested_attribute_resolution(self):
        class Inner:
            name = "inner_name"

        class Outer:
            inner = Inner()

        result = ConfigManager.get_parameter(
            "inner.name",
            code_value=None,
            obj=Outer(),
            env_var=None,
            default="fallback",
        )
        self.assertEqual(result, "inner_name")


import unittest.mock


class TestUpdateConfig(unittest.TestCase):
    def tearDown(self):
        cm_module._config_instance = None

    def test_update_data_dir(self):
        ConfigManager.get_config()
        updated = ConfigManager.update_config(data_dir="/tmp/updated_data")
        self.assertEqual(updated.data_dir, "/tmp/updated_data")

    def test_update_logs_dir(self):
        ConfigManager.get_config()
        updated = ConfigManager.update_config(logs_dir="/tmp/updated_logs")
        self.assertEqual(updated.logs_dir, "/tmp/updated_logs")

    def test_update_returns_same_singleton(self):
        original = ConfigManager.get_config()
        updated = ConfigManager.update_config(data_dir="/tmp/x")
        self.assertIs(original, updated)

    def test_update_without_args_is_noop(self):
        config = ConfigManager.get_config()
        original_data_dir = config.data_dir
        ConfigManager.update_config()
        self.assertEqual(ConfigManager.get_config().data_dir, original_data_dir)

    def test_update_refreshes_derived_paths(self):
        ConfigManager.get_config()
        updated = ConfigManager.update_config(data_dir="/tmp/newdata")
        self.assertIn("input", str(updated.default_paths.get("DATA_INPUT_DIR", "")))
