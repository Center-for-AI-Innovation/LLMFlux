"""Tests for JobRegistry CRUD operations."""

import json
import tempfile
import unittest
from pathlib import Path

from llmflux.core.registry import JobRegistry


class TestJobRegistryInit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry_file = str(Path(self.tmp.name) / "jobs.json")

    def tearDown(self):
        self.tmp.cleanup()

    def test_creates_file_on_init_if_missing(self):
        registry = JobRegistry(registry_file=self.registry_file)
        self.assertTrue(Path(self.registry_file).exists())

    def test_creates_parent_dirs_if_missing(self):
        nested = str(Path(self.tmp.name) / "a" / "b" / "jobs.json")
        JobRegistry(registry_file=nested)
        self.assertTrue(Path(nested).exists())

    def test_empty_registry_has_no_jobs(self):
        registry = JobRegistry(registry_file=self.registry_file)
        self.assertEqual(registry.get_all_jobs(), {})

    def test_loads_existing_data_on_init(self):
        path = Path(self.registry_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"123": {"model": "llama3"}}), encoding="utf-8")
        registry = JobRegistry(registry_file=self.registry_file)
        self.assertEqual(registry.get_job("123"), {"model": "llama3"})

    def test_non_dict_json_treated_as_empty(self):
        path = Path(self.registry_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        registry = JobRegistry(registry_file=self.registry_file)
        self.assertEqual(registry.get_all_jobs(), {})


class TestCreateJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry_file = str(Path(self.tmp.name) / "jobs.json")
        self.registry = JobRegistry(registry_file=self.registry_file)

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_stores_metadata(self):
        self.registry.create_job("42", {"model": "llama3", "status": "submitted"})
        self.assertEqual(self.registry.get_job("42"), {"model": "llama3", "status": "submitted"})

    def test_create_persists_to_disk(self):
        self.registry.create_job("99", {"model": "mistral"})
        reloaded = JobRegistry(registry_file=self.registry_file)
        self.assertEqual(reloaded.get_job("99"), {"model": "mistral"})

    def test_duplicate_id_raises_value_error(self):
        self.registry.create_job("10", {"model": "llama3"})
        with self.assertRaises(ValueError):
            self.registry.create_job("10", {"model": "other"})

    def test_integer_id_normalized_to_string(self):
        self.registry.create_job(42, {"model": "llama3"})
        self.assertIsNotNone(self.registry.get_job("42"))

    def test_multiple_jobs_stored_independently(self):
        self.registry.create_job("1", {"model": "a"})
        self.registry.create_job("2", {"model": "b"})
        self.assertEqual(self.registry.get_job("1")["model"], "a")
        self.assertEqual(self.registry.get_job("2")["model"], "b")


class TestGetJob(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = JobRegistry(registry_file=str(Path(self.tmp.name) / "jobs.json"))
        self.registry.create_job("100", {"model": "llama3"})

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_metadata_for_known_id(self):
        result = self.registry.get_job("100")
        self.assertEqual(result["model"], "llama3")

    def test_returns_none_for_unknown_id(self):
        self.assertIsNone(self.registry.get_job("999"))

    def test_integer_id_lookup_works(self):
        result = self.registry.get_job(100)
        self.assertIsNotNone(result)


class TestGetAllJobs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.registry = JobRegistry(registry_file=str(Path(self.tmp.name) / "jobs.json"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_returns_all_jobs(self):
        self.registry.create_job("1", {"model": "a"})
        self.registry.create_job("2", {"model": "b"})
        all_jobs = self.registry.get_all_jobs()
        self.assertIn("1", all_jobs)
        self.assertIn("2", all_jobs)

    def test_get_all_job_ids(self):
        self.registry.create_job("5", {"model": "x"})
        self.registry.create_job("6", {"model": "y"})
        ids = self.registry.get_all_job_ids()
        self.assertIn("5", ids)
        self.assertIn("6", ids)

    def test_empty_registry_returns_empty_list(self):
        self.assertEqual(self.registry.get_all_job_ids(), [])


if __name__ == "__main__":
    unittest.main()