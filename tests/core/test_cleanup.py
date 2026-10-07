"""Tests for llmflux.core.cleanup, which backs `llmflux clean` / `llmflux remove`."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llmflux.core.cleanup import clean_paths, delete, remove_paths


class TestPathLists(unittest.TestCase):
    def setUp(self):
        self.workspace = Path("/ws")
        env = patch.dict(os.environ, {"LLMFLUX_WORKSPACE": "/ws"})
        env.start()
        self.addCleanup(env.stop)
        for var in ("LLMFLUX_LOGS_DIR", "LLMFLUX_CONTAINERS_DIR", "LLMFLUX_MODELS_DIR"):
            os.environ.pop(var, None)

    def test_clean_takes_scratch_but_not_models(self):
        paths = clean_paths()
        self.assertIn(self.workspace / "logs", paths)
        self.assertIn(self.workspace / "tmp", paths)
        self.assertNotIn(self.workspace / "models", paths)
        self.assertNotIn(self.workspace / ".cache", paths)

    def test_remove_takes_models_but_never_the_user_data(self):
        paths = remove_paths()
        self.assertIn(self.workspace / "models", paths)
        self.assertIn(self.workspace / ".cache", paths)
        # data/ holds the user's own inputs and results; src/ holds the repo on
        # a source checkout. Neither command may reach them.
        self.assertNotIn(self.workspace / "data" / "input", paths)
        self.assertNotIn(self.workspace / "data" / "output", paths)
        self.assertNotIn(self.workspace / "src", paths)
        self.assertNotIn(self.workspace, paths)


class TestDelete(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def test_empties_directories_and_unlinks_files(self):
        logs = self.root / "logs"
        logs.mkdir()
        (logs / "19398443.out").write_text("stdout\n")
        (logs / "sub").mkdir()
        (logs / "sub" / "nested.txt").write_text("x")
        script = self.root / "job.sh"
        script.write_text("#!/bin/sh\n")

        deleted, errors = delete([logs, script, self.root / "missing"])

        self.assertEqual(errors, [])
        self.assertEqual(deleted, [logs, script])
        self.assertTrue(logs.is_dir())  # the directory itself stays
        self.assertEqual(list(logs.iterdir()), [])
        self.assertFalse(script.exists())

    def test_a_symlinked_directory_is_unlinked_not_emptied(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("belongs to someone else")
        link = self.root / ".cache"
        link.symlink_to(outside, target_is_directory=True)

        deleted, errors = delete([link])

        self.assertEqual(errors, [])
        self.assertEqual(deleted, [link])
        self.assertFalse(link.is_symlink())
        self.assertTrue((outside / "keep.txt").exists())


if __name__ == "__main__":
    unittest.main()
