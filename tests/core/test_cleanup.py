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
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.workspace = self.root / "ws"
        self.workspace.mkdir()
        env = patch.dict(os.environ, {"LLMFLUX_WORKSPACE": str(self.workspace)})
        env.start()
        self.addCleanup(env.stop)

    def test_empties_directories_and_unlinks_files(self):
        logs = self.workspace / "logs"
        logs.mkdir()
        (logs / "19398443.out").write_text("stdout\n")
        (logs / "sub").mkdir()
        (logs / "sub" / "nested.txt").write_text("x")
        script = self.workspace / "job.sh"
        script.write_text("#!/bin/sh\n")

        deleted, skipped, errors = delete([logs, script, self.workspace / "missing"])

        self.assertEqual((skipped, errors), ([], []))
        self.assertEqual(deleted, [logs, script])
        self.assertTrue(logs.is_dir())  # the directory itself stays
        self.assertEqual(list(logs.iterdir()), [])
        self.assertFalse(script.exists())

    def test_symlinks_are_unlinked_not_followed(self):
        outside = self.root / "outside"
        outside.mkdir()
        (outside / "keep.txt").write_text("belongs to someone else")
        link = self.workspace / ".cache"
        link.symlink_to(outside, target_is_directory=True)
        dangling = self.workspace / "job.sh"
        dangling.symlink_to(self.root / "gone")

        deleted, skipped, errors = delete([link, dangling])

        self.assertEqual((deleted, skipped, errors), ([link, dangling], [], []))
        self.assertFalse(link.is_symlink())
        self.assertFalse(dangling.is_symlink())
        self.assertTrue((outside / "keep.txt").exists())

    def test_skips_the_workspace_and_anything_outside_it(self):
        (self.workspace / "keep.txt").write_text("x")

        deleted, skipped, errors = delete([self.workspace, self.root, self.workspace / ".."])

        self.assertEqual((deleted, errors), ([], []))
        self.assertEqual(len(skipped), 3)
        self.assertTrue((self.workspace / "keep.txt").exists())

    def test_remove_leaves_a_site_container_dir_alone(self):
        site = self.root / "sw" / "containers"
        site.mkdir(parents=True)
        (site / "llm_processor.sif").write_text("shared image")

        with patch.dict(os.environ, {"LLMFLUX_CONTAINERS_DIR": str(site)}), \
             patch("llmflux.core.cleanup.Path.home", return_value=self.root / "home"):
            deleted, skipped, errors = delete(remove_paths())

        self.assertNotIn(site, deleted)
        self.assertEqual(errors, [])
        self.assertTrue(any(str(site) in message for message in skipped))
        self.assertTrue((site / "llm_processor.sif").exists())


if __name__ == "__main__":
    unittest.main()
