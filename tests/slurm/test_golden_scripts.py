"""Golden (characterization) tests for the generated SLURM batch scripts.

These pin the *exact* text of every script the engine builders emit. They are
not correctness tests: they exist so that any change to script generation shows
up as a reviewable line diff rather than being invisible until it fails on a
compute node.

The generated script is the only artifact that actually runs on the cluster, and
most of it is bash assembled from Python string lists — a form that no other
test in this suite reads end to end.

Regenerate after an intentional change, then read the diff before committing:

    LLMFLUX_UPDATE_GOLDEN=1 python -m pytest tests/slurm/test_golden_scripts.py
    git diff tests/slurm/golden/

A golden diff that you cannot explain line by line is a bug, not a refresh.
"""

import os
import unittest
from pathlib import Path

from .helpers import GOLDEN_CASES, build_text

GOLDEN_DIR = Path(__file__).parent / "golden"
UPDATE = os.environ.get("LLMFLUX_UPDATE_GOLDEN") == "1"



class TestGoldenScripts(unittest.TestCase):
    """One test per (engine, mode); each pins a full generated script."""

    def _check(self, engine, mode, nodes="1"):
        actual = build_text(engine, mode, nodes=nodes)
        path = GOLDEN_DIR / f"{engine}-{mode}-n{nodes}.sh"

        if UPDATE:
            GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(actual)
            self.skipTest(f"regenerated {path.name}")

        self.assertTrue(
            path.exists(),
            f"missing golden {path}; regenerate with "
            f"LLMFLUX_UPDATE_GOLDEN=1 python -m pytest {Path(__file__).name}",
        )
        expected = path.read_text()
        if expected != actual:
            # unittest's diff on a 250-line string is unreadable; point at the
            # first differing line, which is what a reviewer actually needs.
            exp_lines, act_lines = expected.splitlines(), actual.splitlines()
            for i, (e, a) in enumerate(zip(exp_lines, act_lines), start=1):
                if e != a:
                    self.fail(
                        f"{path.name} differs at line {i}\n"
                        f"  golden:    {e!r}\n"
                        f"  generated: {a!r}\n"
                        f"If intentional: LLMFLUX_UPDATE_GOLDEN=1 and review the diff."
                    )
            self.fail(
                f"{path.name} differs in length: golden {len(exp_lines)} lines, "
                f"generated {len(act_lines)} lines"
            )

    def test_vllm_batch(self):
        self._check("vllm", "batch")

    def test_vllm_serve(self):
        self._check("vllm", "serve")

    def test_ollama_batch(self):
        self._check("ollama", "batch")

    def test_ollama_serve(self):
        self._check("ollama", "serve")

    def test_vllm_batch_multinode(self):
        self._check("vllm", "batch", nodes="2")

    def test_vllm_serve_multinode(self):
        self._check("vllm", "serve", nodes="2")


class TestGoldenCoverage(unittest.TestCase):
    """Guards on the goldens themselves."""

    def test_every_case_has_a_golden(self):
        if UPDATE:
            self.skipTest("regenerating")
        missing = [
            f"{e}-{m}-n{n}.sh"
            for e, m, n in GOLDEN_CASES
            if not (GOLDEN_DIR / f"{e}-{m}-n{n}.sh").exists()
        ]
        self.assertEqual(missing, [], f"golden files missing: {missing}")

    def test_no_orphan_goldens(self):
        """A golden with no test is a file nobody regenerates."""
        if UPDATE or not GOLDEN_DIR.exists():
            self.skipTest("regenerating or no golden dir")
        expected = {f"{e}-{m}-n{n}.sh" for e, m, n in GOLDEN_CASES}
        actual = {p.name for p in GOLDEN_DIR.glob("*.sh")}
        self.assertEqual(
            actual - expected, set(), "orphan golden files with no owning test"
        )


if __name__ == "__main__":
    unittest.main()
