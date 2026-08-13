"""Shell-safety checks on the generated SLURM batch scripts.

Two classes of defect these catch, both of which ship green under every other
test in this suite because nothing else treats the generated script as *code*:

1. **Syntax errors inside heredoc bodies.** `bash -n` on the assembled script
   does not check heredoc contents — bash treats them as data. Any shell code
   emitted inside a heredoc (a per-rank launcher, a helper library) is therefore
   unchecked by a naive `bash -n`, which is precisely the code that runs where
   it is hardest to debug.

2. **`export VAR=$(cmd)` swallowing a failure.** `export` is a command with its
   own exit status, so the status of the substitution is discarded:

       $ false_cmd() { return 3; }
       $ export V=$(false_cmd);   echo $?   ->  0    # failure lost
       $ export V="$(false_cmd)"; echo $?   ->  0    # quoted form, same
       $ V=$(false_cmd); export V; echo $?  ->  3    # correct

   A `|| { handle; exit 1; }` attached to the `export` form is dead code. In a
   launcher this silently substitutes an empty value for something like a
   rendezvous port, producing an intermittent hang rather than a clean failure.

Both checks run against freshly generated scripts, not the golden files, so they
still apply if a golden is stale.
"""

import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from .helpers import CASES, build_text

BASH = shutil.which("bash")

# Heredoc bodies that are shell CODE and must be syntax-checked. Empty today;
# the multi-node launcher adds entries here.
SHELL_HEREDOC_DELIMS = set()

# Heredoc bodies that are DATA (JSON, mail text) and must not be shell-checked.
DATA_HEREDOC_DELIMS = {"EOF", "MAIL_EOF"}

# Matches the opening of a heredoc, capturing the delimiter without its quotes.
_HEREDOC_OPEN = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")

# `export`/`declare -x`/`readonly` assigned directly from a command
# substitution, quoted or not, `$(...)` or backticks.
_EXPORT_CMDSUB = re.compile(
    r"""^\s*(?:export|declare\s+-x|readonly)\s+     # the offending keyword
        [A-Za-z_][A-Za-z0-9_]*                      # NAME
        =\s*["']?                                   # = with optional quote
        (?:\$\(|`)                                  # command substitution
    """,
    re.VERBOSE,
)


def _bash_n(text, label):
    """Run `bash -n` over text. Returns (ok, message)."""
    with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as fh:
        fh.write(text)
        path = fh.name
    try:
        proc = subprocess.run(
            [BASH, "-n", path], stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        return proc.returncode == 0, f"{label}: {proc.stderr.decode().strip()}"
    finally:
        Path(path).unlink(missing_ok=True)


def _heredocs(text):
    """Yield (delimiter, body) for every heredoc in text.

    Deliberately simple: the generated scripts are machine-built and never nest
    heredocs or open two on one line. If that changes, this must be revisited
    rather than silently under-reporting.
    """
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        m = _HEREDOC_OPEN.search(lines[i])
        if m:
            delim = m.group(2)
            body, i = [], i + 1
            while i < len(lines) and lines[i].strip() != delim:
                body.append(lines[i])
                i += 1
            yield delim, "\n".join(body)
        i += 1


@unittest.skipIf(BASH is None, "bash not available")
class TestGeneratedScriptSyntax(unittest.TestCase):
    def test_outer_script_parses(self):
        for engine, mode in CASES:
            with self.subTest(engine=engine, mode=mode):
                ok, msg = _bash_n(build_text(engine, mode), f"{engine}-{mode}")
                self.assertTrue(ok, msg)

    def test_shell_heredoc_bodies_parse(self):
        """`bash -n` on the outer script does NOT cover heredoc bodies."""
        for engine, mode in CASES:
            for delim, body in _heredocs(build_text(engine, mode)):
                if delim not in SHELL_HEREDOC_DELIMS:
                    continue
                with self.subTest(engine=engine, mode=mode, heredoc=delim):
                    ok, msg = _bash_n(body, f"{engine}-{mode} heredoc {delim}")
                    self.assertTrue(ok, msg)


class TestHeredocClassification(unittest.TestCase):
    def test_every_heredoc_is_classified(self):
        """A new heredoc must be declared shell or data, deliberately.

        Without this, adding a heredoc full of shell code silently opts out of
        the syntax check above.
        """
        known = SHELL_HEREDOC_DELIMS | DATA_HEREDOC_DELIMS
        for engine, mode in CASES:
            for delim, _ in _heredocs(build_text(engine, mode)):
                with self.subTest(engine=engine, mode=mode, heredoc=delim):
                    self.assertIn(
                        delim,
                        known,
                        f"heredoc <<{delim} in {engine}-{mode} is unclassified; add it "
                        f"to SHELL_HEREDOC_DELIMS (shell code, gets bash -n) or "
                        f"DATA_HEREDOC_DELIMS (JSON/text, must not be shell-checked)",
                    )


@unittest.skipIf(BASH is None, "bash not available")
class TestHeredocMachinery(unittest.TestCase):
    """Self-tests for the heredoc extractor and checker.

    `SHELL_HEREDOC_DELIMS` is empty today, so `test_shell_heredoc_bodies_parse`
    currently checks nothing — it is scaffolding for the multi-node launcher,
    which emits its per-rank script inside a quoted heredoc. Scaffolding that
    has never been shown to work is not scaffolding. These prove the machinery
    catches what it claims to, using synthetic input.
    """

    SCRIPT = "\n".join(
        [
            "#!/bin/bash",
            "echo start",
            "cat > /tmp/rank.sh <<'RANK_EOF'",
            "#!/bin/bash",
            "if [ -n \"$X\" ]; then",       # deliberately unterminated below
            "echo hi",
            "RANK_EOF",
            "cat > /tmp/data.json <<EOF",
            '{"a": 1}',
            "EOF",
            "echo done",
        ]
    )

    def test_extracts_each_heredoc_with_its_body(self):
        found = dict(_heredocs(self.SCRIPT))
        self.assertEqual(set(found), {"RANK_EOF", "EOF"})
        self.assertIn("echo hi", found["RANK_EOF"])
        self.assertIn('{"a": 1}', found["EOF"])
        self.assertNotIn("echo done", found["RANK_EOF"])

    def test_quoted_and_unquoted_delimiters_both_match(self):
        for opener in ["<<'D'", '<<"D"', "<<D", "<<-D"]:
            with self.subTest(opener=opener):
                text = f"cat > f <<{opener[2:]}\nbody\nD\n"
                self.assertEqual([d for d, _ in _heredocs(text)], ["D"])

    def test_outer_bash_n_does_not_see_the_broken_body(self):
        """The whole point: the outer parse passes despite a broken body."""
        ok, _ = _bash_n(self.SCRIPT, "outer")
        self.assertTrue(ok, "outer script should parse — heredocs are data to bash")

    def test_checking_the_body_directly_does_catch_it(self):
        body = dict(_heredocs(self.SCRIPT))["RANK_EOF"]
        ok, msg = _bash_n(body, "RANK_EOF")
        self.assertFalse(ok, "unterminated `if` in a heredoc body must fail bash -n")
        self.assertIn("syntax error", msg.lower())


class TestNoExportCommandSubstitution(unittest.TestCase):
    def test_no_export_from_command_substitution(self):
        for engine, mode in CASES:
            script = build_text(engine, mode)
            for n, line in enumerate(script.splitlines(), start=1):
                with self.subTest(engine=engine, mode=mode, line=n):
                    self.assertIsNone(
                        _EXPORT_CMDSUB.match(line),
                        f"{engine}-{mode}:{n} exports directly from a command "
                        f"substitution, which discards its exit status:\n  {line}\n"
                        f"Use: NAME=$(cmd) || {{ handle; exit 1; }}; export NAME",
                    )

    def test_detector_catches_all_offending_forms(self):
        """The guard is only worth having if it matches what people write."""
        offenders = [
            "export V=$(cmd)",
            'export V="$(cmd)"',
            "export V='$(cmd)'",
            "    export LLMFLUX_MASTER_PORT=$(llmflux_find_free_port 8000)",
            "export V=`cmd`",
            "declare -x V=$(cmd)",
            "readonly V=$(cmd)",
        ]
        for line in offenders:
            with self.subTest(line=line):
                self.assertIsNotNone(_EXPORT_CMDSUB.match(line), f"missed: {line}")

    def test_detector_does_not_fire_on_safe_forms(self):
        safe = [
            "V=$(cmd)",                      # assignment, status preserved
            "V=$(cmd); export V",
            "export V=literal",
            'export APPTAINERENV_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}',
            "export PATH=$PATH:/opt/bin",    # parameter expansion, not substitution
            "# export V=$(cmd)",             # a comment is not code
        ]
        for line in safe:
            with self.subTest(line=line):
                self.assertIsNone(_EXPORT_CMDSUB.match(line), f"false positive: {line}")


if __name__ == "__main__":
    unittest.main()
