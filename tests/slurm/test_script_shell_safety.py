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

#: (engine, mode, nodes). Multi-node only exists for vllm.
SHAPES = [(e, m, "1") for e, m in CASES] + [
    ("vllm", "batch", "2"),
    ("vllm", "serve", "2"),
    ("vllm", "batch", "4"),
]

BASH = shutil.which("bash")

# Heredoc bodies that are shell CODE and must be syntax-checked. These are the
# multi-node helper library and per-rank launcher — the latter is the only code
# that runs on nodes 2..N, so a syntax error in it is invisible everywhere else.
SHELL_HEREDOC_DELIMS = {"LLMFLUX_LIB_EOF", "LLMFLUX_RANK_EOF"}

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
        for engine, mode, nodes in SHAPES:
            with self.subTest(engine=engine, mode=mode, nodes=nodes):
                ok, msg = _bash_n(
                    build_text(engine, mode, nodes=nodes), f"{engine}-{mode}-n{nodes}"
                )
                self.assertTrue(ok, msg)

    def test_shell_heredoc_bodies_parse(self):
        """`bash -n` on the outer script does NOT cover heredoc bodies."""
        for engine, mode, nodes in SHAPES:
            for delim, body in _heredocs(build_text(engine, mode, nodes=nodes)):
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
        for engine, mode, nodes in SHAPES:
            for delim, _ in _heredocs(build_text(engine, mode, nodes=nodes)):
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
        for engine, mode, nodes in SHAPES:
            script = build_text(engine, mode, nodes=nodes)
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


class TestMultiNodePortIsolation(unittest.TestCase):
    """VLLM_PORT must not reach the container on a multi-node rank.

    vLLM seeds every internal ZMQ port from VLLM_PORT and walks upward, probing
    with bind() on the local node only. Two nodes walk the same sequence
    independently, so ranks on different nodes select the same port for a shared
    endpoint and the engine dies with "Address already in use". Observed on
    job 2941612 before this was fixed.
    """

    def _rank_body(self, nodes="2"):
        script = build_text("vllm", "batch", nodes=nodes)
        return dict(_heredocs(script))["LLMFLUX_RANK_EOF"]

    def test_rank_script_unsets_the_container_port(self):
        self.assertIn("unset APPTAINERENV_VLLM_PORT", self._rank_body())

    def test_unset_precedes_the_container_exec(self):
        body = self._rank_body().splitlines()
        unset_at = next(i for i, l in enumerate(body) if "unset APPTAINERENV_VLLM_PORT" in l)
        exec_at = next(i for i, l in enumerate(body) if "apptainer exec" in l)
        self.assertLess(unset_at, exec_at, "unsetting after the exec has no effect")

    def test_rank_zero_still_pins_its_api_port_on_the_command_line(self):
        """Unsetting the env must not lose the API port the client dials."""
        self.assertIn('--port "$VLLM_PORT"', self._rank_body())


class TestServeModeMultiNode(unittest.TestCase):
    """serve mode has its own launcher tail and its own ways to go wrong."""

    def _rank_body(self, mode):
        return dict(_heredocs(build_text("vllm", mode, nodes="2")))["LLMFLUX_RANK_EOF"]

    def test_rank_zero_authenticates_the_endpoint(self):
        """Multi-node serve must not be weaker than single-node serve.

        Without --api-key the endpoint is unauthenticated while connection.json
        still hands the user a key — a security regression introduced by the
        launcher, not present on the path it replaces.
        """
        self.assertIn('--api-key "$LLMFLUX_API_KEY"', self._rank_body("serve"))

    def test_headless_workers_never_take_the_api_key(self):
        """Only rank 0 serves an API; a headless worker rejects --api-key."""
        body = self._rank_body("serve")
        # Split on the `else` LINE, not on the substring. Prose elsewhere in the
        # script — an error message containing the word "elsewhere", say — would
        # otherwise move the split and silently change what this asserts.
        lines = body.splitlines()
        idx = next(i for i, l in enumerate(lines) if l.strip() == "else")
        headless_branch = "\n".join(lines[idx:])
        self.assertIn("--headless", headless_branch)
        self.assertNotIn("--api-key", headless_branch)

    def test_batch_mode_has_no_api_key(self):
        self.assertNotIn("--api-key", self._rank_body("batch"))

    def test_trap_tears_down_the_whole_deployment(self):
        """pkill only reaches the batch host; workers live on other nodes."""
        script = build_text("vllm", "serve", nodes="2")
        trap = next(l for l in script.splitlines() if l.startswith("trap "))
        self.assertIn("$CONNECTION_FILE", trap, "the key-bearing file must still go")
        self.assertIn("VLLM_PID", trap, "must kill the srun step, not just local procs")


class TestPublishedEndpointParity(unittest.TestCase):
    """Every address a serve job publishes must come from one resolved value.

    Swept over both engines, and deliberately NOT part of
    `TestServeModeMultiNode`: nothing here is multi-node-specific, and living in
    a class whose other tests all read the vLLM per-rank launcher heredoc is
    what kept these checks vLLM-only. The Ollama builder is an independent copy
    of the same logic, and it shipped 2.0.0 with all three sites calling
    `$(hostname)` separately, while the release notes described the resolved
    behaviour unconditionally. That drift is what this class exists to catch.
    """

    #: (engine, nodes, expected right-hand side of the assignment). Ollama is
    #: single-node only — topology.resolve rejects it above one node at submit
    #: time — so it has no fabric-address branch to pin.
    SHAPES = [
        ("vllm", "1", '"$(hostname -s)"'),
        ("vllm", "2", '"$LLMFLUX_MASTER_ADDR"'),
        ("ollama", "1", '"$(hostname -s)"'),
    ]

    def test_every_published_address_comes_from_one_expression(self):
        """`hostname` returns the FQDN here, which resolves to IPv6 link-local
        only — a client dialling it gets nothing.

        A serve job publishes its endpoint three times: the connection file, the
        notification email's Endpoint line, and the email's OpenAI() example. All
        three must read the same variable, or one job advertises two addresses
        and the reader has no way to tell which one works.
        """
        for engine, nodes, expected in self.SHAPES:
            with self.subTest(engine=engine, nodes=nodes):
                script = build_text(engine, "serve", nodes=nodes)
                lines = script.splitlines()
                assign = next(
                    (l for l in lines if l.startswith("LLMFLUX_ENDPOINT_HOST=")),
                    None,
                )
                self.assertIsNotNone(
                    assign,
                    f"{engine}-serve-n{nodes} derives no single endpoint host, so "
                    f"each publication site resolves the address independently",
                )
                self.assertEqual(assign, f"LLMFLUX_ENDPOINT_HOST={expected}")

                published = [
                    l for l in lines
                    if '"node"' in l or "Endpoint: http" in l or "base_url=" in l
                ]
                self.assertEqual(len(published), 3, f"expected 3 published addresses, got {published}")
                for line in published:
                    self.assertIn("$LLMFLUX_ENDPOINT_HOST", line)
                    self.assertNotIn("$(hostname)", line, "bare FQDN is not dialable")

    def test_endpoint_host_is_assigned_before_it_is_published(self):
        """An expansion above its assignment is empty, not an error."""
        for engine, nodes, _expected in self.SHAPES:
            with self.subTest(engine=engine, nodes=nodes):
                lines = build_text(engine, "serve", nodes=nodes).splitlines()
                assign = next((i for i, l in enumerate(lines)
                               if l.startswith("LLMFLUX_ENDPOINT_HOST=")), None)
                first_use = next((i for i, l in enumerate(lines)
                                  if "$LLMFLUX_ENDPOINT_HOST" in l), None)
                self.assertIsNotNone(assign, "no endpoint host is derived at all")
                self.assertIsNotNone(first_use, "the derived host is never published")
                self.assertLess(assign, first_use)
