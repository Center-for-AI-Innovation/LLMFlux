# Working on LLMFlux with an AI assistant

Guidance for anyone using Claude Code (or another AI coding assistant) on this
repo, and for the assistants themselves. Keep it in context when making changes.

## Tests are not optional

**Every code change must come with tests.** This is the single most important
rule in this file. When you add or modify behavior, add or update the tests that
cover it in the same change — do not defer it, and do not treat "the existing
tests still pass" as sufficient.

Concretely:

- **New function or branch** → add tests for the happy path *and* the failure /
  edge cases (empty input, malformed input, boundary values).
- **Bug fix** → add a regression test that fails before your fix and passes
  after. State that expectation in the PR/commit so a reviewer can verify it.
- **Security fix** → add tests for the specific bypass or exploit you closed,
  using the concrete malicious inputs, not just a generic "invalid" case. See
  `tests/slurm/test_connection.py` for the pattern (encoded-IP SSRF bypasses are
  each pinned by a test).
- **Changed public behavior** → update the affected tests and the docs under
  `docs/` in the same change.

If a change genuinely has no observable behavior to test (pure docs, comments,
formatting), say so explicitly in the PR description rather than silently
shipping without tests.

## For AI assistants specifically

- **Add tests by default.** Treat "write the code" as "write the code and its
  tests." Do not ask whether tests are wanted — assume yes and include them.
- **Run the suite before reporting done.** A change is not finished until
  `python -m pytest` passes locally. Report the actual result (pass count /
  failures), never assume.
- **Match the existing test style.** Tests use `unittest.TestCase` classes with
  `unittest.mock` (`patch`, `MagicMock`); mirror the conventions in the
  neighboring test file rather than introducing a new framework or style.
- **Verify the fix actually exercises the change** — drive the real code path
  (e.g. call `connect()` end-to-end), not just the helper in isolation.
- **Keep changes scoped.** Don't stage or commit unrelated working-tree files;
  commit only what the task touched.

## Running tests

```bash
# Install test dependencies (once)
pip install -e ".[test]"

# Run the full suite
python -m pytest

# Run one file / one test
python -m pytest tests/slurm/test_connection.py
python -m pytest tests/slurm/test_connection.py::TestValidateNode
```

Test layout mirrors the package: tests live under `tests/`, named `test_*.py`.
See `docs/TESTING.md` for coverage reports, parallel runs, and more detail.
