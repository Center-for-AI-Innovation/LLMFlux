"""Guards on .env.example.

`core/client.py` calls `load_dotenv(override=True)` at import time, and
python-dotenv parses a bare `KEY=` line as `''` rather than `None`. So a blank
entry in `.env.example` does not mean "unset" once a user follows the
`cp .env.example .env` instruction in docs/README.md — it erases whatever they
exported in their shell, and every consumer that resolves the value with `or`
silently falls back to its default.
"""

import os
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import load_dotenv

ENV_EXAMPLE = Path(__file__).resolve().parents[1] / ".env.example"

# Blank entries that predate the check, grandfathered so that new blanks fail the
# test rather than joining them silently. Now empty: every entry was commented
# out for 2.0.0, so the guard below is fully enforcing rather than frozen around
# nine known offenders. Removing a name from this list is always safe; adding
# one should be a deliberate decision, not a way to make this test pass.
KNOWN_BLANK_ENTRIES = set()


def _assignments():
    """Yield (name, raw_value) for every uncommented assignment in .env.example."""
    for line in ENV_EXAMPLE.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, _, value = stripped.partition("=")
        yield name.strip(), value.split("#")[0].strip()


class TestEnvExample(unittest.TestCase):
    def test_file_exists(self):
        self.assertTrue(ENV_EXAMPLE.is_file(), f"missing {ENV_EXAMPLE}")

    def test_api_key_is_not_assigned_a_blank_value(self):
        blank = [name for name, value in _assignments() if not value]
        self.assertNotIn(
            "LLMFLUX_API_KEY",
            blank,
            "`LLMFLUX_API_KEY=` in .env.example erases an exported key once copied "
            "to .env, because client.py loads it with override=True. Comment the "
            "line out instead.",
        )

    def test_no_new_blank_assignments(self):
        blank = {name for name, value in _assignments() if not value}
        self.assertEqual(
            blank - KNOWN_BLANK_ENTRIES,
            set(),
            "New blank entry in .env.example. A bare `KEY=` overwrites an exported "
            "value with '' under load_dotenv(override=True). Comment the line out, "
            "or give it a real default.",
        )

    def test_copied_env_file_does_not_erase_an_exported_api_key(self):
        """The path docs/README.md tells users to take: cp .env.example .env, export the key."""
        with patch.dict(os.environ, {"LLMFLUX_API_KEY": "llmflux-exported"}):
            load_dotenv(dotenv_path=ENV_EXAMPLE, override=True)
            self.assertEqual(os.environ["LLMFLUX_API_KEY"], "llmflux-exported")


if __name__ == "__main__":
    unittest.main()
