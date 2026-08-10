#!/usr/bin/env bash
# Stop hook (warn-only): when a turn leaves uncommitted changes under src/ or
# tests/, run the test suite and surface the result to the user, plus a nudge
# if source changed without any accompanying test change.
#
# This NEVER blocks — it only prints a systemMessage. It exists to reinforce
# the rule in AGENTS.md: every code change ships with tests, and the suite must
# pass before the work is done. See `/hooks` to review or disable it.
set -uo pipefail

emit() {  # print a non-blocking systemMessage as JSON on stdout
  MSG="$1" python3 -c 'import json, os; print(json.dumps({"systemMessage": os.environ["MSG"]}))'
}

# No-op outside a git work tree.
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0

# Only act when this turn left uncommitted changes under src/ or tests/.
[ -n "$(git status --porcelain -- src tests 2>/dev/null)" ] || exit 0

src_changed="$(git status --porcelain -- src 2>/dev/null)"
tests_changed="$(git status --porcelain -- tests 2>/dev/null)"

note=""
if [ -n "$src_changed" ] && [ -z "$tests_changed" ]; then
  note="⚠ src/ changed but no tests/ files were touched — add or update tests for this change (see AGENTS.md)."$'\n\n'
fi

if out="$(python3 -m pytest -q 2>&1)"; then
  emit "${note}✓ Tests: $(printf '%s\n' "$out" | tail -n 1)"
else
  emit "${note}✗ Tests FAILED — fix before finishing (see AGENTS.md):"$'\n'"$(printf '%s\n' "$out" | tail -n 12)"
fi
exit 0
