#!/usr/bin/env bash
# Every quality gate, in order of how fast it fails. See qa/QA-PROCEDURES.md.
# Exit non-zero if any gate fails, so this can be a release check or a CI job.
set -uo pipefail
cd "$(dirname "$0")/.."

fail=0
gate() {                       # gate "<name>" <command...>
    local name="$1"; shift
    printf '\n=== %s ===\n' "$name"
    if "$@"; then
        printf '    PASS  %s\n' "$name"
    else
        printf '    FAIL  %s\n' "$name"
        fail=1
    fi
}

# Check the tooling before running anything, because the failure it produces
# otherwise is actively misleading. Without pytest-cov, `--cov` is not an
# unknown flag with a helpful message -- argparse prints "unrecognized
# arguments: --cov", which reads as a typo in this script. It then took gate
# 3 down with it (mutation_check passes --no-cov), whose failure text is
# "baseline (unmutated suite must be green)" -- a statement about the suite,
# which was green. Three gates red, one cause, and none of the three messages
# named it.
if ! python -c "import pytest_cov" 2>/dev/null; then
    printf 'MISSING TOOLING: pytest-cov is not installed.\n'
    printf 'Gates 1, 2 and 3 all need it and all fail with unrelated-looking errors without it.\n\n'
    printf '    pip install -r qa/requirements-dev.txt\n\n'
    printf 'Refusing to run rather than reporting three false failures.\n'
    exit 2
fi

gate "1. unit + integration + specs (with branch coverage floor)" \
    python -m pytest polymarket_bot/tests -q --cov

gate "2. audit reproducers (permanent regressions)" \
    env AUDIT_REPRO=1 python -m pytest polymarket_bot/tests/audit -q --no-cov

gate "3. mutation check (would the tests NOTICE a wrong guard?)" \
    python qa/mutation_check.py

gate "4. chaos drills (does the system REACT to failure?)" \
    python -m polymarket_bot.main --mode chaos

gate "5. lint — bug rules only, advisory" \
    ruff check polymarket_bot

printf '\n========================================\n'
if [ "$fail" -eq 0 ]; then
    printf 'ALL GATES PASSED\n'
else
    printf 'ONE OR MORE GATES FAILED\n'
fi
exit "$fail"
