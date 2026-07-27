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
