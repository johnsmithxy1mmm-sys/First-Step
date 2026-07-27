# QA Procedures

How this codebase is verified, and — more importantly — *why each layer exists*.
Every layer below was added because something got through the ones above it.

## The gates, in order

| # | Gate | Command | Blocks a release? |
|---|---|---|---|
| 1 | Unit + integration suite | `pytest polymarket_bot/tests -q` | yes |
| 2 | Coverage floor (branch) | `pytest --cov` | yes (`fail_under` in `pyproject.toml`) |
| 3 | Executable specs (Gherkin) | included in gate 1 | yes |
| 4 | Audit reproducers | `AUDIT_REPRO=1 pytest polymarket_bot/tests/audit -q` | yes |
| 5 | Mutation check | `python qa/mutation_check.py` | yes |
| 6 | Chaos drills | `python -m polymarket_bot.main --mode chaos` | yes |
| 7 | Lint (bug rules only) | `ruff check polymarket_bot` | review, not a hard block |
| 8 | Dependency audit | `pip-audit` | review |

Run all of them: `bash qa/run_all.sh`.

## Why each layer exists

**1. Unit + integration.** The baseline. Its limitation is the reason every
other layer exists: 458 of these were green while two Critical defects sat in
the money paths, because none of them drove a *resting* order, a *killed* fill,
or a *poisoned* number through a money path.

**2. Branch coverage, not line coverage.** Branch is on deliberately. The gap
between the two (≈73% branch vs ≈84% line) is made of untaken error paths —
precisely where the audit findings lived. A line-coverage number would have
looked healthier and told us less.

The floor sits just *below* the current value. A ratcheting gate gets disabled
the first time it is inconvenient; raise it deliberately when a module is
finished, never automatically.

**3. Executable specifications (`tests/features/*.feature`).** Plain-language
scenarios bound to real code, so a reviewer or a buyer can check the safety
contract without reading Python. They are written to assert **effects** ("no new
orders are allowed"), never internal state ("the paused flag is true") — a
mutant that removed the pause term from `trading_allowed` survived the old
state-asserting drills.

**4. Audit reproducers (`tests/audit/`).** Each one *failed* on the commit that
introduced it and now passes. They are kept forever: a regression suite whose
provenance is a real defect, not a guess about what might break. Gated behind
`AUDIT_REPRO=1` so the everyday run stays fast.

**5. Mutation testing (`qa/mutation_check.py`).** Coverage says a line ran;
mutation asks whether anything would *notice* if that line were wrong. Targeted
at the modules that guard capital rather than run repo-wide — a single
whole-repo score is the least actionable number in testing, and a full `mutmut`
run takes hours. A survivor is a **test gap**, not a code bug.

Current: **13/13 killed** across `risk.py`, `fees.py`, `rewards.py`, `clob.py`,
`ledger.py`, `models.py`. This layer found F-009 (four survivors in the
safety module, including "pause no longer stops trading").

**6. Chaos drills.** Failure injection against the live object graph: WS
outage, corrupt feed, desync, loss breach, losing streak. Verifies the system
*reacts*, where mutation verifies the tests *notice*.

**7. Lint, scoped to bugs.** `ruff` is configured to bug-catching rules only
(`B`, `PLE`, `F`, `RUF`, `DTZ`, `S`, `PLW`). The full `ALL` set produces 900+
`assert`-in-test and missing-docstring complaints that bury real findings — we
ran it once, and every genuine hit was already covered. Style is not a release
gate here.

**8. Dependencies.** `requirements.txt` uses unpinned `>=` with no lockfile, so
builds are **not reproducible** — a known, accepted gap (audit F-013). Pin a
lockfile before shipping to a third party.

## Adding a finding

1. Write the reproducer in `polymarket_bot/tests/audit/test_fNNN_*.py` and
   confirm it **fails on the current commit**. A finding without a failing test
   is a hypothesis.
2. Record it in `docs/audit/REGISTER.md` with severity, confidence, location,
   trigger condition, and the observable effect.
3. Fix it. Re-run the reproducer, the full suite, and the mutation check.
4. If the fix guards a behaviour, add a mutant for it to `qa/mutation_check.py`
   — otherwise the guard itself is untested.

## Before enabling live trading

- [ ] All gates green
- [ ] `--mode paper` has run ≥ 7 days with the intended config
- [ ] `--mode report` shows non-negative markout at the fill horizon
- [ ] Exposure/loss limits set for the capital actually deployed
- [ ] `.env` holds real keys; `execute: false` flipped only where intended
- [ ] A manual `/pause` via Telegram has been tested end-to-end

## Known limits of this QA stack

Stated plainly, because a QA document that claims completeness is the least
trustworthy kind:

- **No live-exchange verification.** No order has ever been placed against
  Polymarket from this environment. Real fill semantics, duplicate responses,
  and reconnect gaps are reasoned from code, not observed.
- **No `client_order_id` idempotency** (audit F-010, still open): a duplicated
  exchange response cannot be de-duplicated. Needs an SDK-level reproducer.
- **Mutation is targeted, not exhaustive** — 13 hand-chosen mutants, not a
  repo-wide score. Do not quote it as one.
- **Fuzzing is boundary-targeted**, not a campaign: Gamma REST, the WS frame
  parser and the LLM cache are covered; the Telegram parser is not.
- **The on-chain redemption path is untested** — it needs a fork/testnet
  harness. It is gated behind three separate opt-ins.
