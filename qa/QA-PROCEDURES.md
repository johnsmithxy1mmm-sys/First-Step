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
- **Mutation is targeted, not exhaustive** — 29 hand-chosen mutants, not a
  repo-wide score. Do not quote it as one.
- **Fuzzing is boundary-targeted**, not a campaign: Gamma REST, the WS frame
  parser, the LLM cache and the Telegram parser are covered.
- **The on-chain redemption path is untested** — it needs a fork/testnet
  harness. It is gated behind three separate opt-ins.


## Seam tests — the layer added after three escapes

`polymarket_bot/tests/test_seams.py`

Three defects in a row (audit F-020, F-026, F-027) were found by a live paper run
while the whole stack above was green: 500+ tests, every targeted mutant killed,
73% branch coverage. That is a fact about the stack, not bad luck, and it is worth
being precise about why.

**None of the three was a wrong rule.** Each was a wrong *joint*:

| | seam | what the unit tests could not see |
|---|---|---|
| F-020 | a rule vs its reachable RANGE | `mark / 0.976 >= 7.0` is correct arithmetic. No price the venue allows satisfies it, so the exit returned None for the entire life of every position. |
| F-026 | an OLD BOOK vs a NEW GATE | the take was written against positions the entry gate admits, then met twenty legs the gate would refuse and liquidated three at or below cost. |
| F-027 | who OPENS vs who CLOSES | two ledger aggregations disagreed about who owned a token; the wrong one fed the capital allocator. |

Why the existing layers were structurally blind to these:

- **Unit tests** ask "given these inputs, is the rule right?" — and the inputs are
  chosen by whoever wrote the rule, so they are the inputs on which it works.
  Nobody writing a take rule constructs the book that predates it.
- **Mutation testing** asks "would the suite notice this code changing?" It cannot
  ask "is there a scenario nobody wrote a test for?" A missing scenario has no
  line to mutate.
- **Coverage** was ~74% through all three. Every line involved was executed. The
  bugs were in which lines ran *together*, and in what state.

### The four invariants

Written as properties over generated ranges and as whole-lifecycle flows, never as
examples, so they keep holding when thresholds are retuned:

1. **A take never realizes a loss; a stop never fires in profit.** Asserted over
   an entry x mark grid that deliberately includes entries the gate refuses —
   i.e. the legacy book.
2. **Every rule must be satisfiable inside its domain.** For every position there
   must EXIST a price in (0,1) at which it exits. This is the generic form of
   F-020, and it fails with the exact diagnosis: *"a fade position bought at 0.976
   cannot be exited at ANY price in (0,1) — its only exit is resolution, at full
   notional."*
3. **Ownership is a property of the opening trade, in every aggregation.**
   Parametrised over every strategy name, so adding a sixth cannot reintroduce the
   split.
4. **No money path may inherit its identity from a default.** `record_trade` now
   *requires* `strategy` — the default was F-027's root cause — and an AST scan
   asserts every production entry/exit call site states it explicitly. Zero
   exemptions: adding an exit path is precisely the change that reintroduces the
   bug, and it fails here rather than in a paper report.

Each was verified to FAIL against the reverted code before being committed.

### When to add one

Add a seam test, not a unit test, when a change introduces a joint:

- a new rule whose threshold is compared against a *bounded* quantity (price,
  probability, fraction) — check satisfiability, not just correctness;
- a new gate, filter or cap that positions already in the book would not pass;
- a new component that closes, reduces or relabels something another component
  opened;
- a new argument with a default on a path that writes to the ledger.

### What this layer still does not cover

It reasons about seams inside the process. It does not cover the seam between the
bot and the exchange, which remains the largest untested joint in the system and
is not closable from a machine that has never placed an order.
