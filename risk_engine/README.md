# Risk engine — Hyperliquid portfolio risk layer

Pre-trade risk estimation for a book of Hyperliquid perpetual positions.
Estimates the distribution of outcomes under stated assumptions. It does not
forecast prices, does not give advice, and cannot return a point estimate
without an interval around it.

Section references (§) are to the product specification. The engineering
decisions that depart from it — and the places where it cannot be
implemented as literally written — are in
[`docs/hl-risk/OPEN-QUESTIONS.md`](../docs/hl-risk/OPEN-QUESTIONS.md).

## Status: Phases 1-3 and 5 complete

| Phase | Scope | State |
|---|---|---|
| 1 | Liquidation model, risk engine, §3.1 benchmarks, calibration journal, shadow cron | **complete, gate passing** |
| 2 | `pre_trade_delta` | **complete** |
| 3 | Read-only frontend, backend, degradation contract | **complete, acceptance verified** |
| 4 | `max_safe_size`, builder fee — gated on 21 days x 200 addresses of shadow validation | not started, gate closed |
| 5 | `funding_drag`, observability, champion/challenger | **complete** |

Phases 2 and 3 added an internal REST service (`risk_engine/service/`), a
Node backend (`services/backend/`) that enforces the §6 degradation
contract, and a Next.js frontend (`apps/web/`). See
[`docs/hl-risk/RUNNING.md`](../docs/hl-risk/RUNNING.md).

Phase 5 is code-complete but Phase 4 does not follow from it. The gate is
shadow validation, not features, and the shadow counter has not started —
see "Before the shadow clock starts" below.

## Running the gate

```bash
pip install -r risk_engine/requirements.txt
python -m risk_engine.validation.cli benchmarks     # §3.1, exits non-zero on failure
python -m risk_engine.validation.cli shadow --journal shadow.db
pytest risk_engine/tests -q                          # excludes the slow gate
pytest risk_engine/tests/test_benchmarks.py -q       # the gate itself, ~15s
```

To run the whole read-only stack, see
[`docs/hl-risk/RUNNING.md`](../docs/hl-risk/RUNNING.md) or
`./scripts/run-stack.sh`.

## Running the shadow harness

Two jobs on a daily cadence. `snapshot` writes today's predictions,
`resolve` fills in what happened a day later; between them they accumulate
the window Phase 4 is gated on.

```bash
python -m risk_engine.shadow init-addresses --out addrs.json   # template
python -m risk_engine.shadow snapshot --journal shadow.db --addresses addrs.json
python -m risk_engine.shadow resolve  --journal shadow.db --addresses addrs.json
python -m risk_engine.shadow progress --journal shadow.db
```

`--fixture` swaps in a synthetic market and books, which exercises every
moving part without a venue and validates nothing; every row it writes is
stamped with a frame that says so. `--journal` takes a path for SQLite or a
`postgresql://` DSN; the journal behaves identically on both, which
`test_journal_backends.py` checks by running the same code against a real
server (`HL_TEST_POSTGRES_DSN=... pytest -k backends`).

A live run needs `--addresses`, and the file needs a non-empty `frame`
field. That is deliberate: the Info API enumerates no addresses, so every
list is biased somehow (OPEN-QUESTIONS B4), and a calibration score is
uninterpretable without knowing what it is a sample of. `FileAddressSource`
refuses a list that omits it rather than defaulting to something plausible.

### Before the shadow clock starts

Changing the distribution resets the counter (§3.3, §10). Five open
questions still move it — A1, A8, C1, C2 and C5 — so 21 days accumulated now
are 21 days that will be thrown away when any of them is answered. Resolve
them first, or accept the reset knowingly. Details in
[`OPEN-QUESTIONS.md`](../docs/hl-risk/OPEN-QUESTIONS.md).

`progress` reads the gate off the `book_unchanged` cohort and the
day-clustered interval, not the naive one. B1 records why: addresses
observed on the same day share one market move, so the naive interval calls
a coin flip significant, and under honest clustering §0.3's VaR criterion is
not reachable in 21 days.

## Champion/challenger

`risk_engine/shadow/champion.py` scores two distribution versions against
the same realised outcomes:

```python
from risk_engine.shadow.champion import compare
print(compare(journal, champion_version="0.2", challenger_version="0.3"))
```

Paired by (address, day) and bootstrapped over days. Only observations both
versions predicted are compared — a challenger always starts later, and
scoring it on days the champion never saw compares two models on different
markets. It migrates only when both bounds of the day-clustered interval sit
below zero; "looks better" is not a verdict it can return.

## Layout

```
domain/       value types. RiskEstimate cannot be built without an interval.
liquidation/  §1. Margin tiers, closed-form liquidation price, the simulator.
model/        §2. EWMA, Ledoit-Wolf, PSD projection, Student-t marginals,
              t-copula fitting, funding AR(1), the global correlation matrix.
sim/          §2.3/2.5. Path generation, Monte Carlo engine, interval estimation.
tools/        §4. portfolio_risk, pre_trade_delta, funding_drag.
service/      §8. Internal REST service the Node backend consumes.
validation/   §3.1 benchmarks, §3.2 baselines, CLI.
shadow/       §3.3/3.4. Calibration journal (SQLite or Postgres), snapshot
              cron, resolver, metrics, champion/challenger, CLI.
market/       §5.1. Info client and parsers.
observability/§7. Counters and latency histograms.
```

## The four decisions worth knowing before reading the code

**One cross pool plus N independent isolated conditions.** The simulator
tracks N+1 separate liquidation conditions on every step of every path. A
single aggregated condition over the whole book is a different and wrong
model: a blown isolated pocket must not touch the cross pool.

**Within-step monitoring.** Checking the margin condition only at hourly
closes misses every excursion that breaches and recovers inside the hour,
which *understates* liquidation probability — the one direction §10
prohibits. Each step applies a Brownian-bridge correction on the margin gap.
`benchmark_1b` measures the difference and asserts its sign: on the reference
book, close-only monitoring gives 12.0% against a corrected 14.9%.

**One global correlation matrix, sliced per request.** A principal submatrix
of a positive-definite matrix is positive definite, so every slice
factorises without a per-request projection, there is nothing to cache per
user universe, and an asset outside the user's book costs a slice rather
than a cold start.

**Intervals are load-bearing.** `RiskEstimate` validates that the point sits
inside its interval and that its timestamp is timezone-aware. The engine
escalates its path count until the interval on `P(liq)` meets §2.5's 2 pp
ceiling, and marks the result `converged=False` if it cannot — callers refuse
to publish those.

## The §3.1 gate, as it currently measures

```
[PASS] 3.1.1 closed-form first passage: simulated=0.14805 closed_form=0.14931 |diff|=0.125pp tol=0.5pp
[PASS] 3.1.1b discrete monitoring understates risk: close-only=0.12039 < bridge-corrected=0.14886
[PASS] 3.1.2 rho=1 equals one doubled position: |diff|=0.0000pp tol=0.3pp
[PASS] 3.1.3 P(liq) monotone in correlation: -1: 0.00000 < 0: 0.00470 < +1: 0.03635
[PASS] 3.1.4 unlevered book does not liquidate: cross 1x=0.000000 isolated 1x=0.000000
[PASS] 3.1.5 P(liq) strictly increasing in leverage: cross 0.02202..0.62790, isolated 0.02167..0.62915
```

Benchmarks 2, 3 and 5 run under common random numbers. Without that,
§3.1.2's 0.3 pp tolerance is one standard error at 20 000 paths, and
"strictly increasing over 20 grid points" is a coin flip on a correct engine.

## Not verified against the live API

`api.hyperliquid.xyz` was blocked at the proxy in the environment this was
built in (HTTP 403). Every parser in `market/` is written against the
documented response shapes and recorded fixtures, and **has not been
exercised against a live response**. Re-verify before anything downstream of
it is trusted. Same applies to the funding-rate protocol clamp in
`model/funding.py` and to the mark-versus-trade price basis in §1.4.
