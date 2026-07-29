# Risk engine — Hyperliquid portfolio risk layer

Pre-trade risk estimation for a book of Hyperliquid perpetual positions.
Estimates the distribution of outcomes under stated assumptions. It does not
forecast prices, does not give advice, and cannot return a point estimate
without an interval around it.

Section references (§) are to the product specification. The engineering
decisions that depart from it — and the places where it cannot be
implemented as literally written — are in
[`docs/hl-risk/OPEN-QUESTIONS.md`](../docs/hl-risk/OPEN-QUESTIONS.md).

## Status: Phases 1-3 complete

| Phase | Scope | State |
|---|---|---|
| 1 | Liquidation model, risk engine, §3.1 benchmarks, calibration journal, shadow cron | **complete, gate passing** |
| 2 | `pre_trade_delta` | **complete** |
| 3 | Read-only frontend, backend, degradation contract | **complete, acceptance verified** |
| 4 | `max_safe_size`, builder fee — gated on 21 days x 200 addresses of shadow validation | not started, gate closed |
| 5 | `funding_drag`, observability, polish | partial (metrics exist) |

Phases 2 and 3 added an internal REST service (`risk_engine/service/`), a
Node backend (`services/backend/`) that enforces the §6 degradation
contract, and a Next.js frontend (`apps/web/`). See
[`docs/hl-risk/RUNNING.md`](../docs/hl-risk/RUNNING.md).

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

## Layout

```
domain/       value types. RiskEstimate cannot be built without an interval.
liquidation/  §1. Margin tiers, closed-form liquidation price, the simulator.
model/        §2. EWMA, Ledoit-Wolf, PSD projection, Student-t marginals,
              t-copula fitting, funding AR(1), the global correlation matrix.
sim/          §2.3/2.5. Path generation, Monte Carlo engine, interval estimation.
tools/        §4. portfolio_risk, pre_trade_delta.
service/      §8. Internal REST service the Node backend consumes.
validation/   §3.1 benchmarks, §3.2 baselines, CLI.
shadow/       §3.3/3.4. Calibration journal, snapshot cron, resolver, metrics.
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
