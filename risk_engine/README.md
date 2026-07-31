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
pytest risk_engine/tests -q -m load_sensitive        # the wall-clock assertions alone
```

`load_sensitive` marks the assertions that time the engine against §2.6's
300 ms budget. They measure the machine as much as the code — on a four-core
box the §9 Phase-2 budget test passes idle and misses by more than 2x with all
four cores busy, with nothing in the engine changed — so CI runs them in a
separate **non-blocking** step and the blocking step deselects them. They are
not skipped and the budget is not relaxed: a local `pytest risk_engine/tests`
runs them like anything else, and OPEN-QUESTIONS D7 remains the record that
the budget is genuinely missed at the 5-8 positions §0 describes. If the
advisory step goes red, read the timings in the failure message before
concluding anything — a saturated runner and a real regression look identical
apart from those numbers.

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

Each address is `0x` plus 40 hex digits, in any case — paste the checksummed
form a block explorer shows you. `normalise_address` (in `domain/types.py`)
folds it to lowercase at the Info client and again at the journal write, so
one account cannot acquire two identities and inflate the §3.3 address count;
anything that is not an address is refused when the list loads, with its
index, rather than part-way through a sweep that has already spent weight.
That last clause is a statement about ordering, and it is true only because
two callers were changed to make it true: `snapshot` loads the list before it
builds the live bundle, and `ShadowCron.run_once` reads it before it asks for
`specs` or `spot`. Built the other way round, a one-character typo cost 40
weight on a 1-coin universe before the file was opened.

Two limits on that, both in OPEN-QUESTIONS C7 and neither obvious from the
code. **It prevents a split identity, it cannot heal one**: normalising at the
write makes future rows canonical and repairs nothing already written, and one
account written checksummed before the fix and lowercase after it reads back
as two accounts against §3.3's gate. That is harmless today only because the
counter has not started and no journal exists anywhere — a precondition, not a
property, and there is deliberately no detection query and no backfill.
**And the venue-side motivation is weaker than the commit that introduced it
claims**: an HTTP 422 on a checksummed address was seen once, the command that
produced it was never recorded, and it has never been reproduced. The fix does
not depend on it. It stands on §5.1's silent empty state and on the journal's
byte-compared `address` column, both of which are properties of this codebase.

**A refused list refuses the whole run, loudly.** One bad entry is not one
skipped address: the sweep writes nothing, prints the offending index, and
`snapshot` exits non-zero (1 — the list is caught while the world is being
built, before the sweep starts, so it surfaces as `SystemExit` rather than
through the sweep's own return path). That is the deliberate choice — a list
containing
something that is not an address is not the sampling frame its `frame` field
describes, and sweeping the entries that happen to parse would publish a
score against a frame nobody wrote down. It is also why the exit code
matters: an address file does not change between days, so a run that failed
quietly would fail again every day, and §3.3 needs 21 of them. `resolve` is
deliberately *not* blocked by the same file — it takes its addresses from the
journal's pending rows, and the Info API serves current state only, so
refusing to run it would strand yesterday's predictions past their staleness
window instead of costing a day of new ones.

`resolve` retries a failed row forever; there is no attempt counter. Mostly
that is what you want. But a row whose stored address is not an address —
written before the journal canonicalised, and left visible because the read
paths deliberately do not normalise — fails identically on every run and
never clears. It is documented rather than bounded (the reasoning, and the
two worse alternatives, are in `shadow/resolve.py`) and reported rather than
left to look transient: such rows are counted separately as permanent, with
the fix being a deliberate correction of the journal row.

### Collecting the address list

```bash
python -m risk_engine.market.collect_addresses --minutes 30 --out addresses.json
```

The frame is **decided**: the public trades feed. The leaderboard was
rejected because it ranks on realised performance, which is the variable the
calibration score measures — sampling on the outcome would make the model
look mis-calibrated in whichever direction the sample was skewed, and nothing
downstream recovers from that. Activity bias is awkward; performance bias is
circular.

The collector subscribes to the trades WebSocket, harvests the accounts named
on each trade, folds them through `normalise_address`, and writes a file
`FileAddressSource` reads directly — including a generated `frame` that states
the window, the coins that actually **delivered** trades (not the ones
`--coins` subscribed to; the difference can only overstate the sample), the
activity bias, and the tension that the book-unchanged cohort the gate is read
from (B2) discards precisely the most active accounts this frame selects for.
It stops at `--target` (default 500, the top of §3.3's range, because the
sweep drops flat and zero-equity accounts before any of them count towards a
200-address gate) or when `--minutes` elapses, and refuses to write fewer than
the gate's requirement without `--allow-short`. Progress prints while it runs.

A refusal never throws the harvest away. A run that comes back short writes
its addresses to `<out>.refused-<window-start>` — a loadable list whose frame
opens by saying it was refused — because the alternative is telling an
operator who just stood over a 30-minute window that their 199 addresses are
in no file anywhere. Exit codes are distinct so that each one implies its own
next move: `0` wrote it, `1` collected and refused to publish, `2` the feed
did not match the assumed message shape, `3` no usable connection (`--ws-url`,
DNS, TLS, refusal, or the missing package), `4` bad invocation, caught before
anything connects, `5` the harvest succeeded and the *write* failed — a full
disk, a read-only mount, a bad `--out` path.

`5` is the one worth reading carefully, and it was missing from this list
until 2026-07-31. It is not a collection failure: the addresses exist and the
window will not come back. **The complete file is printed to stdout** — not
written to a fallback path, because any fallback is another write that can
fail the same way — so redirect it or paste it somewhere writable. Another
collection window is not needed, and spending thirty minutes on one is
exactly the mistake collapsing `5` into `1` would cause.

Two things to know before running it. `websockets` is **not** an engine
dependency and is imported lazily — `pip install 'websockets>=12.0'`, and add
the same line to `deploy/Dockerfile.engine` if the collector is to run in the
container. And the message shape **was** an assumption; it is now an observed
fact, which this paragraph denied for a day after the fact arrived. Two live
runs from an operator's machine: 36 addresses from 30 records on 2026-07-30,
then 515 addresses from 2 989 records over 1 110 frames on 2026-07-31 with
zero unparseable records and zero anomalies. The URL, the subscribe envelope,
the channel name and the `users` field all held. Records arrived for all
three subscribed coins, so the subscription is per-coin as assumed.

The runtime assertion stays regardless, because it guards the venue changing
rather than the venue being unknown: the collector asserts the shape while
collecting and aborts with the frame quoted verbatim if a trade carries no
address where it expects one. It will never write an empty list and report
success — that file would load cleanly, sweep nothing, and show up three
weeks later as a gate that never advanced.

That assertion is fatal **only until the first address is read**. After one
has come out of the expected field the venue has demonstrated the shape, so a
later odd record is counted, warned about on the progress line, and published
in the frame and `_provenance` as a trade the list does not contain — not
turned into an abort that discards a 300-address harvest while telling the
operator the assumption "did not hold".

Any trade frame you see quoted in this repository, in the collector's tests
or in a review of it is stub-generated — the live venue is 403 at this
environment's proxy, so **no example in the tree is evidence that the shape
is right**. That caveat survives the live runs above and is narrower than it
looks: what carries the shape is the runs, recorded in OPEN-QUESTIONS B4/E4
as summary lines, and nothing in the tree. So the suite proves the parser
matches the specification, never that the specification matches the venue.
Anyone extending `TRADE_ADDRESS_FIELDS` on the strength of a passing suite is
reading it wrong; re-run `--dry-run`, it costs sixty seconds.

### Before the shadow clock starts

Changing the distribution resets the counter (§3.3, §10), so any question
that moves it has to be settled before days start accumulating — otherwise
they are days that get thrown away.

That list is now **empty**. The last two entries on it, **A1 and A8**, were
decisions rather than measurements, and both were taken on 2026-07-30: A1
keeps zero log-return drift (`DriftConvention.ZERO_LOG_RETURN`) on the
grounds that no convention is uniformly conservative — `-sigma^2/2` is harsher
for longs and softer for shorts — and that the 8 bp at stake over 24 h is an
order of magnitude below the estimation error on `sigma`; A8 accepts that
funding is simulated independently of price and bounds the resulting bias by
narrowing `funding_drag`'s default horizon from a week to 24 h, a week
remaining reachable by explicit argument and flagged as indicative there.
Neither changed the distribution, so neither cost a counter reset — but either
one taken *later* would have.

C1, C2 and C5 were closed against live data: C5 confirmed on
testnet that isolated funding is debited from the position's own margin
(§1.1 holds, no coupling term needed), C2 measured a mark-mid basis 35×
inside §1.4's threshold over a 12-hour window — and, more usefully, found
that the `BasisModel` guard C2 claimed to have never existed, so the
approximation was silent all along and is now justified by measurement
rather than by a mechanism — and C1's clamp turned out not to bind the
fitted AR(1) at all. Details and bounds in
[`OPEN-QUESTIONS.md`](../docs/hl-risk/OPEN-QUESTIONS.md).

### Sizing the window first — a pilot that does not reset

```bash
python -m risk_engine.shadow icc --journal shadow.db
```

How long the window has to be depends almost entirely on how much of a day's
breaches are one event, and that number takes data. This measures it and
prints the window it implies.

Running this pilot does **not** burn the §3.3 counter. What makes two
addresses breach together on one day is the common market move, not the
model version: a new version shifts the VaR levels, it does not change
whether BTC fell 8% that day. So the fortnight is not thrown away when the
distribution changes.

It estimates through the PIT values rather than the breaches. A breach is a
5% event — a day of 200 addresses carries about ten, and ten events cannot
resolve a correlation. The PIT values carry the same co-movement across
every observation, and a copula map converts back. `--direct-interval` shows
what the breaches alone support, which at pilot length is nearly nothing;
that contrast is the point rather than a caveat.

The window is sized off the *upper* end of the interval. Sizing off the
point estimate is wrong half the time in the direction that shortens it, and
a short window yields a gate that passes without establishing anything.

Two separate sources of noise are both handled conservatively, because both
shorten the window when read optimistically:

- the **measured ICC** has sampling error, so sizing reads its upper
  confidence bound rather than its point estimate;
- the **power simulation** has Monte Carlo error, so a day count is accepted
  only when the 95% *lower* bound on its power clears the target. A point
  estimate near the target is a coin flip: at 30 days and breach ICC 0.15
  the true power against a 10% rate is ~0.80, and at 40 trials 11 of 20
  seeds put the point estimate over target while the bound stays under it.

`--trials` controls the second one. The default (500) is what makes the
bound tight enough to read; lowering it widens the bound and pushes the
recommendation longer, never shorter.

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
market/       §5.1. Info client and parsers, the live-API verification
              harness, and the B4 trades-feed address collector.
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

## What has and has not been verified against the live API

`api.hyperliquid.xyz` is blocked at the proxy in the environment this was
built in (HTTP 403), so nothing here can be re-run from the build machine.
It **has** been run from an operator's machine, and this section said
otherwise — "has not been exercised against a live response" — for two days
after that stopped being true. Current state, each line naming the command
that produced it:

| # | Assumption | Status | Evidence |
|---|-----------|--------|----------|
| E5 | `meta`, `candleSnapshot`, `clearinghouseState` parse | **PASS** 2026-07-29 | `market.verify`: 177 assets / 34 multi-tier; 720 hourly BTC returns, 0 gaps; 10 positions, cross collateral $4.5M |
| E4 | trades-feed message shape | **PASS** 2026-07-31 | `collect_addresses`: 515 addresses from 2 989 records, 0 unparseable, 0 anomalies |
| B2 | ledger delta types classify | **PASS** 2026-07-31 | `market.verify --address`, after `send` was added; it FAILED first and that is what found `send` |
| C2 | mark ≈ mid within §1.4's threshold | **PASS** 2026-07-30 | 12-hour series: median \|basis\| 2.61e-05 against a 9.07e-04 threshold, 35× margin |
| C5 | isolated funding debits the isolated pocket | **PASS** 2026-07-31 | `probe_isolated_funding` across a funding tick: pocket absorbed 100%, cross moved $0.00 |
| C1 | the funding clamp is the documented constant | **INCONCLUSIVE** | no breach in 1 500 observations over 30 d, worst 0.1% of the cap — but a clamp is a protocol constant and no sample of realised rates can establish one. Needs the source, not more data. |
| C4 | the `webData3` subscription exists | **UNCHECKABLE** here | a WebSocket question; `market.verify` speaks only the Info POST API. The shard planner is agnostic either way. |

So: the §5.1 parsers are no longer unverified, and treating them as such
would now be its own kind of wrong — it invites re-doing settled work and
discounts a real result. What remains genuinely unestablished is C1, whose
status will not improve with more observations, and C4.

`market.verify` is the command that produces this table; run it rather than
trusting the table, because a table is a claim about a past run and the venue
can change under it.
