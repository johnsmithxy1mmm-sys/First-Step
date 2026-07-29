# Specification review — contradictions, ambiguities, missing data

Written before any code, per §11.2. Each entry states what the specification
says, why it cannot be implemented as literally written, what the code does
instead, and — where the choice biases risk — in which direction.

`[BLOCKER]` = cannot be resolved from the specification, needs a decision or
external data before the phase that depends on it can close.
`[RESOLVED]` = a defensible reading exists, taken, and recorded here.

---

## A. Mathematical contradictions

### A1 `[BLOCKER]` Zero price drift is unattainable with Student-t marginals

§2.4 mandates zero drift **in price** under the real measure. §2.2 mandates
Student-t marginals with `df` clamped as low as 2.1. These are mutually
exclusive: zero price drift requires `E[e^r] = 1`, and the moment generating
function of a Student-t is infinite for every `df`. There is no admissible
drift constant that makes a t-distributed log-return series a price
martingale — the expected price is `+inf` at every horizon regardless of drift.

Taken: **zero mean log-return** (`E[r] = 0`), i.e. the price is
median-flat rather than mean-flat. This is still "no prediction" in the sense
§2.4 is defending — it is the unique location that treats up and down
symmetrically.

Direction of bias vs. the literal reading: the literal reading implies a log
drift of `-sigma^2/2`, so zero log-drift is *less* conservative for longs and
*more* conservative for shorts, by `sigma^2/2` per horizon. At a 4% daily
vol that is 8 bp over 24 h — an order of magnitude below the estimation
error on sigma itself, but it is a real one-sided choice and §10 forbids
silent risk-understating choices, hence this entry.

Decision needed: accept zero log-drift, or accept `-sigma^2/2` log-drift
(price-martingale in the limit of finite moments) as the conservative option.
The engine exposes this as `DriftConvention` with no default hidden in code.

### A2 `[RESOLVED]` The two liquidation-price formulas in §1.2 are not equivalent

§1.2 gives the Hyperliquid formula and then says isolated is "equivalent to"
`liq = entry * (1 - 1/L + mmr)`. Deriving the exact isolated liquidation
price from the §1.1 condition gives

```
long : liq = entry * (1 - 1/L) / (1 - mmr)
short: liq = entry * (1 + 1/L) / (1 + mmr)
```

The spec's version is the first-order expansion of this; the two differ by
`O(mmr/L)`. That is small (≈6 bp of entry at L=20, mmr=1.25%) but signed:
the approximation puts the long liquidation price *above* the true one
(conservative) and the short liquidation price *above* the true one
(**understates short risk**). §10 forbids the latter, so the exact form is
implemented and the approximation is not used anywhere.

The Hyperliquid formula itself reproduces the exact result, provided
`position_size` in it is read as `|size|` and `l` as the maintenance margin
*rate*. Read with a signed `position_size` it returns a below-spot
liquidation price for shorts, which is nonsense; the code notes this.

### A3 `[RESOLVED]` Hourly-close monitoring understates liquidation

The simulator steps hourly (§2.5), but liquidation is a continuous-time
barrier event. Checking only at step closes misses every excursion that
breaches and recovers inside the hour, so a naive discrete simulator
*understates* `P(liq)` — exactly the direction §10 prohibits.

Implemented: a Brownian-bridge correction on the margin gap `g = equity -
maintenance_margin`. Within each step, given `g` at both ends and the local
gap volatility from a delta approximation, the probability of an interior
breach is `exp(-2 g0 g1 / var(dg))`, drawn per path per step. This restores
continuous-monitoring behaviour and, as a side effect, makes benchmark §3.1.1
comparable against the *uncorrected* closed form rather than needing a
Broadie-Glasserman-Kou barrier shift. The correction can be disabled, and a
test asserts that disabling it lowers `P(liq)` — i.e. that the bias is real
and signed as claimed.

### A4 `[RESOLVED]` "PSD projection" is not sufficient for Cholesky

§2.1 argues that a slice of a positive-definite matrix stays positive
definite. True — but the section asks for a **PSD** projection, and a merely
positive-*semi*-definite global matrix can have a zero eigenvalue, whose
slice breaks the Cholesky factorisation that the same section relies on. The
projection therefore targets positive-*definiteness* with a strictly positive
eigenvalue floor. The floor is logged with every correction, as §2.1 requires.

### A5 `[RESOLVED]` Ledoit-Wolf is not defined for exponentially weighted moments

§2.1 asks for EWMA moments (half-life 20 days) *and* Ledoit-Wolf shrinkage.
The published LW shrinkage intensity is derived for an i.i.d. sample
covariance; it has no closed form under exponential weighting. Implemented
with the effective sample size `n_eff = (sum w)^2 / sum w^2` substituted for
`n` in the LW constant-correlation estimator. This is an adaptation, not the
textbook estimator, and is marked as such in the code and in the model
version string.

### A6 `[BLOCKER]` §3.1.2's 0.3 pp tolerance is below Monte Carlo noise

At 20 000 paths the standard error on a probability near 10% is 0.21 pp, so
the *difference* of two independent estimates has a standard error of 0.30 pp
— the tolerance is one sigma, and the benchmark would fail roughly a third of
the time on a correct engine. The benchmarks therefore run with common random
numbers across the compared configurations and an elevated path count, which
makes the comparison near-deterministic. Benchmark §3.1.5 (monotonicity in
leverage) is likewise only meaningful under common random numbers; evaluated
on independent path sets, "strictly increasing" on a 20-point grid is a
coin-flip proposition regardless of correctness.

### A7 `[BLOCKER]` §3.1.5 does not say which leverage

For **cross** positions the spec itself establishes (§1.2) that the set
leverage does not affect the liquidation price — only the notional-to-equity
ratio does. So "P(liq) strictly increases in leverage" is false as stated for
a cross book: raising the leverage slider changes nothing. The benchmark is
implemented twice: for cross books the grid varies *effective* leverage
(position size at fixed collateral), and for isolated positions it varies the
set leverage `L`. Confirmation wanted that this is the intent.

### A8 `[BLOCKER]` Funding is simulated independently of price

Funding shocks are drawn independently of price shocks. In reality the
funding rate tracks the perp-spot premium, which correlates with recent
returns, so a falling market pushes funding negative. Under independence a
long in a crashing market keeps paying funding it would in fact be
receiving, and a short in a rallying market likewise. The sign of the net
bias differs by side and by horizon, so it cannot be waved through as
"conservative". Wanted: the empirical correlation between hourly funding and
hourly returns per asset, measured over the shadow window; if it is
material, the AR(1) needs a return-driven term.

### A9 `[RESOLVED]` §2.3 never says how the copula's degrees of freedom are chosen

The marginals' `df` has an estimator (§2.2) and a clamp; the copula's does
not, and it is a different parameter with a different meaning — it controls
how strongly assets go extreme *together*. Implemented as a two-stage (IFM)
MLE: correlation fixed at the shrunk estimate, copula `df` profiled over a
grid by the copula likelihood on pseudo-observations.

### A10 `[RESOLVED]` The §2.3 diagnostic must not compare against the asymptotic coefficient

The natural implementation — compare the empirical lower-tail dependence
against the t-copula's closed-form `lambda` — is wrong, and wrong in the
direction that discredits the check. The closed form is the limit as the
threshold goes to zero; at a workable threshold like the 5% quantile the
finite-sample quantity is much larger (for `nu=4, rho=0.6`: about 0.40
against an asymptotic 0.31). A correctly specified model would therefore be
flagged as understating its own lower tail every time, and a gate that
cries wolf gets switched off. The diagnostic compares the empirical estimate
against the *same statistic at the same threshold* computed from copula
draws.

Note the outcome §2.3 prescribes when the check does fire — switch to a
skewed-t — is **not implemented in Phase 1**. The check raises rather than
degrading quietly, per §9.

---

## B. Validation-methodology problems

### B1 `[BLOCKER]` The calibration tests assume independent observations; they are not

§0.3 and §3.3 want the VaR@95 breach rate inside a *binomial* 95% interval,
and PIT uniformity under a Kolmogorov-Smirnov test. Both assume independent
observations. 200-500 addresses observed on the same day share one market:
their 24 h equity changes are strongly cross-correlated, and on a day when
BTC drops 8% nearly every address breaches at once. The effective sample
size is closer to the number of *days* (21) than to the number of
observations (4200+), so the naive binomial interval is far too tight and
will reject a correctly calibrated model.

Implemented: breach-rate and PIT statistics are computed both naively and
with day-clustered inference (block bootstrap over days; the day is the
independent unit). The gate should be read off the clustered version.

**Measured**, replacing the "roughly ±4 pp" estimate that stood here, which
was arithmetic on the day count rather than a measurement. Run it with
`python -m risk_engine.validation.power`; the generator is beta-binomial, its
realised intra-class correlation is checked against the requested one, and
the fast interval is checked against `clustered_bootstrap_ci`
(`test_power.py`). 300 trials per cell:

| days | addr/day | ICC | §0.3 rejects a *correct* model | clustered ±pp | power vs 8% | power vs 10% |
|---|---|---|---|---|---|---|
| 21 | 200 | 0.00 | 3.7% | 0.61 | 100% | 100% |
| 21 | 200 | 0.05 | 58.0% | 1.94 | 71% | 97% |
| 21 | 200 | 0.10 | 65.7% | 2.70 | 41% | 81% |
| 21 | 200 | 0.20 | 77.7% | 3.60 | 18% | 49% |
| 21 | 200 | 0.40 | 81.7% | 4.87 | 11% | 20% |
| 21 | 500 | 0.20 | 88.0% | 3.43 | 23% | 47% |
| 30 | 200 | 0.20 | 81.3% | 3.19 | 25% | 66% |
| 60 | 200 | 0.20 | 76.7% | 2.42 | 55% | 92% |
| 90 | 200 | 0.20 | 76.0% | 1.98 | 75% | 98% |
| 180 | 200 | 0.20 | 77.0% | 1.38 | 97% | 100% |
| 180 | 200 | 0.40 | 81.3% | 1.99 | 71% | 97% |

The ICC=0 row is the harness calibrating itself: with genuinely independent
observations §0.3's interval rejects a correct model 3.7% of the time against
its nominal 5%, so the failures in every other row are the clustering, not a
bug in the measurement.

Three findings, none of which the estimate had:

1. **§0.3 as written is not marginally wrong, it is inverted.** At any
   non-zero clustering it rejects a *correctly calibrated* model more often
   than it accepts one — 58% at ICC 0.05, 78% at 0.20. Whatever it is
   measuring, it is not calibration.

2. **More addresses buy almost nothing.** 200 → 500 per day moves the
   clustered half-width from 3.60 to 3.43 pp at ICC 0.20. Under independence
   2.5× the sample would cut it by 37%. The day is the unit; sampling harder
   is not a substitute for waiting, and the §3.3 window's "× 200 addresses"
   is doing far less work than its "21 days".

3. **Days are the only lever, and 21 is not enough at plausible
   clustering.** To reach 80% power against a model whose true breach rate is
   double what it claims: ~21 days at ICC 0.05, ~30 at 0.10, ~60 at 0.20,
   ~180 at 0.40. Against a 60% understatement (8% vs 5%) the same targets are
   roughly 21 / 45 / 180 / not reached at 180.

ICC itself is the one number neither the specification nor this code can
supply — it takes real data, which is why it is swept rather than assumed.

**Measuring it is `python -m risk_engine.shadow icc`**, and doing so does not
burn the §3.3 counter: what makes two addresses breach together is the common
market move, not the model version, so the estimate survives a version change
to first order. A pilot can therefore run *before* A1, A8, C1, C2 and C5 are
settled.

Three things had to be got right for that command to mean anything, and each
was measured rather than assumed:

**The obvious estimator does not work.** ANOVA moments on the breach
indicators are unbiased and useless at pilot length: a breach is a 5% event,
so a day of 200 addresses carries about ten of them, and ten events cannot
resolve a correlation. At 14 days the estimate has sd 0.117 on a true 0.20,
and an interval with correct coverage spans roughly [0, 0.85]. Thirty days
barely improves it.

**Two standard intervals undercover, both failing low.** The day-clustered
percentile bootstrap covers 43% at 14 days against a nominal 95%; the
normal-theory F interval covers 66% at ICC 0.20 and 50% at 0.40. Failing low
matters specifically: the window is sized off the *upper* end, so an interval
whose ceiling is too low produces a window that is too short — §10's
forbidden direction reached by arithmetic. Both were discarded. The shipped
intervals invert the test against the validated generator and cover 92-100%.

**The PIT values carry the same information and far more of it.** They exist
for every observation, not just the 5% that breach. On the `Phi^-1(PIT)`
scale a day's common shock is an ordinary intra-class correlation, and it is
recovered accurately (true 0.30 → 0.3044) at roughly 2.5× the relative
precision. `breach_icc_from_latent` maps it back through the bivariate
orthant probability, matching the empirically realised breach ICC to within
0.02 across latent 0.05–0.50.

That map compresses, and the copula chosen decides by how much — both
columns measured, and **the conclusion must be read off the t column,
because that is the tool's default and the engine's own copula** (§2.3; one
chi-square mixing draw shared across assets per step, `sim/paths.py`):

| latent ρ | breach ICC (Gaussian) | breach ICC (t, df=4 — default) |
|---|---|---|
| 0.00 | 0.000 | 0.077 |
| 0.05 | 0.012 | 0.095 |
| 0.15 | 0.041 | 0.130 |
| 0.30 | 0.098 | 0.195 |
| 0.50 | 0.204 | 0.301 |

The prior stated here before — "on a day BTC drops 8% nearly every levered
address breaches at once, so ICC is well above 0.2" — conflated the two
scales. Latent co-movement of 0.3–0.5 is plausible; under the default t map
that implies breach ICC ~0.20–0.30, i.e. **roughly 60–180 days** by the
power table, and under the Gaussian map ~0.10–0.20, i.e. 30–90. The pilot
decides which row is real; the honest range before it runs is wide.

Two facts about the t column, both verified by simulating the actual
shared-mixing t world rather than trusting the map:

- **It has a floor.** At ρ=0 the breach ICC is ~0.077 (measured 0.079 in
  simulation), because a fat-tailed day inflates every address at once even
  with zero correlation. Under the engine's own copula the §0.3 gate can
  never be sized as if observations were independent, however quiet the
  market — the floor alone puts the effective day-clustering near the 0.05
  row of the power table.
- **The estimator composes without material bias.** The latent ρ is measured
  by Gaussian-scores ANOVA and fed to a t-parameterised map; against true
  shared-mixing t data (16 replications per ρ) the mean bias is within
  ±0.007 across ρ 0–0.5, with per-pilot scatter up to ±0.05 at high ρ. The
  scatter is what the confidence interval absorbs — sizing reads the upper
  bound, not the point. Pinned by a regression test.

**Still a decision, now an informed one.** Three options, in the order I would
take them:

- **Restate the criterion in days.** "The day-clustered 95% interval on the
  breach rate contains 5%" is a test that can be run at 21 days and is honest
  about what it establishes — which is little. Cheapest, and the one that
  keeps §3.3's window; it must be paired with stating publicly that the gate
  bounds the breach rate to roughly ±3.6 pp, not that it validates 5%.
- **Size the window from measured ICC.** Run 14 days, measure ICC, then read
  the required window off the table and commit to it before the counter is
  read. This is the only option that yields a gate with real power.
- **Keep 21 days and the naive interval.** Not defensible: it fails correct
  models four times out of five, and a gate that fails on noise will be
  re-run until it passes, which is §10's overfitting prohibition arrived at
  by procedure instead of by intent.

### B2 `[BLOCKER]` Equity changes for reasons the model does not predict

The PIT test compares predicted 24 h equity change against realised equity
change. But equity also moves when the user deposits, withdraws, opens,
closes or resizes positions — none of which the model claims to predict.
Scoring those observations as model error makes the calibration score
meaningless, and in a direction that varies with how active the sampled
addresses are.

Implemented: the resolver records deposits/withdrawals and a book-change
fingerprint alongside the realised value, and metrics are reported on three
cohorts — all observations, book-unchanged-only, and no-external-flow-only.
The gate should be read off the book-unchanged cohort. Note this filters out
precisely the most active traders, which is a selection effect worth stating
in any public calibration score (§3.4).

### B3 `[BLOCKER]` Baseline A is not a distribution

§3.2 defines Baseline A as "the historical unconditional frequency of
liquidations at that nominal leverage" — a single probability. §0.2 and §3.3
require CRPS comparison against it, and CRPS needs a full predictive
distribution of 24 h equity change.

Implemented as: the book is collapsed to its net notional exposure, and the
predicted equity change is `net_exposure x r`, with `r` bootstrapped from the
historical unconditional distribution of BTC 24 h log returns. That is a
genuinely naive predictor with no correlation structure, no per-asset
volatility and no funding, which is the spirit of the baseline. Its `P(liq)`
comes from the same draws run through the real liquidation model. Confirm
this reading, or supply the intended one.

### B4 `[BLOCKER]` No documented way to obtain the shadow address list

§3.3 requires snapshotting "200-500 active public addresses". The Info API
reads any address but does not enumerate addresses — there is no endpoint
that returns a list of accounts. The candidates are harvesting addresses from
the public trades WebSocket feed, or from the leaderboard endpoint. Both need
to be verified, and the leaderboard is a biased sample (it selects on
performance, which is exactly the variable we are calibrating against). This
is a missing input, not an implementation detail: the shadow cron is built
with a pluggable address source and ships with a file-backed one, so it runs
today, but the sampling frame is undecided and it will bias the calibration
score if chosen carelessly.

### B5 `[RESOLVED]` §0.2 and §3.2 disagree about which baseline decides

§0.2 requires CRPS strictly better than *both* baselines. §3.2 says B is the
decisive one and that failing against B means simplifying. Both are computed
and reported separately; the champion/challenger machinery treats B as the
gate and A as context.

---

## C. Hyperliquid integration — facts that must be verified, not assumed

### C1 `[BLOCKER]` The funding-rate protocol clamp (§1.5)

§1.5 correctly forbids inventing the bound, and the AR(1) is unusable without
it. The engine takes the clamp from a configuration record carrying a
`source` field, defaults to ±4%/hour attributed to the Hyperliquid docs, and
**refuses to run** if any observed historical funding rate exceeds the
configured clamp — a stale or wrong bound then fails loudly instead of
silently truncating reality. The value still has to be confirmed against the
live API before Phase 4.

### C2 `[BLOCKER]` Mark-vs-trade price basis (§1.4)

§1.4's procedure requires live WebSocket collection of mark and mid to
measure the basis. That measurement does not exist yet, so the "mark ≈ trade
price" approximation is *not* silently adopted: the engine requires an
explicit `BasisModel`, and the only one available before measurement is
`UnmeasuredBasis`, which is flagged in every result it touches and counted in
observability. §1.4's own condition (median |basis| < 25% of a typical hourly
move) cannot be evaluated until shadow mode has run.

### C3 `[RESOLVED]` Builder fee units (§5.4)

`f: 20` is 20 *tenths of a basis point* = 2 bp = 0.02%, and `maxFeeRate`
0.03% is the approval ceiling above it. Consistent, but the unit is a 10x
footgun and the spec's own prose alternates between the two numbers when
describing the positioning; the UI copy should quote the charged 0.02%, not
the 0.03% ceiling.

### C4 `[BLOCKER]` `webData3` (§5.2)

`webData2` is the documented subscription. I have no confirmation that
`webData3` exists. To be verified against the live API before the Phase 3
listener is written; the shard planner is agnostic either way.

### C5 `[BLOCKER]` Isolated-position funding

The model debits funding on an isolated position from that position's
isolated margin (which is what makes isolated liquidations independent, per
§1.1). Behaviour at the protocol level should be confirmed — if funding on
isolated positions is instead debited from the cross pool, the independence
claim in §1.1 is violated and the simulator needs a coupling term.

### C6 `[BLOCKER]` Shadow cron vs. rate limit (§3.3 vs §5.3)

500 addresses x ~20 weight per `clearinghouseState` is 10 000 weight, against
a 1200/minute budget: a full sweep costs 8.3 minutes of the *entire* budget,
before any candle or funding traffic, and §5.3 requires the cron to yield to
live users. Daily snapshots are feasible; the resolver doubles the traffic.
The cron is built with a weight-budget governor and an explicit low-priority
lane, but the address count and the live-user headroom are coupled and should
be sized against real traffic, not assumed.

---

## D. Product-surface ambiguities

### D1 `[RESOLVED]` §2.5's path count and §2.6's latency rule contradict each other

§2.5 fixes 20 000 paths and forbids ever returning a `P(liq)` whose 95% CI
half-width exceeds 2 pp. §2.6 says that when the 300 ms budget is missed,
paths should be cut "to the lower bound in 2.5" — but §2.5 states no lower
bound, and cutting paths widens the CI, which is the one thing §2.5 forbids.

Taken: **the CI rule wins**. The engine computes the smallest path count that
satisfies the 2 pp rule at the observed probability and treats that as the
floor; if meeting it costs more than 300 ms, the result is returned late and
the latency violation is counted in observability. A fast wrong number is the
failure mode this whole document exists to prevent. Note that near `p = 0.5`
the 2 pp rule alone requires ~9 600 paths, so the binding constraint is
usually 20 000 anyway.

### D2 `[RESOLVED]` `max_safe_size` may have no answer

§4.3 defines the answer as the largest size whose *upper CI bound* on
`P(liq)` stays under the threshold. If the smallest tradable increment
(`szDecimals`) already breaches the threshold, there is no safe size, and
the type must be able to say so rather than returning zero — zero and "no
safe size" are different statements to a user. The return type carries that
case explicitly. Binary search over a noisy objective also needs common
random numbers across iterations to stay monotone; the search fixes its base
randomness per query.

### D3 `[BLOCKER]` "Effective leverage" is unsigned (§4.1)

§4.1 defines it as `sigma(equity 24h) / sigma(BTC 24h)` and the UI string is
"your book moves like BTC with leverage X". Those do not match: the ratio of
volatilities is unsigned and direction-free, so a market-neutral book with
large idiosyncratic variance reads as "BTC with leverage 3" while having no
BTC exposure at all — and a short book reads identically to a long one. The
engine returns the specified ratio *and* the regression beta to BTC; the UI
copy should not claim directionality that the ratio does not carry.

### D4 `[BLOCKER]` §6's 60-second staleness rule vs §2.1's 5-minute rebuild

The global correlation matrix is rebuilt every 5 minutes by design. If the
60-second staleness rule applied to it, every result would be permanently
stale and the execution button permanently disabled. The two clocks are
therefore separated: the 60 s / 5 min contract governs the *book and mark
price* inputs (WebSocket-driven, sub-second in normal operation), and matrix
age carries its own longer threshold, exposed separately on the health
endpoint as §6 requires. Confirmation wanted, since this is the one rule
§6 says not to soften for conversion.

### D6 `[BLOCKER]` §4.2's overlap rule is the wrong significance test

§4.2 says that when the "before" and "after" intervals overlap, the UI must
report statistical indistinguishability. Applied to the paired estimates this
tool actually produces, that rule is wrong, and wrong in the dangerous
direction.

Both books are walked over identical price paths (common random numbers),
because the effect of one order is far smaller than the Monte Carlo error on
either side -- two independent runs would report mostly sampling noise. Under
that pairing the concordant paths cancel exactly, and the interval on the
*difference* is several times tighter than either marginal interval.
Non-overlap implies significance; overlap does **not** imply insignificance.

Measured on the reference book: an order of 40 SOL raises `P(liq)` by
0.17 pp with a paired interval six times tighter than the marginal ones.
The marginal intervals overlap heavily, so §4.2's rule would tell the user
"no detectable change" about an order that provably increases their
liquidation probability -- the §10-forbidden direction.

Implemented: both answers are returned. `marginal_intervals_overlap` is
exactly what §4.2 asks for; `change` is the paired interval and
`distinguishable` is read off it. `distinguishable` is what the UI should
act on, and `overlap_rule_would_mislead` fires (and is counted in
observability) whenever the two disagree, so the discrepancy is visible
rather than silently resolved. Confirmation wanted that the paired test is
the intended one.

### D7 `[BLOCKER]` §2.6's 300 ms budget is not met at the book size §0 targets

Measured, not estimated. A `pre_trade_delta` request at 20 000 paths on the
build machine:

| positions | serial | 2 threads | vs budget |
|---|---|---|---|
| 2 | 336 ms | **236 ms** | 0.8x — met |
| 4 | 526 ms | 318 ms | 1.1x |
| 5 | 592 ms | 373 ms | 1.2x |
| 6 | 668 ms | 431 ms | 1.4x |
| 8 | — | 528 ms | 1.8x |

(Minimum of seven runs; the build container is shared and medians move by
30% between sweeps, which is why the minimum is quoted.)

§0 describes the target user as holding 5-8 simultaneous positions, so the
budget is missed by 2-3x for exactly the person the product is for. It is met
only for a two-asset universe.

The cost scales with the universe because both path generation and the
liquidation walk are per-asset. Three optimisations already went in and are
reflected in the numbers above (uniform-grid quantile map, branch-free map
application, vectorised float32 CVaR bootstrap), taking a two-asset request
from 426 ms to 242 ms. Reaching the budget at eight positions needs
something structural.

What is NOT available: cutting the path count. §2.5's interval rule outranks
the clock (D1), and the engine escalates paths rather than trimming them.

**Taken since:** thread-parallel path blocks, worth 1.4-1.65x. Threads
rather than processes because numpy releases the GIL on the operations that
dominate. The default is two workers, not the core count: on a contended
four-core box, four threads measured *slower* than two (648 ms against
418 ms at six positions), because oversubscription costs more than the
parallelism buys and the liquidation walk's per-step Python loop does not
parallelise at all. The optimum is hardware-dependent — tune
`RISK_ENGINE_WORKERS` on the deployment target rather than trusting a
number measured somewhere else.

Admissible only because it does not change the predicted distribution, only
which sample is drawn from it, so it is a PATCH release and the shadow
window survives it.

**Tried and reverted:** vectorising the funding AR(1) across assets. It
measured *slower* than the per-asset loop it replaced (24.6 ms against
19.6 ms per 5 000-path block at seven assets) because the per-step slice
`x[:, s, :]` is strided on both read and write, and that costs more than
the interpreted loop it removes. The finding is recorded in the code so
nobody re-attempts it.

Still open, none taken unilaterally because they trade against things the
specification cares about:
- production hardware, which this shared build container is not;
- float32 path generation, roughly a 2x saving, but it costs precision in a
  cumulative-sum over 24 steps and this is a risk engine. It *is*
  distribution-affecting, so it must land before the shadow clock starts or
  not at all;
- a smaller default path count with the interval rule still binding, which
  in practice means accepting wider intervals on high-probability books.

Reported rather than worked around, per §9. The engine counts every
over-budget request (`pre_trade_budget_exceeded`) and never trades the
interval guarantee for the clock.

### D5 `[RESOLVED]` Full-liquidation modelling is not conservative in every metric

§1.6 permits modelling full cross liquidation instead of partial, calling it
conservative. It is conservative for severity, and roughly neutral for "was
there a liquidation event in 24 h". It is *anti*-conservative for nothing,
but it does distort the terminal-equity distribution used by the PIT test in
§3.3: real accounts sometimes survive a partial liquidation and recover,
and the model assigns them zero. Because liquidations are rare this shifts a
small tail mass, but it is a known defect in the calibration target, not just
in the risk number, and is recorded in the journal's model version.

---

## E. Missing inputs (cannot be invented)

| # | Missing | Blocks |
|---|---|---|
| E1 | Builder address + its ≥100 USDC perp balance (§5.5) | Phase 4 |
| E2 | KMS/age key material and the agent-key encryption boundary (§5.4) | Phase 4 |
| E3 | Postgres DSN / deployment target | Shadow persistence at scale |
| E4 | The shadow address sampling frame (B4) | Phase 4 gate validity |
| E5 | Live API access from the build environment — `api.hyperliquid.xyz` is blocked at the proxy here (403), so every §5.1 parser is written against recorded fixtures and **has not been exercised against the live schema** | Phases 2-5; must be re-verified where the API is reachable |
| E6 | Historical liquidation frequencies by nominal leverage for Baseline A's `P(liq)` arm (B3) | Baseline A's probability output |
