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
independent unit). The gate should be read off the clustered version. With
21 days as the effective sample, the achievable precision on a 5% breach rate
is roughly ±4 pp — which means **the §0.3 criterion as written is not
reachable in 21 days**, and the shadow window needs to be far longer, or the
criterion restated in terms of days rather than observations. This needs a
decision before Phase 4's gate can be evaluated honestly.

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
