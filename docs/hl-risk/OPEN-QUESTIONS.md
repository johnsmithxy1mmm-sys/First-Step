# Specification review — contradictions, ambiguities, missing data

Written before any code, per §11.2. Each entry states what the specification
says, why it cannot be implemented as literally written, what the code does
instead, and — where the choice biases risk — in which direction.

`[BLOCKER]` = cannot be resolved from the specification, needs a decision or
external data before the phase that depends on it can close.
`[RESOLVED]` = a defensible reading exists, taken, and recorded here.

A label may carry an em-dash suffix, and the suffix is the load-bearing part:
it names what the label does *not* cover. `[BLOCKER — downgraded]` (C1) still
wants provenance but no longer gates anything; `[RESOLVED — at the 24h
default, ...]` (A8) settles one horizon and leaves the others carrying the
bias. An unqualified `[RESOLVED]` is a claim that nothing is outstanding, so it
is only correct where the entry needed a convention and not a measurement —
A1 is the clean case, and B4 reached it in two steps rather than one: the
frame was a decision, the feed shape was a measurement that came in
separately once an operator could reach the venue this environment's proxy
blocks. Where an entry's own text still asks for external data, the suffix
has to say so, otherwise the legend and the entry disagree and the reader
believes the legend.

---

## A. Mathematical contradictions

### A1 `[RESOLVED]` Zero price drift is unattainable with Student-t marginals

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

**Decided 2026-07-30: zero log-return drift** (`DriftConvention.ZERO_LOG_RETURN`,
which is what the engine already defaults to — no code change).

The reasoning, in the order it decides the question:

1. **There is no uniformly conservative option, so §10 does not pick one.**
   §10 forbids understating risk, and the reflex is therefore to take the
   harsher constant. But `-sigma^2/2` is harsher only for longs; for shorts it
   is *softer* by exactly the same amount, because the drift enters the two
   sides with opposite sign. Neither convention dominates the other across a
   book that holds both. Choosing `-sigma^2/2` would not be "the conservative
   choice", it would be moving the understatement from longs to shorts while
   claiming to have removed it. §10 is satisfied by disclosure here, which is
   what this entry is.
2. **The magnitude is an order of magnitude below the noise on `sigma`.** The
   gap between the two conventions is `sigma^2/2` per horizon: 8 bp over 24 h
   at a 4% daily vol. The EWMA/Ledoit-Wolf estimate of `sigma` itself carries
   several percent of relative error (A5), so the choice is invisible next to
   the input it modifies. Spending a one-sided distortion of the short book to
   buy an 8 bp shift in the long book is not a trade worth making.
3. **Symmetry is the property §2.4 is actually defending.** §2.4 exists to
   stop the model expressing a market view. `E[r] = 0` treats up and down
   identically under the stated convention; `-sigma^2/2` tilts every path set
   down, which is a directional statement dressed as a moment condition, and
   would have to be explained to a user as "we assume the price drifts down".

Both readings remain implemented and `DriftConvention` stays a parameter, so
`MEDIAN_PRESERVING_CONVEXITY` can be run as a challenger under §3.3 rather
than argued about. What is decided is the default the shipped numbers use.

**Why this had to be decided before the clock starts.** The drift convention
changes the predicted distribution, so changing it later is a MINOR/MAJOR
version bump, and the §3.3 shadow-day counter is keyed on the distribution
version (`risk_engine/version.py`): a bump resets the accumulated validation
window to zero and forces the new version through champion/challenger from
scratch. Deciding this after days had started accumulating would have thrown
every one of them away — the reason A1 gated the clock at all, despite being
worth 8 bp.

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

### A8 `[RESOLVED — at the 24h default; the week and portfolio_risk's 7-day arm still carry the bias]` Funding is simulated independently of price

Funding shocks are drawn independently of price shocks. In reality the
funding rate tracks the perp-spot premium, which correlates with recent
returns, so a falling market pushes funding negative. Under independence a
long in a crashing market keeps paying funding it would in fact be
receiving, and a short in a rallying market likewise. The sign of the net
bias differs by side and by horizon, so it cannot be waved through as
"conservative". Wanted: the empirical correlation between hourly funding and
hourly returns per asset, measured over the shadow window; if it is
material, the AR(1) needs a return-driven term.

**Decided 2026-07-30: accept the independence, and narrow the horizon it is
allowed to compound over.** `funding_drag`'s default `horizon_hours` changes
from 168 to 24 (`risk_engine/tools/funding_drag.py`).

What was accepted, stated plainly: the model draws funding independently of
price and will ship that way. The bias is not conservative — under
independence a long keeps paying through a crash where the real rate would
have turned negative (overstating its cost), and a short keeps receiving
through a rally (understating its cost) — so §10 is met by bounding and
disclosing it, not by claiming it errs the safe way.

The bound is the horizon. The error is an unmodelled correlation compounding
hour by hour, so it grows with the number of hours; over 24 h it sits below
the estimation error on the rate itself, over a week it does not. Narrowing
the default therefore removes the case where the unquantified part of the
number is large, without pretending the mechanism is modelled.

Why not just model it: a return-driven term in the AR(1) is a §2 change, so it
changes the predicted distribution, so it is a MINOR/MAJOR version bump — and
the §3.3 shadow-day counter is keyed on the distribution version
(`risk_engine/version.py`), so it resets the validation window to zero. Doing
that now would mean the clock never starts. Narrowing the default is the
honest interim position rather than a workaround: it makes the shipped default
the horizon the simplification survives, and discloses the approximation to
whoever receives the number — see "Where the disclosure actually reaches a
user" below, which corrects an earlier version of this sentence that claimed
it was said "on every result". It is not: `FundingDrag.caveats` discloses to a
*caller*, and no shipped path constructs one.

What is *not* fixed, and must not be read as fixed:

- A week is still reachable (`horizon_hours=WEEK_HOURS`) because §4.4 asks for
  the cost of a holding period and a week is a real one. Results there carry
  the full bias; the docstring and the per-result caveat both say the number
  is indicative rather than calibrated at that horizon.
- The default flip does **not** touch `portfolio_risk`. Its 7-day arm walks
  funding independently over 168 h inside the equity path, and that path
  decides `p_liq_7d`, which the API and the UI publish. Funding is second
  order there (≈0.8% of equity over a week on the fitted fixtures, against
  price moves an order of magnitude larger) and the bias is a fraction of
  that, but it is present and it is two-sided — conservative for longs,
  anti-conservative for shorts. `funding_cost` is published at 24 h only, so
  no *funding* figure reaches a user at a week.

Where the disclosure actually reaches a user (corrected 2026-07-30, after
adversarial review): **not** through `FundingDrag.caveats`. Nothing shipped
constructs a `FundingDrag` — the engine serves `/portfolio_risk` and
`/pre_trade_delta`, and the shadow cron calls the engine directly — so the
caveat tuple this entry relied on has no readers. The figure a user sees is
`funding_cost_24h`, from `portfolio_risk.result_24h.funding_cost`, and
`PortfolioRisk` carries no caveats field. The disclosure therefore lives in
three places that are on the shipped path: rendered copy on the funding panel
(`apps/web/app/page.tsx`), which states the 24 h horizon and that a week runs
roughly 7× it; doc comments on the field in `services/backend/src/risk/client.ts`
and `apps/web/lib/contract.ts`; and a test that the published figure really is
a 24 h walk (`risk_engine/tests/test_service.py`), because `portfolio_risk`
owns a `DAY_HOURS` of its own that `funding_drag`'s guard does not reach.

To revisit: measure the correlation between hourly funding and hourly returns
per asset over the shadow window — the measurement this entry originally asked
for, which does not need to precede the clock because it is a property of the
market, not of the model version. If it is material, add the return-driven
term to the AR(1), accept the MINOR bump and the counter reset, and widen the
default back. Until that measurement exists there is nothing to decide with,
which is why this is closed as a decision rather than left open as a gate.

**Why the label is qualified rather than a flat `[RESOLVED]`** (corrected
2026-07-30). The legend reserves `[BLOCKER]` for what needs a decision *or*
external data, and the paragraph immediately above still asks for external
data — the funding/return correlation has never been measured. The two
paragraphs before it name two reachable paths, `horizon_hours=WEEK_HOURS` and
`portfolio_risk`'s 7-day arm behind the published `p_liq_7d`, on which the
bias is not bounded at all, only disclosed. A flat `[RESOLVED]` would say
none of that is outstanding, and a reader who trusts the label over the body
would carry a week-horizon number as calibrated. The suffix is what the entry
actually establishes: the default horizon is decided and the simplification is
survivable there. Compare A1, which is flat `[RESOLVED]` correctly — it is a
choice of convention, nothing external is owed, and no path escapes the
decision.

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
to first order. A pilot can therefore run *before* the remaining
distribution-affecting questions are settled. As of 2026-07-30 there are none
left: A1 (zero log-return drift) and A8 (independence accepted, default
horizon narrowed to 24 h) were decided that day, C1/C2/C5 having already been
closed against live data — so the real counter can start, and this pilot no
longer has to precede it.

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

**Live finding, 2026-07-31 (mainnet).** `verify --address` on a real account
returned a `userNonFundingLedgerUpdates` delta type this build could not
classify: `send`. The record settled it, and it is worth recording *why* it
could not be filed under either existing table:

```json
{"type": "send", "sourceDex": "spot", "destinationDex": "spot",
 "token": "HYPE", "amount": "5.0", "usdcValue": "206.575", ...}
```

That instance is a HYPE transfer between two spot accounts — no perp equity
moved, so subtracting it would corrupt the very correction B2 exists to make.
But the same type with `sourceDex: "perp"` is $206 leaving the perp account,
which must be subtracted. **A type name is not sufficient to classify a
transfer**; `send` is routed by its `sourceDex`/`destinationDex` fields and by
which side of the transfer the queried account was on. Two consequences worth
carrying forward:

  - the amount lives in `usdcValue`, not `usdc`. The old code read `usdc`
    only and would have *raised* on this well-formed row. `amount` is a token
    quantity — treating 5.0 HYPE as $5 would have been a silent 40× error, so
    only USD-denominated fields are accepted;
  - the two legs are summed independently rather than chained, because an
    account can be on both sides. A perp→perp self-transfer moves no equity
    and nets to zero; chained, it would have scored a phantom $206 outflow.

An unrecognised `dex` value raises rather than defaulting to "not perp", on
the same reasoning as an unknown delta type: the default would hide a real
flow, which is the §10-forbidden direction.

This is the argument for running `verify --address` **before** starting the
§3.3 clock rather than during it. Encountered live, this type would have
surfaced as a per-row resolver failure classified TRANSIENT, retried forever,
and shown up only as a repeated traceback in a container log — with the
21-day gate quietly never advancing.

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

### B4 `[RESOLVED]` No documented way to obtain the shadow address list

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

**Decision, 2026-07-30: the frame is the public trades feed.** The
leaderboard is rejected, and the reason is not that its bias is larger but
that its bias is of a different kind. The leaderboard ranks on realised
performance, and realised performance is the variable the calibration score
measures. Sampling on the outcome makes the model appear mis-calibrated in
whichever direction the sample was skewed — calibrated against winners it
looks wrong on losers, and there is no correction downstream, because the
defect is in what the sample *is*, not in how it was scored. The trades feed
selects on trading activity instead. That is awkward rather than circular:
activity is not the quantity being measured, so the bias can be stated,
carried alongside the score and reasoned about by a reader.

Implemented by `risk_engine/market/collect_addresses.py`:

```bash
python -m risk_engine.market.collect_addresses --minutes 30 --out addresses.json
```

It subscribes to the trades WebSocket, harvests the accounts named on each
trade, folds them through `normalise_address`, and emits the JSON
`FileAddressSource` already reads — with the `frame` field generated rather
than left blank, because an operator handed a blank field writes "trades
feed", which satisfies the loader's check and states nothing. The generated
frame names the window, the coins, the activity bias in both directions, the
leaderboard argument above, and the B2 tension below. It refuses to write
fewer addresses than `ShadowProgress.required_addresses` without
`--allow-short`, and refuses an empty list unconditionally.

**What this does not fix, and must not be read as fixing:**

- **The bias is stated, not removed.** The gate is read off the
  book-unchanged cohort (B2), which discards any account whose book moved
  during the observation day — precisely the accounts an activity-selected
  frame contains. The usable sample therefore shrinks in a way correlated
  with how it was drawn, and the effective n behind the published score is
  materially smaller than the list length. This is the cost of the decision.
  It is written into every frame string the collector generates so it cannot
  be lost between the address list and the published number.
- **The message shape — verified 2026-07-30, on mainnet, from an operator's
  machine (this environment's proxy is still 403 on the venue; see E5).**
  `python -m risk_engine.market.collect_addresses --dry-run` against
  `wss://api.hyperliquid.xyz/ws`, 60 seconds, `trades` subscription: 36
  distinct addresses from 30 trade records, read from a field named `users`.
  Every load-bearing assumption held on the first live attempt — the URL, the
  subscribe envelope, the channel name, and the claim (this entry's own
  words) that "a public trade names its participants." That claim was the
  one part of B4 no reasoning could settle in advance; it needed the venue to
  answer, and now it has.

  The abort machinery that was written for the case this *didn't* hold stays
  in place regardless — a trade record carrying no address where one is
  expected aborts the run quoting the frame verbatim, an
  acknowledged-but-undelivered subscription aborts, silence on connect
  aborts, a connection that never opens aborts naming `--ws-url`, and no path
  reaches a written file with zero addresses. The failure it is built to
  prevent is the quiet one — a valid, empty, confidently framed list that
  loads cleanly, sweeps nothing, and surfaces three weeks later as a gate
  that never advanced. `--dry-run` is the cheap way to re-check this if the
  venue ever changes its message shape: sixty seconds against a real
  collection window's thirty minutes.

  **Those assertions are fatal only until the first address is read**, and
  that boundary is deliberate rather than a softening. Before an address has
  come out of one of `TRADE_ADDRESS_FIELDS`, an odd record or an error frame
  is evidence the assumed shape is wrong. After one has, the venue has
  demonstrated the shape, and aborting both discards a good sample and
  misdiagnoses it — a single non-trade record after 300 harvested addresses
  used to abort claiming this entry's assumption "did not hold", which is
  false, it had just held 300 times, and it would send an operator to edit
  `TRADE_ADDRESS_FIELDS` on evidence that says nothing of the kind. Past that
  point anomalies are counted, warned about on the progress stream while the
  operator can still kill the run, and published in both the generated frame
  and the `_provenance` block as trades the list does not contain. A count of
  zero and a count of nine thousand produce identical address lists
  otherwise.

  **No example frame in this repository is a capture, and that is still
  true after the live run.** Every trade frame, counter, timestamp and
  address in the collector's tests and docstrings is stub-generated — the
  suite drives a scripted socket and a fake clock. The 2026-07-30 dry run
  confirmed the shape but nothing from it was recorded here: what is written
  down is the summary line (36 addresses, 30 records, field `users`), not a
  frame. So the tests still prove only that the *parser* behaves as
  specified, never that the specification matches the venue. Anyone
  extending `TRADE_ADDRESS_FIELDS` on the strength of a passing suite is
  reading it wrong; re-run `--dry-run`.
- **More addresses still buy almost nothing.** B1 measured it: 200 → 500 per
  day moves the clustered half-width from 3.60 to 3.43 pp. Days are the
  lever. `--target` defaults to 500 for headroom against the accounts the
  sweep drops (flat books, non-positive equity, stale resolution), not
  because a bigger sample tightens the interval.
- **One sweep may not cover a long list.** C6's arithmetic still applies: at
  ~20 weight per address against the 25% of the 1200/minute budget the sweep
  is allowed, a 500-entry list takes over two minutes of continuous sweeping,
  and `ShadowCron` reports truncation rather than waiting.

E4 stays open until the shape is confirmed against the live venue: what is
settled is which frame to use and what to say about it, not that this
collector reads the real feed correctly.

### B5 `[RESOLVED]` §0.2 and §3.2 disagree about which baseline decides

§0.2 requires CRPS strictly better than *both* baselines. §3.2 says B is the
decisive one and that failing against B means simplifying. Both are computed
and reported separately; the champion/challenger machinery treats B as the
gate and A as context.

---

## C. Hyperliquid integration — facts that must be verified, not assumed

### C1 `[BLOCKER — downgraded]` The funding-rate protocol clamp (§1.5)

§1.5 correctly forbids inventing the bound, and the AR(1) is unusable without
it. The engine takes the clamp from a configuration record carrying a
`source` field, defaults to ±4%/hour attributed to the Hyperliquid docs, and
**refuses to run** if any observed historical funding rate exceeds the
configured clamp — a stale or wrong bound then fails loudly instead of
silently truncating reality.

**Measured against live mainnet 2026-07-29** (`market.verify`, 1500 hourly
observations over 30 days across BTC/ETH/SOL): no breach, and the worst
observed rate was `2.27e-05/h` on SOL — **0.06% of the configured 0.04/h
cap**, i.e. the clamp sits roughly 1760× above anything the market did in a
month.

Two consequences, and together they change what this question is worth:

- **The clamp is not a binding constraint in normal conditions.** It never
  activates, so the simulated funding distribution is determined entirely by
  the AR(1) fit, not by the bound. C1 was feared as a distribution-shaping
  parameter; measurement says it is a guard rail far off to the side.
- **Any error in it is very likely in the safe direction.** A cap set too
  high lets the model simulate funding paths more extreme than the protocol
  permits, which *overstates* cost of carry — permitted under §10. A cap too
  low would truncate reality and understate, which is forbidden; at 1760×
  headroom, too-low is not the plausible failure.

What remains is provenance, not calibration: the `source` field still says
"unverified against live API". Confirm 0.04/h from protocol documentation or
source and rewrite that string. **This no longer blocks the pilot**, because
a non-binding constraint cannot move the distribution the shadow counter is
accumulating against. It should still be settled before Phase 4 touches real
money.

### C2 `[RESOLVED — measured over 12 hours; the guard this entry claimed never existed]`

§1.4 asks whether the mark price a liquidation is judged against can be
treated as the trade price the engine is fed. Two separate things were wrong
here, and the correction matters more than the measurement.

**What this entry used to claim, and what is actually true.** It said the
approximation was "*not* silently adopted: the engine requires an explicit
`BasisModel`, and the only one available before measurement is
`UnmeasuredBasis`, which is flagged in every result it touches and counted in
observability." **No such class exists.** `grep -rn 'BasisModel\|UnmeasuredBasis'`
finds hits only in `market/verify.py` and its test — i.e. only in the text of
the checker that describes the guard, never in the engine. What the live path
actually does (`shadow/providers.py:210-227`) is fetch the last hourly candle
close per asset and hand it to the simulator, which checks the §1.1 margin
condition against it directly. The candle close is a *trade* price; §1.1's
condition is defined on the *mark* price. So the approximation is adopted
exactly as silently as this entry denied — the same defect class as A8's
caveat tuple hanging on an object nothing constructs: documentation
describing a safety mechanism that was never built.

**The measurement, which is what now justifies it.** Two runs of
`market.verify` against mainnet `metaAndAssetCtxs`:

| date | window | worst median &#124;basis&#124; | §1.4 threshold | margin |
|---|---|---|---|---|
| 2026-07-29 | 12 samples / 1 min | `2.63e-05` (ETH) | `9.06e-04` | 34× |
| 2026-07-30 | 720 samples / 12 h | `2.61e-05` (ETH) | `9.07e-04` | 35× |

The threshold is §1.4's own: 25% of a typical hourly move, from the measured
0.00363 hourly BTC vol. The 12-hour result is a window §1.4 would accept, and
it reproduces the 1-minute figure to two significant figures rather than
regressing toward the bound — the basis is small and stable, not small
because the first sample was lucky.

**Taken: the identity, with the margin as its justification.** At 35× headroom
a basis term would be modelling something two orders of magnitude below the
estimation error on `sigma` itself. Building the `BasisModel` machinery this
entry once described would add a parameter with nothing to fit.

**What remains inexact, stated rather than buried.** The measured quantity is
`|markPx - midPx| / midPx`, and the price actually fed to the engine is the
hourly candle *close* — a last-trade price, not the mid. Mid and last-trade
diverge by at most a spread on a liquid perp, so the measurement bounds the
quantity that matters closely but not exactly. The harness still reports
INCONCLUSIVE by construction: promotion is a judgement about window adequacy
and it should not grant that to itself. This entry is that judgement, made
explicitly and dated.

Re-run if the venue's fee or matching model changes:

```
python -m risk_engine.market.verify --address 0x... --samples 720 --interval-s 60
```

### C3 `[RESOLVED]` Builder fee units (§5.4)

`f: 20` is 20 *tenths of a basis point* = 2 bp = 0.02%, and `maxFeeRate`
0.03% is the approval ceiling above it. Consistent, but the unit is a 10x
footgun and the spec's own prose alternates between the two numbers when
describing the positioning; the UI copy should quote the charged 0.02%, not
the 0.03% ceiling.

### C4 `[BLOCKER — non-blocking in practice]` `webData3` (§5.2)

`webData2` is the documented subscription. I have no confirmation that
`webData3` exists. The shard planner is agnostic either way, so nothing
downstream waits on this.

A WebSocket question, which the Info-API harness cannot reach. Two minutes to
settle by hand:

```bash
pip install websockets
python -c "
import asyncio, json, websockets
async def main():
    async with websockets.connect('wss://api.hyperliquid.xyz/ws') as ws:
        await ws.send(json.dumps({'method':'subscribe','subscription':{'type':'webData3'}}))
        print(await asyncio.wait_for(ws.recv(), 10))
asyncio.run(main())"
```

An error response means it does not exist and §5.2 should say `webData2`.

### C5 `[RESOLVED]` Isolated-position funding — confirmed on live testnet

The model debits funding on an isolated position from that position's
isolated margin (which is what makes isolated liquidations independent, per
§1.1). This was the only remaining open question that could invalidate
**structure** rather than shift a number.

**Measured, 2026-07-30, Hyperliquid testnet.** One isolated BTC position
observed across an hourly funding tick:

```
[PASS] C5
  BTC: funding $-0.0621, the pocket absorbed 100% of it -> isolated
  cross account value moved $+0.00 over the window
```

The pocket's own collateral absorbed the entire payment and the cross
account value did not move at all. §1.1's independence claim holds and the
simulator needs no coupling term — the code is correct as written.

Bounds of the result, stated because a single clean observation is not a
protocol guarantee: one tick, one asset, on testnet, on an idle account. It
falsifies the dangerous hypothesis (cross-pool debit) rather than proving the
mechanism for all cases. The probe stays in the tree and is cheap to re-run
on mainnet against a larger position; a re-run is warranted if the venue ever
changes its margin accounting. The stated assumption — that the venue settles
isolated funding into pocket collateral (`rawUsd`) rather than into the
pocket's own unrealised PnL — was borne out: the pocket's `rawUsd` moved by
exactly the payment.

**Answerable read-only, no capital required.** The `verify` harness reports
it UNCHECKABLE because no single read distinguishes the two, and the original
note here assumed that meant opening a funded position. It does not:
`clearinghouseState` exposes `leverage.rawUsd` per isolated position — the
collateral in that pocket — and `userFunding` gives what the account actually
paid. Snapshot both sides of an hourly funding tick and ask whether the
pocket absorbed its own payment. The pocket is the only side worth reading:
its `rawUsd` is ledger collateral (funding and explicit transfers move it,
mark prices do not), while the cross account value drifts with every cross
position's uPnL and buries a funding-sized move within minutes — an audit
(P-2) showed a verdict that read the cross side could never return FAIL on a
real account. A pocket that provably paid nothing while `userFunding` shows
the account paid **is** the §1.1 violation: there is no third bucket.

```
python -m risk_engine.market.probe_isolated_funding --address 0x... --report c5.json
```

Needs an address holding at least one isolated position across the tick — any
address, since the data is public. It waits for the top of the hour plus a
settle grace, aborts if the book changed during the window (a trade moves
both balances for unrelated reasons), and reports INCONCLUSIVE when the
payment is too small to separate from rounding rather than resolving it in
whichever direction the arithmetic landed. Exit code 2 means the cross pool
absorbed it — §1.1 is false and the simulator needs a term it does not have.

Testnet works too (`--testnet`) if a suitable mainnet address is hard to
find; opening a minimal isolated position there costs nothing.

### C6 `[BLOCKER]` Shadow cron vs. rate limit (§3.3 vs §5.3)

500 addresses x ~20 weight per `clearinghouseState` is 10 000 weight, against
a 1200/minute budget: a full sweep costs 8.3 minutes of the *entire* budget,
before any candle or funding traffic, and §5.3 requires the cron to yield to
live users. Daily snapshots are feasible; the resolver doubles the traffic.
The cron is built with a weight-budget governor and an explicit low-priority
lane, but the address count and the live-user headroom are coupled and should
be sized against real traffic, not assumed.

### C7 `[BLOCKER — non-blocking in practice]` Is the Info API case-sensitive on `user`?

**Recorded 2026-07-30, after adversarial review, because it was recorded
nowhere.** Commit 667f539 opens by asserting a live venue finding as
established fact:

> The C5 probe returned HTTP 422 on an address pasted in EIP-55 checksummed
> form ... Lowercasing the same string made the call succeed against an
> unchanged account, so the venue's Info API is case-sensitive on the `user`
> field.

That is the only place in this repository where the finding appears, and a
commit message is not one of the places this project records live findings.
Every other one has a dated entry here with its numbers attached: E5's three
parsers on live mainnet, C1's 1500 hourly funding observations, C2's 12 basis
samples, C5's funding tick on testnet. Case sensitivity — the observation that
motivated a change to two boundaries across thirteen files — left this
document silent on the subject. A claim that lives only in pushed history is a claim
nobody can check or correct, which is why it is being written down here
instead of by rewriting the commit.

**What is known, and what is not.** Known: a 422 was seen; the string was the
checksummed spelling; lowercasing it produced a successful response against an
account that had not changed in between. The sequence is consistent with the
conclusion drawn from it. Not known: the exact command line that produced the
422 was never captured, so the failure cannot be replayed, and the ordinary
alternatives are not excluded — a different defect in that one invocation, or
a transient venue-side response that the second attempt happened to clear. One
unrepeated observation with no recorded command is evidence; the commit stated
it as a protocol property. The distinction matters because a protocol property
is something later work is entitled to build on, and this is not yet that. The
API is 403 at this environment's proxy (E5), so it cannot be re-run from here.

To settle it, from a network where the API is reachable, and record the result
in this file rather than in a commit message:

```bash
A=0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed   # any real account, checksummed
for u in "$A" "$(printf %s "$A" | tr 'A-F' 'a-f')"; do
  curl -s -o /dev/null -w "%{http_code}  $u\n" -X POST https://api.hyperliquid.xyz/info \
    -H 'Content-Type: application/json' \
    -d "{\"type\":\"clearinghouseState\",\"user\":\"$u\"}"
done
```

Two different status codes confirm case sensitivity; two 200s refute it and
this entry should then say so.

**The normalisation fix does not rest on this observation and must not be read
as doing so.** If the venue turns out to be perfectly case-insensitive,
`normalise_address` stays exactly as it is, because the two failures it
defends against are properties of this codebase, verified offline against
stubs rather than against the venue:

- **§5.1's silent empty state.** The venue answers an address it does not
  recognise with a well-formed *empty* clearinghouse state, never an error. A
  wrong address therefore does not fail — it reads downstream as "this account
  holds no positions", which a risk tool renders as no risk. Case sensitivity
  would only change *which* wrong strings land there, not what happens when
  one does.
- **Journal identity.** `address` is TEXT with no COLLATE in `schema.sql`, and
  neither Postgres's default collation nor SQLite's BINARY folds case, so the
  UNIQUE constraint cannot see two spellings of one account as a duplicate.
  `progress()` counts `DISTINCT address` towards §3.3's 200-account gate. That
  is arithmetic over storage this repository owns outright; the venue has no
  say in it.

The honest ordering is that the fix was worth making either way, and the 422
is what prompted somebody to look.

**What the fix cannot do: heal a journal that already holds a split account.**
Normalising at the write makes every future row canonical and repairs nothing
already written, and the two mix badly. Reproduced against an in-memory
journal — one real account written checksummed the way a pre-fix writer wrote
it, then lowercase the way the post-fix writer writes it, both resolved:

```
shadow gate CLOSED for 0.2: 2/21 days, 2/200 addresses, 2 resolved observations
```

`distinct_addresses == 2` for **one** account: precisely the §3.3 gate
inflation the fix is described as preventing, reached from the other side.

**The blast radius is currently empty, and that emptiness is the precondition
the fix rests on.** The shadow counter has not started, no journal database
exists in this repository or on any deployment target, and
`deploy/docker-compose.yml` keeps the shadow jobs behind a profile so `up`
cannot create one as a side effect. There is therefore no split row anywhere
to find — which is why there is **no detection query and no backfill**, stated
here rather than left for someone to discover by looking for them. That is a
statement about today, not a property of the design. It expires the first time
a sweep writes, and it would already be false for a journal restored from
before 2026-07-30 or for any future writer that bypasses `record_prediction`.
If either happens, the first thing to write is the grouping the identity
column cannot express — group on `LOWER(address)` and report every group whose
`COUNT(DISTINCT address)` exceeds 1 — followed by a decision about rows that
cannot be edited (§3.4, audit A-09). That is a migration question, not a bug
fix, and it is cheaper to never need it than to answer it.

**One further correction to the same commit.** Its closing paragraph justifies
folding rather than verifying the checksum as avoiding "either rejecting the
perfectly legal all-lowercase form or drag[ging] keccak into a module that
needs none". The first horn is false: EIP-55 is a *case* pattern, so an
all-lowercase or all-uppercase address carries no checksum information and is
accepted unverified by every implementation — only mixed-case strings are
checkable, and verification would have rejected no legal spelling. It would
have caught a mistyped capital on exactly the checksummed paste path the
commit cites as its motivation, for free. The decision to skip it is still
right, on its real cost: EIP-55 is defined over keccak-256, `hashlib` has no
keccak, and `hashlib.sha3_256` is NIST SHA-3 rather than keccak-256 (different
domain-separation padding, unrelated digest), so the routes are a third-party
dependency in a numpy+scipy package or a hand-rolled Keccak permutation inside
the module that defines account identity. Corrected in `normalise_address`'s
docstring and in `test_address.py`, where the assertion that a broken checksum
is accepted now records a known gap instead of a virtue. The gap that remains
either way: `0x...beaed` mistyped as `0x...beaec` is 40 valid hex digits and a
different real account, caught by nothing, returning the §5.1 empty state that
reads as "no positions" — and a lowercase string carries no checksum, so
keccak would not have caught that one either.

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
| E4 | ~~The shadow address sampling frame~~ **DECIDED 2026-07-30** (B4): the public trades feed, activity-selected; the leaderboard rejected because it ranks on the outcome being calibrated. Implemented by `risk_engine.market.collect_addresses`, which generates the `frame` text as well as the list. Still open: the feed's message shape has never been observed from here (the API is 403 at this proxy), so the collector asserts it at runtime and aborts rather than writing an empty list. | Phase 4 gate validity — frame settled, one live run needed to confirm the shape |
| E5 | ~~Live API access — every §5.1 parser written against fixtures, never exercised against the live schema~~ **RESOLVED 2026-07-29** by `python -m risk_engine.market.verify` run from a network where the API is reachable (it is still 403 at the build-environment proxy). All three parsers PASS on live mainnet: `meta` → 177 assets, 34 with multiple margin tiers; `candleSnapshot` → 720 hourly BTC returns, 0 gaps, hourly vol 0.00363; `clearinghouseState` → 10 positions, cross collateral $3,957,459.72. The documented response shapes were correct. | closed |
| E6 | Historical liquidation frequencies by nominal leverage for Baseline A's `P(liq)` arm (B3) | Baseline A's probability output |
