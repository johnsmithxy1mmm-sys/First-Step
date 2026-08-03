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

**Reopened and re-resolved (2026-08-02 audit).** "The exact form is
implemented" was true only within a single maintenance tier, and this entry's
closed status manufactured false confidence that the short side was done.
Both closed forms held mmr fixed at one tier while §1.3's tables make mmr
non-decreasing in notional, so a SHORT — which walks into heavier tiers as it
loses — got a displayed liquidation price *beyond* the true one: measured at
126 bp of entry on a 1490 BTC short over the real 150M tier boundary, with
the stepped simulator liquidating inside the gap, and 21/21 shorts optimistic
in a randomized sweep (0/29 longs). The margin.py caveat at the time named
the LONG as the optimistic case — inverted. Both public functions now iterate
the solve to the tier the answer lands in (`_tier_consistent`); the same
sweep finds zero optimistic cases. The single-tier algebra above remains
correct as the inner step.

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

**`fit_copula_df` was implemented, tested, and used by no bundle** — both
builders in `service/state.py` passed a hardcoded `copula_df=4.0` while the
paragraph above described the IFM estimator in the present tense. Found
2026-07-31 while wiring A10, and **fixed 2026-07-31** as a separate change,
because the two are not the same kind of edit: wiring a check that can only
refuse changes no number the product outputs, whereas switching 4.0 for a
fitted value changes every number.

That makes it a distribution change, so `MODEL_VERSION` went to **0.3.0**
(MINOR) and the §3.3 counter resets. **The timing was the whole point.** The
counter had not started, so the change cost nothing; at any point after it
would have cost up to twenty-one days, and the alternative was spending the
window validating a magic constant nobody could source. On the fixture the
fitted value is **6.5** against the 4.0 it replaces — a materially thinner
joint tail, not a rounding difference.

Three things this does not change, each worth stating because the obvious
worry is wrong:

- **Ongoing refits are not version changes.** The fitted df moves as new
  returns arrive, exactly like the EWMA volatilities and marginal dfs beside
  it. `DISTRIBUTION_VERSION` keys on the specification, and the specification
  changed once, here. A parameter that tracked data *and* reset the counter
  would make a 21-day window unreachable by construction.
- **The §3.1 gate is unaffected.** `validation/benchmarks.py` builds its own
  `PathSpec` with explicit inputs — it tests the engine's mathematics against
  analytic properties, not the fitted model. Re-run under 0.3.0: **6/6**.
- **The grid bounds it.** The df is profiled over `COPULA_DF_GRID`
  (2.5–30.0), so it cannot run away. Landing on either end is recorded as a
  `copula_df_at_grid_edge` counter and a `df_clamps` entry rather than
  trusted: the floor means joint tails heavier than the grid can express, the
  ceiling means dependence indistinguishable from Gaussian, and both are the
  data outrunning the model family — the same reason §2.2's marginal clamps
  are logged.

**A9 and A10 are coupled, and the direction matters.** A fitted df is thinner-
tailed than 4.0 wherever the data say so, and a thinner model tail sits
further below the empirical one — which makes A10's assertion *more* likely
to fire. On the fixture that moved the worst pair's gap from −0.011 to
**+0.022** against a 0.05 margin: still passing, and now within 0.028 of
refusing. If the live build starts refusing to come up, that is the two
working as specified — a fitted copula that cannot represent real crypto
crashes is precisely what §2.3 exists to catch — and not a regression to
route around.

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

**The check ran nowhere until 2026-07-31.** `diagnose_tail_asymmetry` and
`assert_lower_tail_not_understated` were implemented, tested and called from
no shipped path — every test in `TestCopula` invoked them directly, so all of
them passed while production never did. Meanwhile `model/copula.py` described
the assertion in the present tense as something that "turns it into a hard
failure". That is the worse failure mode: an unguarded model that says so is
at least honest, whereas this one read as guarded. Same defect class as the
A8 caveats and the D2 entry above — prose asserting a mechanism nobody wired.

Now called from both bundle builders, so it runs at every startup on the
returns the copula was fitted from, and refuses the build when it fires.
Three decisions worth recording:

- **It refuses; there is no override flag.** A flag would be used the first
  time it was inconvenient, and "the risk model understates crashes" is not a
  condition anyone should be able to click past. §2.3 calls it blocking, §9
  requires stopping and reporting, §10 forbids simplifications that understate
  risk, and there is no skewed-t to fall back to. On the live path this can
  therefore prevent the service starting — that is the intended behaviour and
  not a bug to route around.
- **The measurement is recorded before the assertion**, so a refused build
  still leaves its numbers in `METRICS.tail_diagnostics`. Recording afterwards
  would leave exactly the build whose numbers matter most unmeasured.
- **The Gaussian baseline (`copula_df=None`) is exempt.** §3.2's baseline is
  deliberately naive; holding it to the t-copula's criterion would refuse the
  comparator for being what it is specified to be.

The fixture path runs it too, though a symmetric one-factor market cannot
trip it — a check that only runs where nobody exercises it offline is a check
that rots. Cost measured at 0.78 s for six pairs at 200k draws, once per
bundle build. `TestTheDiagnosticRunsOnTheShippedPath` asserts the wiring
rather than the statistic, and was verified to fail when the wiring is
removed; without that, the next refactor silently restores the original
defect and the suite stays green.

**It fired on live mainnet, 2026-08-01.** 90 days of hourly BTC/ETH/SOL:

| pair | empirical lower | model @ q=0.05 | gap |
|------|-----------------|----------------|-----|
| ETH/SOL | 0.750 | 0.644 | +0.106 |
| BTC/SOL | 0.704 | 0.637 | +0.067 |
| BTC/ETH | 0.694 | 0.637 | +0.057 |

The engine refused to build, which is the specified behaviour. What was NOT
specified is what it took down with it: `shadow/cli.py` builds through the
same `_build_live_bundle`, so the refusal also stopped the §3.3 shadow
harness — **the instrument that measures whether an unvalidated model is any
good.** Refusing to measure a model because it is unvalidated is circular,
and it is the one outcome that guarantees the defect is never characterised.

Resolved by scoping the refusal to the **consumer**, not by softening it:

- a path that shows a number to a person keeps refusing (`fatal=True`, the
  default, so a caller that does not think about it inherits strictness);
- a path that only records observations runs (`fatal=False`), still runs the
  check, still records it, still logs the refusal text at WARNING, and prints
  a defect note on every sweep.

`serving=False` appears exactly once, in `shadow/cli.py:_live_world`, and a
test pins it there — dropping it would silently restart the refusal and stall
the window again.

**The days this records are NOT §3.3 gate-days**, and the note says so on
every run. The remedy bumps `MODEL_VERSION` and resets the counter. They are
diagnostic evidence, collected to answer a question that blocks the remedy's
design — see A11.

### A11 `[BLOCKER]` The skewed-t remedy is not uniformly conservative

§2.3 prescribes a skewed-t when A10 fires, and the prescription is written as
though the fix were obviously in the safe direction. An adversarial design
review on 2026-08-01 (three independent designs, four refutation passes, all
findings measured against this tree) established that it is not, and produced
four blocking findings. They are recorded here because they must be answered
BEFORE the remedy is built, not discovered inside it.

1. **The sign of the error depends on the shape of the book.** `_any_liq`
   (`sim/engine.py:610`) is a **union** over positions, and `Position.size`
   is signed (`domain/types.py:247`). Heavier joint downside does not move a
   hedged or mixed book's liquidation probability the way it moves a
   long-only one — for a signed combination, co-movement cancels rather than
   accumulates. So "tail-heavier is safer" is false as a blanket claim, and
   §10 cannot be argued per-parameter; it has to be argued per-output. This
   is the A1 situation again: where no uniformly conservative option exists,
   §10 is satisfied by disclosure.

2. **A shared skew silently overwrites the fitted correlation.** With one
   mixing variable `W` and a shared `γ`, the term `γW` is a common component:
   realised dependence is no longer the IFM/EWMA estimate, and there is a
   hard floor `ρ_eff ≥ k(γ, ν)`. Fitting `γ` with the correlation matrix held
   fixed is therefore not well-posed. Either fit `(ρ, γ)` jointly or invert
   the distortion (`ρ* = (ρ − k)/(1 − k)`), and refuse — not clamp — when
   `ρ < k`.

3. **An unconditional conservatism margin has no null.** A `+1·SE` margin on
   the tail-dependence target installs a spurious `γ` on data with zero true
   skew in ~99% of replications. The margin must be conditional on first
   rejecting symmetry, which is what §10's "a fit that is *uncertain* must
   err toward a heavier lower tail" actually says.

4. **λ_U is not monotone in γ**, so a two-sided absolute tail criterion is
   unsatisfiable by any admissible member of the family (§10 permits only
   `γ ≤ 0`). A guard no reachable model can pass is the A10 defect inverted.

Latency is a separate constraint on the remedy and is already tight:
`generate_log_returns` at 20 000 × 24 × 8 measures **301 ms**, the whole of
§2.6's budget, with 221 ms of it in the quantile maps. A skew-t marginal is
not odd, so `QuantileMap`'s half-table, `np.abs` and `copysign`
(`sim/quantile_map.py:92-94, 122, 148`) are all invalid for it, and the two
tails have different polynomial indices (`ν/2` toward the skew, `ν` away —
confirmed by Hill estimation), so a single extrapolation slope reused across
both would understate the far lower tail by a factor of two.

What the shadow window under A10's recording mode is for: measuring the
magnitude and the direction of (1) on real books.

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

**Second live finding, same day, same account: `spotTransfer`.**

```json
{"type": "spotTransfer", "token": "UFART", "amount": "20.0",
 "usdcValue": "4.9884", "user": "0x2000...010d", "destination": "0xd475...", ...}
```

An airdrop landing in a spot wallet. It was filed under `EXTERNAL_FLOW_SIGNS`
as directional-needs-`toPerp`; the record carries no `toPerp`, no `sourceDex`,
and nothing else naming the perp account, so B2 refused it.

It is a **non-flow**, and the reason is worth stating because it is the
general rule the whole table should be read against: *what matters is whether
a record moves the quantity the model predicts.* `Book.equity` is cross
collateral plus the isolated pockets — the perp account. A spot balance is not
in it. Twenty UFART arriving in a spot wallet changes nothing being forecast,
so scoring $4.99 as external flow would corrupt the correction exactly as
counting a spot-to-spot `send` would.

That classification is an inference from a type name plus one record, so it is
guarded rather than trusted: `_assert_no_perp_leg` refuses any `spotTransfer`
or `spotGenesis` carrying `toPerp`, `sourceDex` or `destinationDex`. The guard
is scoped to `SPOT_ONLY_NON_FLOW_TYPES` rather than every non-flow, because
`liquidation` is also a non-flow and for a completely different reason — it is
a perp event the model *predicts*. Asserting a liquidation never names the
perp account would refuse correct records.

**The harness itself was costing a round trip per type.** `check_external_flow`
stopped at the first unreadable record, so `send` and `spotTransfer` surfaced
one per run — each needing a fix, a push, a pull and a re-run to reach the
next. Ledger delta types are a long tail and that is the slowest possible way
to enumerate them. It now probes each type on its own records and reports
**every** unreadable one in a single pass, with the failing record (not merely
any record of that type) attached as evidence. `net_external_flow` still
raises on the first refusal, which is correct for a resolver: it must not
proceed on a partial read. The harness has the opposite job.

This is the argument for running `verify --address` **before** starting the
§3.3 clock rather than during it. Encountered live, either type would have
surfaced as a per-row resolver failure classified TRANSIENT, retried forever,
and shown up only as a repeated traceback in a container log — with the
21-day gate quietly never advancing.

**Two questions the 2026-07-31 audit raised and deliberately did NOT fix**,
because both would change money math on a guess — the exact error class this
module refuses. Each names the record that settles it:

- **F-9: does `withdraw.usdc` include the withdrawal fee?** Hyperliquid
  charges ~$1 per withdrawal. If the record's `usdc` is the gross amount and
  the fee rides in a separate field, the correction understates every
  withdrawal's outflow by the fee — small, but systematically one-signed
  across a window. If `usdc` is already net, adding a fee would double-count
  it. **Settled by:** one live `withdraw` record printed in full (any
  `--report` from `verify --address` on an account that withdrew; the
  record is in `evidence.examples` only when a type fails, so dump the raw
  `userNonFundingLedgerUpdates` response and read the fields). Whichever
  reading is true, the other is the bug.
- **F-10: what does `subAccountTransfer` look like in a MASTER account's
  ledger?** The classifier signs it by which side the queried account was on
  and raises when it is neither. A transfer between two of a master's
  sub-accounts, if it appears in the master's own ledger naming only the
  subs, would raise on every resolution of that master — a permanent,
  per-address resolver failure. The 30-address frame sweep passed, so no
  sampled account hit it; masters with active sub-accounts remain untested.
  **Settled by:** the ledger of one master account whose subs transferred
  between themselves inside the window. If the shape appears, the likely
  correct reading is "neither side is this account's perp → 0.0 flow", but
  that is to be confirmed from the record, not assumed.

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
- **One sweep may not cover a long list.** C6's arithmetic still applies, and
  this entry got it wrong by 16× until 2026-07-31 — it said "over two
  minutes", which is not a conservative round-down of the real figure but a
  different number entirely. From the shipped constants
  (`INFO_REQUEST_WEIGHT = 20`, `WEIGHT_BUDGET_PER_MINUTE = 1200`,
  `SHADOW_RESERVED_FRACTION = 0.75`, so 300 weight/min for the sweep):

  | list | weight | at the sweep's 25% share | at the whole budget |
  |------|--------|--------------------------|---------------------|
  | 200  | 4 000  | **13.3 min**             | 3.3 min             |
  | 500  | 10 000 | **33.3 min**             | 8.3 min             |

  C6's own text quotes the right-hand column (8.3 minutes of the *entire*
  budget); this entry quoted neither. Thirty-three minutes is a different
  operational fact from two: it is most of an hour of continuous sweeping for
  one daily snapshot, it constrains how a cron may be scheduled, and it is
  the reason `--target` defaulting to 500 has a cost worth stating. Read off
  the left-hand column, because §5.3 reserves the other three quarters for
  live users and the sweep never gets them.

  **This table described a sweep the code did not implement, until
  2026-08-01.** The last line here used to read "`ShadowCron` reports
  truncation rather than waiting" — and truncating is exactly what made the
  arithmetic above fiction. A sliding minute at 300 weight buys **15
  addresses**, and the loop then broke. §3.3 wants 200 a *day*, so the gate
  was unreachable by construction; the first live sweep wrote **6 of 515**
  and stopped. The 33 minutes was always the cost of the sweep §3.3 needs,
  and the code gave up 32 of them in.

  `ShadowCron` now sleeps until the window refills, bounded by
  `MAX_SWEEP_SECONDS` (90 min) so a pathological list cannot pin a container
  until the next daily run collides with it. Truncation is still reported,
  and still means what it said — it is now reached at the ceiling rather than
  at the first refusal. Waiting was never the impolite option: §5.3 asks the
  sweep to yield to live users, and sleeping while their reserve refills *is*
  yielding, it simply also finishes the work.

**E4 is closed.** This paragraph used to end "E4 stays open until the shape
is confirmed against the live venue", which contradicted the bullet sixty
lines above it recording that exact confirmation, and the E4 row in §Register
repeated the stale side ("the feed's message shape has never been observed
from here"). Two live runs settled it, and the second is a full collection
rather than a probe:

| run | date | window | result |
|-----|------|--------|--------|
| `--dry-run`  | 2026-07-30 | 60 s   | 36 addresses / 30 records, field `users` |
| `--minutes 30 --out` | 2026-07-31 | 3.2 min | **515 addresses / 2 989 records / 1 110 frames**, 0 unparseable, 0 anomalies, stopped on the 500-address target |

The second run is the stronger evidence and not merely the larger one: 2 989
consecutive records parsed with **zero** anomalies, where an anomaly is
specifically a record that contradicts the shape after the first address has
been read. A guessed field name does not survive three thousand records; a
guessed URL, subscribe envelope or channel name does not produce any. Records
arrived for all three subscribed coins (BTC 1 057, ETH 1 731, SOL 201), so
the subscription is per-coin as assumed rather than silently global.

Two things this does **not** establish, both still true and both narrower
than the sentence they replace:

- **No frame in this repository is a capture.** Every trade frame in the
  tests and docstrings is stub-generated. The suite proves the parser matches
  the specification, never that the specification matches the venue — that
  link is carried by the runs above and by nothing in the tree. Anyone
  extending `TRADE_ADDRESS_FIELDS` on the strength of a passing suite is
  reading it wrong; re-run `--dry-run`.
- **The collector's own frame text still says NOT VERIFIED**, and correctly
  so: it describes the *runtime assertion* as a floor rather than a proof,
  because once the first address is read the shape is treated as confirmed
  and later contradictions are counted rather than aborting a window that
  already produced a sample. That is a statement about the collector's
  failure mode, not about whether the shape is known. It should stay.

What is settled: which frame to use, what to say about it, and that this
collector reads the real feed correctly.

### B5 `[RESOLVED]` §0.2 and §3.2 disagree about which baseline decides

§0.2 requires CRPS strictly better than *both* baselines. §3.2 says B is the
decisive one and that failing against B means simplifying. Both are computed
and reported separately; the champion/challenger machinery treats B as the
gate and A as context.

### B6 `[BLOCKER — disclosure]` The 3-asset universe silently narrows the cohort

Observed live 2026-08-01. The model universe is BTC/ETH/SOL (Phase 1). An
account holding any other coin — `KeyError: 'ATOM'`, `'HYPE'`, `'XMR'`,
`'XRP'`, `'BNB'` all appeared in one 15-address sample — fails
`_predict_all` when the engine cannot find a marginal for the off-universe
position, and the **whole address** is dropped as a per-address skip.

Dropping is the conservative direction and must stay: modelling only the
in-universe legs of a mixed book would ignore a position that contributes to
liquidation, which understates risk (§10). The problem is not the drop, it is
what the drop does to the **calibration cohort**. The surviving addresses are
"accounts holding ONLY BTC/ETH/SOL" — a strict, less-diversified subset of
the trades-feed frame (B4), and nothing in the published score would say so.
The sampling frame the score cites (B4) describes accounts that *traded*
those coins; the cohort that actually gets scored is the narrower set that
*holds only* them.

Two numbers make this concrete and worrying for §3.3. In the first live
(truncated) sample, 5 of 9 skips were off-universe holdings — a ~33% drop
rate on that account. Stacked with flat books, no-position and non-positive
equity, the first sample wrote 6 of 15 attempted (40%). At that rate 515
addresses yield ~206 usable, right at §3.3's 200-address floor. The true rate
across the full list is not yet known (the truncation bug that stopped that
sweep at 15 is fixed), but the gate's reachability now depends on it.

The *narrowing* is not fixed, because both options are real decisions rather
than corrections: expanding the universe is a model-scope change (more assets,
more correlation structure, a distribution bump), and dropping fewer addresses
would mean modelling partial books, which §10 forbids. That decision is still
open.

**The disclosure half is fixed (2026-08-01).** The drop reasons were printed
in the sweep report and nowhere durable, so a score computed three weeks later
had no record of how selective its cohort became. There is now a
`calibration_sweeps` table: one row per sweep with `attempted`, `written`,
`budget_exhausted` and a `{reason: count}` tally, written by `record_sweep`
straight from the sweep report. Because the reason strings are verbatim
(`"KeyError: 'ATOM'"` vs `"no open positions"`), the off-universe drop rate is
recoverable per day, so any published score can state its cohort selection
instead of implying it had none. The write is best-effort — a census failure
logs and is swallowed, because the census is provenance about the run, not the
predictions the run paid §5.3 weight to produce.

What remains is the model-scope decision itself, and the empirical input for
it: the true off-universe drop rate across all 515 addresses, which the first
(pacing-truncated) run could not measure and this table now will.

**The decision is now a config change (2026-08-02).** The universe is
`HL_UNIVERSE` (comma-separated, default `BTC,ETH,SOL`), read by
`_build_live_bundle` and passed through the compose stack, so widening it
does not require a code edit. BTC and ETH remain mandatory (§2.1). What
widening costs, per added coin: one 90-day candle snapshot plus one funding
history per rebuild (~40 weight against serving's 900/min — ten coins is
~80/min of the rebuild's budget), a row and column of correlation structure,
one more marginal fit, and a bigger draw per path-step. What it is NOT: a
free fix. It is a sampling-frame change AND a distribution change
(MODEL_VERSION MINOR, §3.3 counter reset), so it must happen BEFORE the
shadow clock starts or cost the accumulated days.

The decision procedure, concretely:
1. let the census accumulate a few days of full sweeps;
2. measure: `SELECT skipped_by_reason FROM calibration_sweeps ORDER BY
   swept_at DESC LIMIT 7;` — sum the `KeyError: '<COIN>'` counts by coin;
3. if the off-universe drop rate keeps `written` comfortably above §3.3's
   200/day floor, keep the universe and let the published score disclose the
   cohort selection this table records;
4. if it does not, set `HL_UNIVERSE` to cover the coins that actually appear
   (the tally names them, most-dropped first), bump MODEL_VERSION, record
   the new frame here, and start the clock then.

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

What remains is provenance, not calibration. **This no longer blocks the
pilot**, because a non-binding constraint cannot move the distribution the
shadow counter is accumulating against. It should still be settled before
Phase 4 touches real money.

**The check could not be closed at all until 2026-07-31.** It told an operator
to "confirm the value from protocol documentation and record it as the
`source` field" — and then built its own `FundingBounds.documented_default()`
and returned INCONCLUSIVE whenever nothing breached the cap. Following the
instruction exactly produced identical output. C1 had two reachable outcomes,
FAIL and INCONCLUSIVE, and no PASS; an assumption that cannot be closed is one
that gets ignored rather than resolved. Same defect class as `mc_non_convergence`
being structurally pinned at zero and A10's diagnostic never being called.

Closing it is now a real action:

```python
FundingBounds.from_protocol_source(0.04, "<url or file:line> (read <date>)")
```

`confirmed` is a field rather than a convention about the wording of `source`,
because prose cannot be checked and this is what the check turns on.
`documented_default()` cannot set it — a default that arrives pre-confirmed is
a default nobody ever confirms — and the citation must contain something
re-checkable, so "Hyperliquid docs" is refused. PASS requires **both** the
citation and a non-contradicting sample: a citation can be stale, and a quiet
month proves nothing about a bound nothing approached. Confirmation does not
excuse a breach — a confirmed-but-contradicted bound fails louder, which is
the correct ordering.

**Secondary corroboration, recorded as such (2026-07-31).** `api.hyperliquid.xyz`
and `hyperliquid.gitbook.io` are both 403 at this environment's proxy, so the
primary source could not be read from here. Several independent third-party
write-ups agree that the documented cap is **4%/hour**, matching
`HL_DOCUMENTED_HOURLY_CAP = 0.04`, and quote the docs as saying the formula
computes an 8-hour rate paid hourly at one eighth. **That is not the bar this
entry asks for and the flag stays off**: agreement among secondary sources is
how a value nobody verified becomes a value everybody trusts, which is exactly
what §1.5 exists to prevent.

One trap for whoever does read the primary source. The published formula
contains **two different clamps**, and recording the wrong one would be a
100× error in the wrong direction:

| clamp | value | what it bounds |
|-------|-------|----------------|
| overall funding cap | **4%/hour** | the realised rate — this is `cap_per_hour` |
| interest-rate term | ±0.0005 | the `clamp(interest − premium, …)` component *inside* the formula |

`cap_per_hour` is used to clip simulated funding paths, so it is the overall
cap. Recording ±0.0005 there would truncate reality at 1/80th of the true
bound — the §10-forbidden direction.

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

**Two 2026-08-02 audit findings sharpen this, one fixed, one open:**

- *Fixed:* `InfoClient.post` charged the budget once and then retried network
  failures up to 3x uncharged — under flakiness the sweep's promised 300/min
  was up to 900/min on the wire, and a venue 429 (an `HTTPError`, which is a
  `URLError` subclass) was itself retried on a 1–2s backoff. Every wire
  attempt now charges, so a retry that cannot be paid for waits for the
  window instead of being sent.
- *Fixed (2026-08-02, same day):* the per-PROCESS reserve. The snapshot job,
  the resolve job and the serving engine each built their own `WeightBudget`
  behind one egress IP: two shadow processes at 300/min each plus a serving
  default of 1200/min was a combined ceiling of 1500 against the venue's
  1200, and during the daily snapshot/resolve overlap the actual reserve was
  50%, not the promised 75%. Now:
    - the shadow jobs draw from ONE shared 300/min pool
      (`shadow/weight_ledger.py`), a Postgres-backed sliding window with a
      transaction-scoped advisory lock so two containers cannot both observe
      the same headroom and spend it. The pool is shared, not halved: 150/min
      each would put §3.3's floor of 200 addresses (8000 weight) at 53
      minutes, past the resolver's 50-minute ceiling. Verified against live
      PG16: two actors interleaving stop at 300 combined; a racing connection
      over a full window admits nothing. The bundle build inside `_live_world`
      charges the same pool. On a sqlite journal it falls back to the
      in-process window (single-machine development, stated in the log);
    - the serving engine takes the complement
      (`SERVING_RESERVED_FRACTION = 0.25` → 900/min), so serving plus shadow
      sums to exactly 1200. Its realised traffic is the five-minute rebuild
      (~140 weight), nothing on the request path.
  The §5.3 weight constants themselves (1200/min, 20/request) remain
  documentation-derived and still need the live confirmation this entry has
  always asked for.

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
the 2 pp rule alone requires **2 401** paths, so the binding constraint is
20 000 everywhere, not merely "usually".

That figure was wrong here until 2026-07-31: it read ~9 600, which is the
count for a **1 pp** half-width, not the 2 pp §2.5 actually specifies
(`1.96² × 0.25 / 0.01² = 9 604`; at 0.02 it is `2 401`, and
`paths_needed_for_half_width(0.5, 0.02)` returns exactly that). Both numbers
support the same conclusion, which is why the error survived — but they
support it by different margins, and the wrong one made the CI rule look
like a live constraint on path count when it is nowhere near binding. That
matters for reading §2.6: cutting paths under latency pressure is forbidden
by a rule with an 8× margin at the worst case, not a 2× one.

The same slack is why `mc_non_convergence` is structurally pinned at zero
(see the note under §2.5's counter): at 20 000 paths the worst-case Wilson
half-width is 0.0069, so `converged=False` is unreachable and the counter
cannot fire. It is a real invariant, not a working metric.

### D2 `[IMPLEMENTED — and the design needed one more thing]` `max_safe_size` may have no answer

**Built 2026-08-02** in `risk_engine/tools/max_safe_size.py`, with tests. It
is NOT wired to any serving path: §3.3 gates a recommendation exactly as it
gates execution, and a test asserts that `service/app.py` and
`service/state.py` do not reference it, so the gate does not depend on anyone
remembering.

**The design below was incomplete, and the missing half points the forbidden
way.** This entry named Monte Carlo noise as the reason bisection needs common
random numbers. That is true and is done — one draw per query, reused for
every candidate. But non-monotonicity here is not only a noise problem, it is
a STRUCTURAL one, and CRN does nothing about that: on a hedged book an order
opposite to the net exposure first REDUCES P(liq), passes through a minimum
near flat, and only then raises it. Measured on the fixture, a book short 6
BTC on $60k equity, scanning long orders: **3.82% → 0.00% → 10.00%**. The safe
set is an interval only when the order ADDS to the existing exposure.

Bisection assumes "safe below, unsafe above". Run across that minimum it can
return a size that is not safe — an understatement of risk, silently, which
§10 forbids. So the implementation scans a grid first (the shape is measured,
not assumed) and bisects ONLY inside a bracket already known to contain a
crossing, where bisection is valid. It also never offers a size past the first
breach even when a larger one measures safe again, because a recommendation
that needs the user to understand a U-shape is not a recommendation.

Two further things the build settled:
- **Three outcomes, not two.** `safe` / `none` / `unresolved`. The third is
  when the interval straddles the threshold everywhere the scan looked; it is
  "this many paths cannot tell you", not "there is no safe size", and
  collapsing either into 0 would say "trade nothing".
- **The scan ceiling has to cover flattening an opposing position.** Sizing it
  as `equity × leverage / price` truncated the answer on exactly the hedged
  books this tool is for, because an offsetting order releases margin rather
  than consuming it: measured 6 against a true 12 on the book above. Fixed,
  and when nothing on the scanned range breaches the result says so
  (`scan_bounded`) rather than presenting a scan artefact as a risk limit.

Cost note: every candidate is a full book walk, but they all ride ONE set of
price paths (`run_blocks` takes a sequence of books), so a query costs one
path generation rather than one per candidate. It is still far more work than
§4.2 and is deliberately not on §2.6's 300 ms clock.

The original entry follows.

---

§4.3 defines the answer as the largest size whose *upper CI bound* on
`P(liq)` stays under the threshold. If the smallest tradable increment
(`szDecimals`) already breaches the threshold, there is no safe size, and
the type must be able to say so rather than returning zero — zero and "no
safe size" are different statements to a user. Binary search over a noisy
objective also needs common random numbers across iterations to stay
monotone, so the search must fix its base randomness per query.

**This entry was tagged `[RESOLVED]` and written in the present tense —
"the return type carries that case explicitly", "the search fixes its base
randomness" — until 2026-07-31. There is no such return type and no such
search.** `grep -rn "def max_safe_size" .` matches nothing; the only mentions
in the codebase are a docstring cross-reference in `domain/types.py` and a
comment in `sim/engine.py` naming a future caller. `max_safe_size` is a
**Phase 4** deliverable and Phase 4 is gate-closed until shadow validation
passes (`risk_engine/README.md`'s phase table has always said so, which is
how the contradiction was found).

What is settled is the *design*: the decision above is the one to build to.
Nothing is settled about the code, because there is none. The tag is now
`[DECIDED, NOT IMPLEMENTED]` — the only new status in this document, added
because `[RESOLVED]` and `[BLOCKER]` cannot express "we know what to do and
have deliberately not done it yet", and collapsing that into `[RESOLVED]`
is what produced a documented safety property with no implementation behind
it.

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
| C4 | ~~Does the `webData3` subscription exist?~~ **RESOLVED 2026-07-31**: it does. `verify --probe-ws` opened `wss://api.hyperliquid.xyz/ws`, sent the subscribe, and the venue acknowledged it. Note what this does and does not establish — the subscription **exists**; nothing here says what it carries or that it is preferable to `webData2`, which the shard planner is specified against and continues to use. Existence was the question; suitability was never asked and is not answered. This check was UNCHECKABLE with the reason "this harness speaks only the Info POST API", which stopped being true when `collect_addresses` shipped and held a live socket to the same venue for 1 110 frames. | closed — planner unchanged |
| E4 | ~~The shadow address sampling frame~~ **RESOLVED 2026-07-31** (B4): the public trades feed, activity-selected; the leaderboard rejected because it ranks on the outcome being calibrated. Implemented by `risk_engine.market.collect_addresses`, which generates the `frame` text as well as the list. The shape is confirmed against the live venue — a 60 s `--dry-run` on 2026-07-30 (36 addresses / 30 records, field `users`), then a full 2026-07-31 collection: 515 addresses from 2 989 records over 1 110 frames, 0 unparseable, 0 anomalies, records arriving for all three subscribed coins. This row said "never been observed from here" until that date, contradicting B4's own text. No frame in this repository is a capture, which is a separate and still-true caveat. | closed |
### E6 `[FIXED — and it was silent on one side]` `leverage.rawUsd` is not the isolated pocket's collateral

Found 2026-08-03, from the first live shadow sweep. `parse_clearinghouse_state`
read `leverage.rawUsd` as "the collateral moved into the pocket", taken from
the documentation and never checked against a response (E5).

**What it actually is.** Measured on mainnet, four isolated positions at four
different leverages, exact to six decimals:

    rawUsd == marginUsed - positionValue

which for a long reduces to `collateral - size*entry`: the pocket's net USD
LEDGER CASH, negative because the position is bought partly with borrowed
dollars. `marginUsed` is the pocket's current EQUITY (collateral + uPnL).

**How that was settled rather than guessed.** Treating `marginUsed` as equity
and solving §1.1's isolated condition reproduces the venue's own
`liquidationPx` with an implied maintenance rate of exactly `0.5/maxLeverage`
on all four — 0.0125, 0.10, 0.05, 0.05. Four different leverages agreeing to
six decimals is not a coincidence, and it independently confirms §1.3's rate
formula against live data.

**The asymmetry is the dangerous part.** For a LONG, rawUsd is negative,
`Position` refuses it, and the whole account is dropped — loud, and visible
in the sweep log as `isolated position needs positive isolated_margin` (5
accounts in one sweep: BTC x2, SOL, TAO, HMSTR, PAXG). For a SHORT the same
field is `marginUsed + positionValue`, i.e. POSITIVE and roughly fifty times
the true collateral on a typical pocket — it would have parsed silently,
placed liquidation far away, and understated P(liq). §10 forbids that
direction, and the silent half is the one that would not have been noticed.

**Fix.** `isolated_margin = marginUsed - unrealizedPnl`, which is
direction-independent because it derives from `equity = collateral + uPnL`
rather than from any cash field. `rawUsd` is no longer read here.

**The check that was missing, now added.** The venue returns `liquidationPx`
on every isolated position — the answer to "did you understand these fields?"
handed over for free — and nothing compared against it. Parsing now
reconstructs it and records mismatches in `ISOLATED_LIQ_PX_MISMATCHES`
(recorded, not raised: a position in a higher margin tier legitimately breaks
the single-rate reconstruction, and dropping those would lose exactly the
large accounts §3.3 needs). On the live account: zero mismatches.

**`probe_isolated_funding` was checked and is NOT undermined.** It reads
`rawUsd` deliberately, and that is correct there: ledger cash is
mark-independent (`collateral - size*entry` has no mark term), which is
exactly the property its C5 attribution needs. Its fallback to `marginUsed`
WAS removed — `marginUsed` carries uPnL and moves with every tick, so a delta
over the probe's window would have been price drift attributed to funding.
C5's verdict stands.

**Still unmeasured: the SHORT case.** The four live positions were all longs.
`marginUsed == equity` is verified for longs and inferred for shorts from the
same balance-sheet identity; the fix does not depend on the rawUsd sign, but
"inferred" is not "measured". A short isolated position should be dumped and
checked against `liquidationPx` before this entry is considered closed on
both sides.


| E5 | ~~Live API access — every §5.1 parser written against fixtures, never exercised against the live schema~~ **RESOLVED 2026-07-29** by `python -m risk_engine.market.verify` run from a network where the API is reachable (it is still 403 at the build-environment proxy). All three parsers PASS on live mainnet: `meta` → 177 assets, 34 with multiple margin tiers; `candleSnapshot` → 720 hourly BTC returns, 0 gaps, hourly vol 0.00363; `clearinghouseState` → 10 positions, cross collateral $3,957,459.72. The documented response shapes were correct **for those three reads, and NOT for `clearinghouseState`'s isolated-margin fields** — see E6, opened 2026-08-03: `leverage.rawUsd` is not the pocket's collateral, and reading it as such dropped whole live accounts. E5's PASS was real but shallow: a parser that RETURNS a book is not a parser that returns the RIGHT book, and nothing compared the parsed numbers against the venue's own `liquidationPx`. | closed, but see E6 |
| E6 | Historical liquidation frequencies by nominal leverage for Baseline A's `P(liq)` arm (B3) | Baseline A's probability output |
