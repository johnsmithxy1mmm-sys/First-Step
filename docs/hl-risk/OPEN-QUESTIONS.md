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

### A6 `[RESOLVED]` §3.1.2's 0.3 pp tolerance is below Monte Carlo noise

At 20 000 paths the standard error on a probability near 10% is 0.21 pp, so
the *difference* of two independent estimates has a standard error of 0.30 pp
— the tolerance is one sigma, and the benchmark would fail roughly a third of
the time on a correct engine. The benchmarks therefore run with common random
numbers across the compared configurations and an elevated path count, which
makes the comparison near-deterministic. Benchmark §3.1.5 (monotonicity in
leverage) is likewise only meaningful under common random numbers; evaluated
on independent path sets, "strictly increasing" on a 20-point grid is a
coin-flip proposition regardless of correctness.

**Not a live decision point.** Unlike D4/D6, nothing here turns on what the
specification *meant* — CRN is the standard fix for exactly this failure
mode (comparing two noisy estimates of a shared, correlated quantity) and
the alternative is a flaky gate. A7's own measurement leans on this directly:
`benchmark_5_leverage_monotonicity`'s cross-slider invariance
(0.161125 at 5x/10x/25x, to six decimals) is only a meaningful zero *because*
the six evaluations share paths. Left labelled `[BLOCKER]` past that point
it would have been the same defect A7 and C1 were both opened to fix —
a question with a settled answer that the label still asked the reader to
re-litigate.

### A7 `[DECIDED — the reading is forced, and now pinned]` §3.1.5 does not say which leverage

**Decided 2026-08-03, by measurement rather than preference.** The shipped
reading stands, because it is the only one under which the benchmark tests
the ENGINE instead of the specification's wording.

Measured on the benchmark's own fixture (40 000 paths, 1% hourly vol,
24 steps), holding position size fixed and moving only the leverage slider on
a cross book:

| set leverage | P(liq) |
|---|---|
| 5x | 0.161125 |
| 10x | 0.161125 |
| 25x | 0.161125 |

Identical to six decimal places — exactly what §1.2 says, and exactly why
"P(liq) strictly increases in leverage" cannot be tested against the slider
on a cross book: the true answer is a flat line, so a strictly-increasing
assertion would fail on CORRECT behaviour. Varying effective leverage
(notional at fixed collateral) over the same grid moves it 0.02202 -> 0.62790.

So: cross grids over effective leverage, isolated grids over set `L` (which
genuinely does move an isolated pocket, 0.02167 -> 0.62915). Both must be
strictly increasing.

**And the invariance is now asserted end to end.** It was already checked at
the unit level — `test_liquidation.py::test_cross_liquidation_ignores_the_leverage_slider`
pins it on the closed-form `liquidation_price`. What was NOT checked is the
same property through the full Monte Carlo, which is the path §3.1.5 actually
grades: the closed form could keep the invariance while the simulator lost
it, and the benchmark would still pass on its two monotone grids. It now
fails if the cross slider moves simulated P(liq) at all. Measured invariant
to six decimal places (0.161125 at 5x, 10x and 25x).

What remains yours: whether the specification INTENDED effective leverage.
The measurement settles what is testable, not what was meant — though the two
coincide here, since the other reading is untestable.

The original entry follows.

---

### A7 (original) `[BLOCKER]` §3.1.5 does not say which leverage

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

**Measured on live mainnet, 2026-08-03 — and it is not always an asymmetry at
all.** The first live shadow sweep fired A10 on all three pairs over the
90-day window (2160 hourly observations, so n=108 exceedances at q=0.05):

| pair | lower | upper | model@q | excess | sigmas | asymmetry |
|---|---|---|---|---|---|---|
| ETH/SOL | 0.750 | 0.685 | 0.641 | +0.109 | **+2.6** | +0.065 |
| BTC/ETH | 0.694 | **0.731** | 0.633 | +0.061 | +1.4 | **−0.037** |
| BTC/SOL | 0.694 | 0.667 | 0.634 | +0.060 | +1.4 | +0.027 |

Two things follow, and the second is new.

**The gate fires readily on noise, by design.** The SE of the empirical
proportion at n=108 is ~0.042, so the 0.05 margin is 1.13 SE. Only ETH/SOL is
a real signal at 2.6 sigma; the other two sit at 1.4. That is the intended
direction — §10 makes a false alarm cheaper than a miss — but a refusal that
does not say which pairs are signal invites someone to dismiss all three. The
diagnostic now reports the sigma alongside each pair.

**BTC/ETH is not an asymmetry: its UPPER tail is heavier than its lower**
(0.731 against 0.694), while still failing the gate because the model sits
under BOTH. A skewed-t buys one tail at the other's expense, so applying it
there would fit the lower tail by making the upper worse. That pair is
evidence about the copula df or the elliptical family, not about skew — the
fitted df is too Gaussian to reach the observed tail dependence in either
direction. `assert_lower_tail_not_understated` now separates the two cases
and names the pairs where skew is the wrong remedy, because a run that read
the old message literally would have reached for it.

So the remedy is not one remedy. Before anything is built, the shadow window
has to say which of the two shapes dominates on real books — and this first
sweep already shows both present at once.

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

   **Derived and measured 2026-08-04, and it is exact, not approximate.** For
   the GH skew-t (`X = γW + √W·Z`, `W ~ IG(ν/2, ν/2)` shared — the family
   whose ν/2-vs-ν tail indices the latency note below confirms by Hill
   estimation): `Cov = γ²·Var(W) + ρ·E[W]`, `Var = γ²·Var(W) + E[W]`, so

       ρ_eff = k + ρ·(1 − k),   k = γ²v / (γ²v + m),
       m = E[W] = ν/(ν−2),      v = Var(W) = 2ν² / ((ν−2)²(ν−4))

   — LINEAR in ρ, floor exactly `k`, inversion exactly the formula above, and
   `ρ* ≥ −1` bounds representable targets at `ρ ≥ 2k−1`. Verified against
   simulation over ν ∈ {6.5, 8, 12} × γ ∈ {−0.3, −0.8, −1.5} × ρ ∈
   {0, 0.5, 0.9}: worst |closed − simulated| = 0.012 at 2M draws per cell.
   The floor is not small at plausible parameters — `k(−0.8, 6.5) = 0.425`,
   `k(−0.8, 5.0) = 0.681` — so an unconditioned fit would replace most of the
   estimated dependence structure with an artefact of the skew.

   **Sharper than the finding as written: below ν = 4 the question dissolves.**
   `Var(W)` exists only for ν > 4, so with `γ ≠ 0` and ν ≤ 4 the marginal
   variance is INFINITE — EWMA vol scaling and correlation targeting are not
   distorted there, they are undefined. The fitted df on a young asset was
   3.5 (this entry's own record), so that is the operating point, not a corner
   case. Measured to show it is visible: sample correlation under
   (ν=3.5, γ=−0.8) across 40 independent windows of 2160 observations spans
   0.878–0.986 — an estimator of a quantity that does not exist, its spread
   driven by single extreme draws of `W` and not shrinking with n. Any
   skew-t design must therefore also constrain ν > 4 (with margin), which the
   family's own fit on this venue's young assets already violates.

3. **An unconditional conservatism margin has no null.** A `+1·SE` margin on
   the tail-dependence target installs a spurious `γ` on data with zero true
   skew. The margin must be conditional on first rejecting symmetry, which is
   what §10's "a fit that is *uncertain* must err toward a heavier lower tail"
   actually says.

   **Measured 2026-08-04, and the rate stated here was wrong.** Simulated on
   symmetric t-copula data at the live window's size (ν=6.5, ρ=0.80, n=2160
   hourly, q=0.05, 4000 replications; the model's λ at that finite threshold
   is 0.5435, ~106 exceedances per tail):

   | rule | demands skew on ZERO-skew data |
   |---|---|
   | unconditional `+1·SE` | **83.9%** |
   | conditional (margin only after symmetry is rejected) | **2.5%** |

   The entry said ~99%; it is 84%. That is not a simulation artefact — it is
   Φ(1) = 84.1% almost exactly, which is what the rule has to produce: the
   empirical estimate is centred on the model's value under symmetry, so
   demanding the model reach `empirical + 1·SE` fails whenever the estimate
   lands above `model − 1·SE`. Being able to derive the number is worth more
   than the number: it means the defect is structural in the rule and not a
   property of this window's size.

   The finding's substance is unaffected and its remedy is confirmed. 84%
   false-positive rate against a nominal 5% is still a criterion with no null,
   and making the margin conditional restores one exactly — 2.5% is the
   one-sided tail of the 95% two-sided symmetry test, i.e. the rate that test
   is supposed to have.

4. **λ_U is not monotone in γ**, so a two-sided absolute tail criterion is
   unsatisfiable by any admissible member of the family (§10 permits only
   `γ ≤ 0`). A guard no reachable model can pass is the A10 defect inverted.

   **Measured 2026-08-04, at the shipped diagnostic's own threshold** (q=0.05,
   ν=6.5, ρ=0.8, 2M draws per γ, rank pseudo-observations exactly as the
   fitting pipeline builds them):

   | γ | λ_L | λ_U |
   |---|---|---|
   | 0.0 | 0.540 | 0.541 |
   | −0.2 | 0.577 | 0.517 |
   | −0.6 | 0.660 | 0.499 |
   | −0.9 | 0.719 | **0.498** |
   | −1.2 | 0.762 | 0.510 |
   | −1.5 | 0.800 | 0.522 |

   λ_U falls to γ ≈ −0.9 and then RISES — non-monotone as claimed (the far
   skew drags the common `W` so hard that big mixing draws lift both assets'
   ranks together even against the skew). Two consequences, both now
   concrete rather than argued:

   - the whole reachable range of λ_U over admissible γ is **[0.498, 0.541]**
     at these (ν, ρ). The live BTC/ETH pair's empirical upper tail is
     **0.741** — far outside it. So for the one pair the 2026-08-03 mainnet
     reading flags as under-modelled in BOTH tails, no admissible γ exists at
     the fitted (ν, ρ), and a two-sided absolute criterion is not merely
     hard to satisfy, it is empty. The remedy for that pair has to move ν or
     the family, exactly as the top of this entry already concluded from the
     shape of the reading;
   - because λ_U is non-monotone, even a target INSIDE the range is reached
     at two different γ values with materially different λ_L (e.g. λ_U ≈
     0.51 at γ = −0.2 and again at γ = −1.2, where λ_L is 0.577 vs 0.762).
     A two-sided system therefore has zero or two solutions, never reliably
     one, and any fitting procedure built on it must state which branch it
     takes and why — or use a one-sided lower-tail criterion with λ_U as a
     reported diagnostic, not a constraint.

**Where A11 stands after the 2026-08-04 measurements — all four findings now
carry evidence, and together they answer the question §2.3 left open.** The
prescription "fit a skewed-t when A10 fires" is refuted for this venue's
measured data, on three independent grounds: (1) heavier joint tails are not
the conservative direction per-output (zero significant rises, three
significant falls across book shapes, worst on hedged books); (2) the family's
common-γ construction overwrites the fitted correlation with a floor of 0.43+
at plausible parameters, and below ν=4 — where the young-asset fit actually
landed — its variance is infinite and correlation targeting is undefined;
(4) for BTC/ETH, the one pair flagged in both tails, NO admissible γ reaches
the observed upper tail (0.741 against a reachable [0.498, 0.541]), so the
skew dimension cannot fix what the reading shows. What survives as remedy
space: moving the copula df (or the elliptical family) to lift BOTH tails,
gated by a one-sided, symmetry-conditional lower-tail criterion
(finding 3's measured 2.5% null instead of the unconditional rule's 84%
false-positive rate) — and finding 1's result stands as the requirement that
any such change be validated per-output on the book population, which is what
the shadow window under A10's recording mode collects. This is a design
conclusion, not a design: the choice of df target and family is a
model-scope decision (MODEL_VERSION MINOR, §3.3 reset) that belongs to the
product owner, informed by the window.

**ADOPTED and IMPLEMENTED (2026-08-05, option A; MODEL_VERSION 0.5.0).** The
proposal below was accepted by the product owner and is live:
`tail_floor_df` in `model/copula.py`, wired through
`state._tail_floored_copula_df` into both bundle builders, before the §2.3
gate. (0.7.0 renamed the wrapper `_tail_remedied_dependence` when the
rho-lift composed in front of the floor; `tail_floor_df` survives inside the
chain.) On the 2026-08-05 readings it floors the df from the ML fit of 6.5 to
**2.5** (the grid floor) on the single ≥2σ demand, ETH/SOL; a dress rehearsal
of the next live build shows all three pairs passing the gate afterwards
(residual shortfalls +0.063/+0.009/+0.011 against SE-scaled margins), so the
recording-mode banner goes dark and gate-days can accumulate.

**One correction to the proposal as first committed, owned rather than
papered over:** it claimed df 3.0 "brings every pair's lower-tail shortfall
inside 0.02" and "the shipped 0.05 gate passes everywhere". False for the
signal pair: ETH/SOL's shortfall at df 3.0 is 0.068 > 0.05, so under the old
flat margin the gate would have kept firing and the clock would never have
started. The adopted design therefore couples the floor with the second half
of finding 3: the gate margin is now `max(0.05, 1.645·SE)` — the flat 0.05
was 1.13·SE at the live n=108, a criterion with no null — and the floor
targets exactly the bound the new margin tests, which is what makes "floored
⟹ gate passes" true by construction rather than by luck. A borderline
Monte-Carlo flip between the two estimates re-fires the gate and the next
build floors deeper: self-correcting, not silent.

Per-output at the actually-adopted df (6.5 → 2.5, 40k paths × 12 seeds):
nothing rises; long-only 13x −0.0010 (−2.8 SE), hedged 36x −0.0026 (−4.2 SE),
the rest ~0. Same shape as the 3.0 table below, slightly larger.

**FIRST LIVE FIRING (2026-08-05 12:15) — the floor engaged and hit the
family's wall; two of this entry's own premises corrected by it.** The live
build floored `3.50 -> 2.50, covered=False`: the ML fit on the live window is
**3.5, not 6.5** (6.5 was the FIXTURE value; the rehearsal above inherited
that error, which is why it predicted the gate would clear), and the pair
correlations back out at **ρ ≈ 0.863**, not 0.887 (both of this morning's
banner asymptotics agree: 0.593@3.5 and 0.642@2.5 both invert to 0.863). At
the real (ρ, df): the grid floor 2.5 reaches model@q ≈ 0.679, ETH/SOL's
bound needs 0.691, **df 2.0 measures 0.688 — still short — and only 1.5
covers**, which is the foreclosed near-Cauchy territory against an ML of 3.5.

So the shortfall that remains is not a tuning problem. ETH/SOL is the
ASYMMETRIC pair (+0.074 lower over upper), and a symmetric family covering
its lower tail must overstate its upper by the same distance; the floor
closed what symmetry allows (shortfall 0.111 → 0.080, 2.7σ → 2.0σ) and the
rest is the family boundary A11's opening paragraphs predicted for exactly
this shape of reading.

**A second premise of this entry was also wrong and matters for the owner's
option B:** "n doubles in ~2 weeks, SE shrinks 1.4x" is false. The window is
a ROLLING 90 days — n stays ≈ 108 forever. Waiting does not shrink the SE;
the reading wanders (0.759 today) and the gate outcome with it. Option B is
a coin flip on that wander, not convergence, and is re-priced accordingly.

**Also fixed on this firing (0.6.0): the stuck band.** As shipped in 0.5.0
the gate fired at 1.645·SE while a pair became a floor DEMAND only at 2.0σ,
so a reading in [1.645σ, 2.0σ) held the gate lit — recording mode, no
gate-days — while the floor never attempted a remedy. The demand threshold
is now the same constant as the gate's (`TAIL_DEMAND_SIGMAS =
TAIL_ONE_SIDED_Z`): every pair that can keep the gate open is a pair the
floor tries to cover. Conditionality survives — 1.645 one-sided is a real
null, which is all finding 3 asked for.

**Standing after the firing:** recording mode continues, correctly — the
demands exceed what the symmetric family can express on the asymmetric pair.
The §3.3 clock is held by that fact, not by a defect. The levers that remain
are the ones this entry already names: the ρ-lever redesign (tail-matched /
crash-regime dependence — reaches 0.691 easily, since ρ ≈ 0.935 reproduces
the observed tail at any df in 3–6.5), or accepting recording mode while the
reading wanders. Chasing df past the grid floor stays foreclosed.

**HYPE probe (2026-08-05 evening) — HYPE adds no gate surface at the ML df.**
The owner ran the engine once with `HL_UNIVERSE=BTC,ETH,SOL,HYPE` (the B6
census's candidate: largest single off-universe recovery, ≈19 addresses per
sweep). The ML fit moved 3.5 → 5.0, which is the proof HYPE entered the joint
fit — the common window re-profiled around its history. The floor banner read
`floored 5.00 -> 2.50, covered=False` with demands ETH/SOL (+4.0σ), BTC/SOL
(+2.1σ), BTC/ETH (+1.9σ) — the two BTC pairs are demands now because 0.6.0's
shared threshold admits what 0.5.0's stuck band excluded, not because their
readings moved. No HYPE pair fired, at the ML df or the floored one, so on
today's window HYPE widens the calibration cohort without widening the §2.3
gate's attack surface. (At the floored 2.5 the residual refusal named
ETH/SOL alone — the two BTC-pair demands are covered by the floor; the
asymmetric pair remains the family's wall, unchanged.)

One observability defect found by the probe, fixed the same evening: the
banner prints FIRING pairs only, so "HYPE is quiet" had to be inferred from
its pairs' absence — and absence cannot distinguish a passing pair from one
that never entered the fit. The remedy wrapper in `service/state.py` (then
`_tail_floored_copula_df`, `_tail_remedied_dependence` since 0.7.0) now logs
every pair's reading (excess σ, empirical lower, model@q) at INFO on every
build, quiet pairs included, pinned by test. The next probe reads readings,
not silence.

**PROPOSAL (2026-08-05, second): a conditional ρ-lift on demand pairs.** The
remaining lever, turned into numbers an owner can decide on — measured, like
the df proposal before it, against the live readings (ETH/SOL empirical lower
0.759, one-sided 95% bound 0.6913, EWMA ρ backed out at 0.863, ML df 5.0 with
HYPE / 3.5 without).

*Reach — the ρ-lever covers where the df lever could not:*

| df_ML | ρ* covering the ETH/SOL bound | lift from EWMA 0.863 |
|---|---|---|
| 5.0 | 0.907 | +0.044 |
| 3.5 | 0.896 | +0.033 |
| 2.5 | 0.878 | +0.015 |

The target is attainable at the ML df with room to spare — no near-Cauchy
territory, no grid wall.

*Body cost:* lifting ETH/SOL 0.863 → 0.907 at df 5.0 costs **+0.0266
nats/obs** of copula log-likelihood on a live-shaped window (57.5 over 2160
observations). Chasing the POINT estimate instead (ρ ≈ 0.935) costs 4× that
(+0.1045 nats/obs, 225.8 total); the bound, not the point, stays the target
for the same reason as in the floor design.

*Positive definiteness:* on the live-shaped 4-asset matrix both lifts stay PD
outright; the standard projection returns the matrix unchanged (entries move
≤ 1e-4). The mechanism still projects and RE-VERIFIES coverage afterwards —
on a future matrix the projection could pull a lifted entry back.

*Per-output disclosure (finding 1's requirement):* measured at (ρ 0.863 →
0.907, df 5.0) on matched two-asset cross books, 40k paths × 12 seeds, the
pair's live readings substituted into the fixture bundle; third column is the
0.6.0 floor's actual output (ρ 0.863, df 2.5) for comparison:

| book | P @ML | P @floor | P @lift | lift−ML | lift−floor |
|---|---|---|---|---|---|
| long-only 10x | 0.175 | 0.174 | 0.180 | **+0.0052** (+35 SE) | +0.0060 |
| long-only 13x | 0.344 | 0.344 | 0.350 | **+0.0057** (+42 SE) | +0.0057 |
| hedged 28x | 0.514 | 0.514 | 0.467 | **−0.0474** (−131 SE) | −0.0470 |
| hedged 36x | 0.792 | 0.791 | 0.766 | **−0.0256** (−114 SE) | −0.0245 |
| short-only 10x | 0.228 | 0.227 | 0.233 | **+0.0052** (+23 SE) | +0.0056 |
| short-only 13x | 0.389 | 0.388 | 0.395 | **+0.0056** (+41 SE) | +0.0064 |

Three readings, all load-bearing:

1. **The sign varies by shape**, exactly as finding 1 requires disclosing:
   same-sign books rise ~+0.5 pp, hedged books FALL 2.4–4.7 pp — an order of
   magnitude larger. The §10 case is that the tail statistic says the market's
   dependence EXCEEDS the model's, so the current model overstates a hedged
   book's P(liq) and understates a same-sign book's; the lift moves every
   shape TOWARD the measurement. The honest counterpoint, stated rather than
   buried: the lift applies a tail-motivated ρ to the WHOLE distribution,
   including the body, where the EWMA 0.863 is the better estimate — part of
   the hedged fall is body distortion, not measured truth. The crash-regime
   redesign (ρ high only in the crash state) would confine the lift to where
   the evidence is; it remains the sharper, larger, deferred design.
2. **The df floor barely moves outputs at all** (ML vs floor columns differ by
   ≤ 0.1 pp everywhere): at ρ 0.863 the pair is near-comonotone and df has
   almost no room to act on a two-asset book at a 24h horizon. The df lever is
   weak in OUTPUT space, not only in tail-statistic reach — on this venue's
   readings, ρ is the operative lever, df is not.
3. **lift−floor ≈ lift−ML**: the comparison the owner actually faces (what
   recording mode records today vs the proposal) is the same table.

*Mechanism, if adopted (option R):* after the ML fit, with the SAME demand set
the floor uses (pairs ≥ 1.645σ significant shortfall): lift each demand
pair's correlation entry upward to the smallest grid value whose model@q at
df_ML covers that pair's one-sided 95% lower bound, cap 0.98; project to PD;
re-verify every demand post-projection; where a demand stays uncovered under
the cap, the df floor composes on top, and failing that `covered=False` keeps
the gate lit exactly as today. Conditional (no demand → matrix untouched),
one-sided (lifts only, never cuts), recomputed each rebuild from current
readings. The serving matrix then diverges from the EWMA on lifted pairs;
both numbers are in the log — the new readings line prints every pair.

*Costs and standing:* MODEL_VERSION MINOR (0.6 → 0.7), §3.3 reset — free
while the clock is held by `covered=False`, which it is. Latency nil (same
machinery, one matrix entry differs). The decision is the owner's; the
options are (R) adopt the ρ-lift mechanism; (R+H) adopt it together with
adding HYPE to `HL_UNIVERSE` in the same bump — the probe above shows HYPE
adds no gate surface at the ML df, the census shows it is the largest single
cohort recovery, and one reset is cheaper than two; or (S) stay in recording
mode while the reading wanders — re-priced by the rolling-window correction
above: waiting is a coin flip, not convergence.

**ADOPTED and IMPLEMENTED (2026-08-05 evening, option R+H; MODEL_VERSION
0.7.0).** The owner took R+H. `tail_remedy_dependence` in `model/copula.py`
is the chain — the conditional rho-lift exactly as specified above (same
demand set as the floor, one-sided, bisection to the smallest covering rho at
the ML df, cap 0.98, PD projection, coverage RE-VERIFIED on the projected
entries), with `tail_floor_df` composing on any demand the cap cannot reach
and `covered=False` keeping the gate lit beyond that. Wired through
`state._tail_remedied_dependence` (the renamed floor wrapper) into both
bundle builders; the bundle carries the LIFTED matrix, so the §2.3 gate and
the simulation judge the same served dependence. (`/health` is NOT in that
list: its diagnostics block reports the ESTIMATION pipeline — shrinkage,
eigenvalues, imputation — computed before the lift; the lift's operator
record is the per-pair WARNING log and the `copula_rho_tail_lifted`
counter.) The lift is logged per pair (from → to, target) beside the
all-pairs readings line, and mutation-gate entries pin the safety
properties. HYPE entered the universe default in the same bump — the B6
record has the census arithmetic and the probe is above. Expected first live
firing: all three demand pairs lifted at the ML df ~5.0 (ETH/SOL to ~0.907,
the BTC pairs by less), no floor, gate dark, recording mode ends, gate-days
begin.

**Hardened the same night: an adversarial review of the diff found two real
holes in the first implementation**, both of the covered-while-firing class
the chain exists to close. (1) Coverage was re-verified only on the DEMAND
pairs, so the PD projection could redistribute a lift's distortion onto a
bystander pair — pushing it past its own margin (the gate fires on a pair
the remedy never looked at, while the log says covered) or below its
measured correlation (served co-crash understated, the §10 direction,
silently). (2) The composed floor still scored coverage with the historical
seed-7 estimator while the gate scores with per-pair seeds — the
two-estimator band, re-opened on the floor branch at a zero-slack boundary.
Fixed by making the chain's verdict `uncovered_at_gate`: the final served
(matrix, df) is re-scored on EVERY pair with the gate's own estimator, so
`covered` IS the gate's verdict computed ahead of time, `uncovered_pairs`
names what will fire, and the floor's walk takes the gate's seeds. Closed in
the same pass: a demand pair whose measured ρ already exceeds the 0.98 cap
is no longer cut down to it (the search domain is [ρ_from, max(cap,
ρ_from)]; the lift's only direction is up); an unmeasurable pair (NaN lower
tail, zero-SE reading) now yields covered=False instead of a vacuous
covered=True over a firing gate; any projection move that leaves a served
entry below its EWMA measurement is logged per pair; and the census header
no longer claims cross-version pooling is universe-clean — 0.7 moved the
universe, so pre-0.7 and post-0.7 census rows describe different cohort
frames.

**PROPOSAL (2026-08-05): a conditional tail floor on the copula df.** The
remedy space above, turned into numbers an owner can decide on. Measured
against the live 2026-08-05 sweep banner (ETH/SOL lower 0.759 at +2.7σ,
BTC/ETH 0.704/0.741 at +1.3σ; n=108/tail, SE ≈ 0.042), with each pair's ρ
backed out of its own model@q reading at the fitted df 6.5: ρ ≈ 0.884–0.887
on all three pairs.

*The df lever is weak, and its reach is now known.* Walking df down at the
fitted ρ:

| df | 6.5 | 5.0 | 4.0 | 3.0 | 2.0 | 1.5 | 1.05 |
|---|---|---|---|---|---|---|---|
| ETH/SOL model@q | 0.651 | 0.661 | 0.673 | 0.691 | 0.716 | 0.735 | 0.762 |

Covering the POINT estimate (0.759) needs df ≈ 1.05 — a near-Cauchy copula,
foreclosed: it would wreck the body fit and every other pair to chase one
number that carries a 0.042 SE. Covering the one-sided 95% lower confidence
bound (0.759 − 1.645·SE = 0.690) needs **df 3.0**, which also brings every
pair's lower-tail shortfall inside 0.02 (BTC/ETH 0.687 vs 0.704, BTC/SOL
0.676 vs 0.694) — the shipped 0.05 gate passes everywhere — and its symmetric
upper-tail cost is negligible where it overstates (ETH/SOL +0.005, BTC/SOL
+0.028 = 0.7 SE) while partially closing BTC/ETH's under-modelled upper
(0.687 vs 0.741, was 0.648).

*The mechanism proposed:* `df* = min(df_ML, df_tail)`, where `df_tail` is the
largest df whose model@q covers `empirical_lower − 1.645·SE` on every pair
failing the gate at ≥ 2σ. Conditional exactly as finding 3 requires (pairs
under 2σ are noise to re-test, not to fit — today that excludes BTC/ETH and
BTC/SOL as demands), one-sided exactly as finding 4 requires (λ_U is reported,
never constrained). Today it yields df* = 3.0 against an ML fit of 6.5.

*Per-output disclosure (finding 1's requirement — the sign cannot be assumed):*
measured at df 6.5 → 3.0 on matched two-asset cross books, 40k paths × 12
seeds:

| book | 10–13x long | 28x hedged | 36x hedged | 10–13x short |
|---|---|---|---|---|
| ΔP(liq) | −0.0007 (−2.1 SE) / ~0 | ~0 | −0.0021 (−3.2 SE) | ~0 |

Nothing rises; two cells fall by ≤ 0.2 pp. So this change is honest about the
measured tail rather than uniformly conservative per-output — the A1
situation, satisfied by this disclosure, exactly as finding 1 concluded it
must be.

*The alternative lever, recorded for the owner but not proposed:* the same
observed tail is reproduced at the CURRENT df by ρ ≈ 0.944 (vs the fitted
0.887, at any df in 3–6.5 the needed ρ is 0.935–0.944). That reading says the
gap may sit in the correlation estimate, not the tail shape: the EWMA matrix
weights recent hours while the tail statistic is dominated by whatever regime
held during the window's crashes — "correlations go to 1 in a crash" is
precisely what the exceedances measure. Tail-matched ρ would be the sharper
model and the larger change (per-pair adjustments threaten positive
definiteness; it reweights the whole matrix pipeline). Deferred, not
dismissed.

*Costs and standing:* MODEL_VERSION MINOR (0.4 → 0.5), §3.3 counter reset —
free while the counter stands at zero, which it does and stays until this
very decision is taken. Latency: nil (same machinery, same draws). The
decision is the owner's; the options are (A) adopt the floor as specified,
(B) defer for more data — n doubles in ~2 weeks and SE shrinks 1.4x, so the
2.7σ either consolidates or regresses — at the price of the window not
starting, or (C) take the ρ-lever redesign instead. Chasing the point with
df ≈ 1 is recorded as foreclosed, not as an option.

Latency is a separate constraint on the remedy and is already tight:
`generate_log_returns` at 20 000 × 24 × 8 measures **301 ms**, the whole of
§2.6's budget, with 221 ms of it in the quantile maps. A skew-t marginal is
not odd, so `QuantileMap`'s half-table, `np.abs` and `copysign`
(`sim/quantile_map.py:92-94, 122, 148`) are all invalid for it, and the two
tails have different polynomial indices (`ν/2` toward the skew, `ν` away —
confirmed by Hill estimation), so a single extrapolation slope reused across
both would understate the far lower tail by a factor of two.

**Finding 1 measured, 2026-08-04 — it holds, and in the direction §10 cares
about.** It had been an argument from the code's shape (`_any_liq` is a union,
`Position.size` is signed) and was waiting on the window. It does not need to:
the claim is about the model's response, so it is measurable on constructed
books today, and only the POPULATION question needs live data.

Heavier joint tail dependence applied by lowering the copula df from the
fitted 6.5 to 3.5, on books matched at 100k equity and differing only in sign
pattern. 40 000 paths x 16 seeds; the SE is over seeds.

| book | leverage | P(liq) @6.5 | P(liq) @3.5 | delta | |
|---|---|---|---|---|---|
| long-only | 10x | 0.04803 | 0.04797 | −0.00006 | — |
| long-only | 13x | 0.15274 | 0.15318 | +0.00044 | +1.4 SE |
| hedged | 28x | 0.17277 | 0.17178 | **−0.00100** | **−2.6 SE** |
| hedged | 36x | 0.40885 | 0.40695 | **−0.00189** | **−3.1 SE** |
| short-only | 10x | 0.07742 | 0.07705 | −0.00038 | — |
| short-only | 13x | 0.19213 | 0.19125 | **−0.00088** | **−2.9 SE** |

**Zero significant rises; three significant falls.** So "heavier joint tails
is the conservative direction" is not merely unproven, it is false on these
books: the direction §2.3 prescribes can LOWER a reported P(liq), which is
what §10 forbids. The largest fall is on the hedged book, exactly the shape
finding 1 predicts — for a signed combination, co-movement cancels rather than
accumulates, so binding the assets together more tightly makes the hedge work
harder.

Note what did NOT happen: long-only did not rise significantly either
(+1.4 SE at 13x). The effect is not symmetric in magnitude — the falls are
larger and cleaner than the rise — so a design that assumed "it helps
long-only more than it hurts hedged" would also be unsupported.

Two limits on this, stated because they bound what it licenses. It uses a
SYMMETRIC heavying (lower df moves both tails), where the prescribed remedy is
a skewed-t that heavies the lower tail at the upper's expense; that makes this
a test of the easier, more obviously-safe-looking direction, and it already
fails. And it is two-asset books on the fixture bundle, so the population mix
of shapes on Hyperliquid — which decides how much this matters in aggregate —
is still what the window is for.

What the shadow window under A10's recording mode is for: measuring the
magnitude and the direction of (1) on real books — the population question
above, now that the mechanism itself is settled.

**Recording mode is now enforced, not announced (2026-08-04).** The sweep
printed "these observations are DIAGNOSTIC EVIDENCE, not §3.3 gate-days" and
that was the whole of it: nothing in the schema distinguished them, so
`progress()` counted a diagnostic day toward Phase 4 exactly as it counted a
clean one. `_print_defect_note`'s own docstring named the standard it was
failing — a journal of observations collected under a known model defect,
indistinguishable from a clean one, "is worse than no journal, it would be
read as gate progress".

It matters more here than the phrasing suggests, because the gate is what
Phase 4 (real money) opens on, and the remedy for this very defect changes
the copula, bumps MODEL_VERSION and resets the counter. Every day counted
under the defect is a day that cannot survive the fix it is waiting for.

Each prediction row now carries `recorded_under_defect`, stamped from the same
predicate the serving path uses (`understates_lower_tail`, which had promised
"one predicate, so the serving path and the provenance stamp cannot drift
apart" while having no stamp to keep in step), and `progress()` excludes it
exactly as it already excluded stale resolutions. Rows written before the
column existed are backfilled to TRUE, not FALSE: their provenance was never
captured, §10 resolves that uncertainty toward not counting them, and the
deployment holding such rows logged the defect on every sweep anyway.

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

**RE-DERIVED 2026-08-04.** The half-widths below were computed with an
interval that did not cover: `clustered_rate_ci` was a day-clustered
PERCENTILE bootstrap, and a percentile bootstrap undercovers badly when the
clusters are few. Measured coverage against a nominal 95% was 94.8% at
ICC 0, 89.0% at 0.10, **82.8% at 0.20** and 79.2% at 0.40. Resampling days
was necessary — this entry's whole argument — and it was not sufficient.

That is the same lesson this entry already records having learned on the
OTHER interval ("the day-clustered percentile bootstrap covers 43% at 14
days ... Both were discarded. The shipped intervals invert the test"). The
fix went into `clustering.py`'s ICC estimator and never reached this one, so
the sizing table was built on a too-narrow interval. Both are now
studentised; coverage after the change is 94-95% across the range.

The corrected table, 400 trials per cell:

| days | addr/day | ICC | §0.3 rejects a *correct* model | clustered ±pp | power vs 8% | power vs 10% |
|---|---|---|---|---|---|---|
| 21 | 200 | 0.00 | 4.5% | 0.70 | 100% | 100% |
| 21 | 200 | 0.05 | 53.0% | 2.43 | 70.0% | 98.0% |
| 21 | 200 | 0.10 | 70.0% | 3.48 | 39.8% | 80.5% |
| 21 | 200 | 0.20 | 74.2% | **5.80** | 21.0% | 47.8% |
| 21 | 200 | 0.40 | 83.5% | **15.36** | 13.0% | 26.8% |
| 60 | 200 | 0.20 | 74.8% | 2.79 | 58.8% | 93.2% |
| 90 | 200 | 0.20 | 75.0% | 2.22 | 75.5% | 99.5% |
| 180 | 200 | 0.20 | 74.8% | 1.51 | 97.0% | 100% |
| 180 | 200 | 0.40 | 83.2% | 2.16 | 76.5% | 98.8% |

What moved, and it is the half-widths rather than the conclusions: 3.60 →
**5.80** pp at ICC 0.20, and 4.87 → **15.36** pp at 0.40. So a 21-day window
bounds the breach rate to about ±5.8pp at the plausible clustering, not
±3.6pp — the gate was reporting a precision it did not have, in §10's
direction. The day counts this entry recommends are unchanged, because power
depends on both endpoints moving together; what changes is the honesty of
the number a published score would quote.

For the record, the superseded figures, computed with the undercovering
interval:

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
   clustered half-width barely at all at ICC 0.20. Under independence
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

### B2 `[RESOLVED for readability 2026-08-04 — one residual, named]` Equity changes for reasons the model does not predict

**The 200-address frame sweep ran, and it closes what B2 was actually afraid
of.** 23 599 ledger records across **200 of 200** sampled addresses, **16
distinct delta types, every one readable**:

    accountClassTransfer, borrowLend, cStakingTransfer, deposit,
    gossipPriorityGasAuction, internalTransfer, liquidation, rewardsClaim,
    send, spotGenesis, spotTransfer, subAccountTransfer, vaultDeposit,
    vaultDistribution, vaultWithdraw, withdraw

That includes all three types this entry flagged as known-unproven and
near-certain to appear — `internalTransfer`, `subAccountTransfer`,
`accountClassTransfer`. The failure mode they threatened was specific and
silent: an unclassifiable type raises inside `resolve_due`, is filed
TRANSIENT by name, and is retried forever, so it withholds an address's
observations while the counter fails to advance. Encountered on day 6 it
costs the window. It cost one command instead.

**The residual, and its direction.** Nine dex names appeared that are neither
the primary perp account nor the known spot value:

    abcd, cash, flx, hyna, km, mkts, para, vntl, xyz

They read as builder-deployed venues (HIP-3), each its own margin space with
its own `clearinghouseState`, so a transfer to one has left the account this
model predicts and is correctly counted as an outflow. That reading is safe
for every case but one: if any name is an ALIAS for the primary dex, a
transfer to it never left the book, and subtracting it as an outflow
overstates the model-attributable equity change — a corruption of the very
correction B2 exists to make.

`UNRECOGNISED_DEX_NAMES` surfaces them rather than merely handling them,
which is why they are in this entry at all. Settling it means confirming
against the venue that each is a deployed builder dex; anything found to be
an alias goes in `PERP_DEX_VALUES`. Not settled by plausibility here — the
names look exactly like builder identifiers, and C4 is the standing reminder
of what that kind of confidence is worth.

The original entry follows.

---

### B2 (original) `[BLOCKER]` Equity changes for reasons the model does not predict

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

### B3 `[DECIDED — reading confirmed, and this entry described it wrongly]` Baseline A is not a distribution

**Decided 2026-08-03.** The implemented reading stands: §3.2's "historical
unconditional frequency" is a single number, CRPS needs a distribution, and
the only faithful way to get one is to keep the naive PREDICTOR and let it
produce a distribution. That is what the code does.

**But the description below is wrong in two ways, both found by measuring the
code rather than reading it.**

*It is not `net_exposure x r`, it is `net_exposure x (e^r - 1)`.* The draws
are 24 h LOG returns and the code exponentiates them. On a long-only
two-coin book at $320 000 net notional, the predicted equity change matches
`net_notional x (exp(r) - 1)` at the 5th and 95th percentiles to within 0.05%
(-19 869 vs -19 860, +21 136 vs +21 264). At crypto's 24 h scale the two
differ enough to matter in the tail, which is the part that decides a
liquidation.

*The book is NOT collapsed to net notional.* One common factor is applied to
every asset and then the REAL liquidation model runs on the REAL positions.
Only the equity change collapses; P(liq) does not. Measured at $1 000 000 net
notional on $60 000 collateral, identical for all three:

| structure | P(liq) | sd(equity change) |
|---|---|---|
| one cross position | 0.3251 | 81 335 |
| one isolated pocket | 0.3251 | 81 335 |
| two isolated pockets | **0.3538** | 81 506 |

Same net notional, same collateral, different P(liq) — because isolated
pockets fail independently and `any_liq` is a union over them. A description
that says "collapsed to net notional" would have someone predict 0.3251 for
the third row.

**Why it is a fair baseline, and not a straw man.** One factor means perfect
correlation and no idiosyncratic risk, so a hedged book looks nearly
riskless to it: measured sd of 1 590 on a book whose legs are $200 000 long
against $160 000 short. The real model gives that book genuine
idiosyncratic risk. So "the model beats Baseline A" is informative precisely
on the books where correlation structure is the thing that matters — which is
the comparison §3.2 is for. It is naive in the intended way: no correlation
structure, no per-asset volatility, no funding, no path (one step, endpoint
only, so no intra-horizon monitoring).

What remains yours: whether §3.2 meant this predictor. Nothing measurable
settles that.

The original entry follows.

---

### B3 (original) `[BLOCKER]` Baseline A is not a distribution

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
`HL_UNIVERSE` (comma-separated; default `BTC,ETH,SOL` then, widened to
`BTC,ETH,SOL,HYPE` by the 2026-08-05 decision below), read by
`_build_live_bundle` and passed through the compose stack, so widening it
does not require a code edit. BTC and ETH remain mandatory (§2.1). What
widening costs, per added coin: one 90-day candle snapshot plus one funding
history per rebuild (~112 weight with response surcharges, C6's corrected
table; this line said ~40 until 2026-08-06 — base weights only, the same
drift resolve.py documents), a row and column of correlation structure,
one more marginal fit, and a bigger draw per path-step. What it is NOT: a
free fix. It is a sampling-frame change AND a distribution change
(MODEL_VERSION MINOR, §3.3 counter reset), so it must happen BEFORE the
shadow clock starts or cost the accumulated days.

**Measured 2026-08-04, first complete live sweep** (515 addresses, mainnet,
`HL_UNIVERSE=BTC,ETH,SOL`). 299 skip lines were captured:

| reason | n | share |
|---|---|---|
| off-universe holding | 196 | 66% |
| no open positions | 103 | 34% |

Two thirds of the loss is the universe, so widening is aimed at the right
thing — but the tail is long. **42 distinct coins** appeared, and the top of
the distribution is not concentrated enough to fix cheaply:

| coin | n | cumulative |
|---|---|---|
| HYPE | 47 | 24% |
| ATOM | 22 | 35% |
| AVAX | 18 | 44% |
| DOGE | 13 | 51% |
| XRP | 11 | 57% |
| BNB | 8 | 61% |
| DYDX | 7 | 64% |
| … 35 more | 70 | 100% |

Reaching 87% of off-universe skips takes about 20 added coins. At ~112
weight per coin per rebuild (C6's corrected table), 20 coins is ~2 240
weight per rebuild against a 300/min shadow pool — roughly eight minutes of
budget for every bundle build, hourly on the resolver.

**And the counts above overstate what any given widening buys**, which is a
measurement defect this entry itself introduced. The procedure below used to
say "sum the `KeyError: '<COIN>'` counts by coin, most-dropped first". That
tally answers *how often a coin is the FIRST one missing*, never *how many
addresses a universe containing it would recover*: an address holding ATOM
and HYPE is filed under whichever a dict lookup reached first, so adding the
top coin alone can recover **nothing**, if every address holding it also
holds a second off-universe coin. Fixed 2026-08-04 — the sweep now names
every missing coin (`off-universe: ATOM, HYPE`), so an address is recovered
by universe U exactly when its whole list is inside U, and that is
computable from the census. The table above is the OLD encoding and its
per-coin counts should be read as upper bounds.

**One thing this entry got wrong about the code**, corrected here rather
than left to mislead the next reading: it says an address whose positions
are all off-universe "is skipped as `no open positions`, byte-identical to a
flat account". That is not what happens. `LiveSnapshotProvider.book` does
not filter to the universe (`universe` is used only by `spot()`), so such a
book has positions and fails later. The two reasons were already distinct in
the log, which is why the table above can separate them at all — and that
separation is the whole value of the census, since flat accounts are not
recoverable by widening and off-universe ones are.

**The procedure is executable as of 2026-08-04:**

    python -m risk_engine.shadow census

Until then this table had been written since 2026-08-01 and read by **nothing**
— `record_sweep` wrote it every day and no code path anywhere loaded it back,
so the data accumulated against a question that could not be asked of it. B6
was blocked on a reader as much as on days. `journal.sweeps()` and
`shadow census` are that reader; the arithmetic below is what it runs.

The decision procedure, concretely:
1. let the census accumulate a few days of full sweeps **under the new
   reason format** — the old rows cannot answer step 2. `census` marks any
   sweep that hit the §5.3 budget, because a truncated sweep's drop rates
   describe a prefix of the address list rather than the list;
2. for each candidate universe U, count the addresses whose entire
   `off-universe:` list is inside U. Not a per-coin sum — `off_universe_demand`
   keys on the whole SET for exactly the reason this entry gives above, and
   `universe_candidates` widens greedily, reporting what each universe
   actually recovers. On a census where ten addresses hold ATOM *and* HYPE, a
   per-coin tally promises "add ATOM, recover 10" and the truth is zero;
3. if the off-universe drop rate keeps `written` comfortably above §3.3's
   200/day floor, keep the universe and let the published score disclose the
   cohort selection this table records;
4. if it does not, set `HL_UNIVERSE` to the smallest U that clears the floor,
   bump MODEL_VERSION, record the new frame here, and start the clock then.

Note what `recovered` is and is not: it counts addresses the sweep would
**attempt**, which is an upper bound on what it writes. Flat books and
non-positive equity are excluded from the demand entirely, because widening
the universe cannot recover them.

**DECIDED (2026-08-05): HYPE enters the universe; the long tail does not.**
The census the procedure asked for exists (10 full sweeps pooled across
distribution versions — the drop pattern depends on `HL_UNIVERSE`, not the
model version) and it answers the steps:

- step 3 first, honestly: `written` ran 210–222 per sweep, already above
  §3.3's 200/day floor, so no widening was FORCED. The floor's binding axis
  is the 21 DAYS, not the addresses.
- the set-based arithmetic names HYPE the largest single recovery (≈19
  addresses/sweep — the venue's own native token, held by a quarter of the
  off-universe cohort) at one coin's cost (~112 weight/rebuild with response
  surcharges — this record first said ~40, repeating exactly the stale-figure
  drift resolve.py documents; the corrected number is what starved the shadow
  bundle build on the first 4-coin start and forced the per-charge pacing fix
  in `_paced_bundle` — one more marginal, three new correlation pairs). The
  next candidates fall off fast
  and the tail is flat: ~166 addresses/sweep hold some coin no plausible
  universe contains, and reaching 87% of skips takes ~20 coins at ~2 240
  weight/rebuild — priced out, deferred, not dismissed.
- the §2.3 surface was probed before adopting, not assumed: a live run with
  `HL_UNIVERSE=BTC,ETH,SOL,HYPE` moved the ML df 3.5 → 5.0 (HYPE entered the
  joint fit) and no HYPE pair fired the gate at any df (A11's probe record).

Adopted as the default (`BTC,ETH,SOL,HYPE` in `_build_live_bundle` and the
compose stack) in the SAME 0.7.0 bump as A11's rho-lift, deliberately: both
are distribution changes, the §3.3 counter was held at zero by the 0.6.0
`covered=False` firings, and two changes sharing one reset is the whole
timing discipline this file keeps re-learning. The published score's frame:
cohort = addresses whose every position is inside BTC,ETH,SOL,HYPE, with the
census disclosing the ~166/sweep that stay outside.

---

## C. Hyperliquid integration — facts that must be verified, not assumed

### C1 `[RESOLVED 2026-08-03]` The funding-rate protocol clamp (§1.5)

**Closed by reading the primary source.** The operator opened
`https://hyperliquid.gitbook.io/hyperliquid-docs/trading/funding` on
2026-08-03, and it states, in its own words:

> Funding on Hyperliquid is capped at 4%/hour. Note that this is much less
> aggressive capping than CEX counterparts. The funding cap and funding
> interval do not depend on the asset.

That is `HL_DOCUMENTED_HOURLY_CAP = 0.04`, confirmed on the basis this code
clips: **per hour**, and asset-independent, which the single constant already
assumed. The shipped bound is now
`FundingBounds.hyperliquid_confirmed()` — the citation, the quoted sentence
and the read date live in `model/funding.py` — and `check_funding_clamp` has
a reachable PASS for the first time.

**The trap this entry named in advance was real, and the page contains both
numbers.** The published formula is

    F = P + clamp(interest_rate - P, -0.0005, 0.0005)

so ±0.0005 bounds the interest-rate term *inside* the formula, while 4%/hour
bounds the realised rate. Recording ±0.0005 as `cap_per_hour` would have
clipped simulated funding at 1/80th of the true bound — understating cost of
carry, the §10-forbidden direction. Both constants are now named in the code
with that distinction attached, so the next reader does not have to
rediscover which is which.

**One residual ambiguity, and it fails safe.** The page also says the formula
computes an 8-hour rate paid hourly at one eighth. If the 4% cap were meant
against that 8-hour rate, the true hourly bound would be 0.5% and this value
would be 8× too permissive — which lets the model simulate funding *more*
extreme than the protocol allows and therefore OVERSTATES cost of carry, a
direction §10 permits. The plain reading is the direct one (the cap is
written "%/hour", the same basis `fundingHistory` reports and this bound
clips), and the alternative reading is the harmless one, so nothing turns on
resolving it further.

Note what closing this does NOT change: the measurement below still stands,
and it is what makes the bound unimportant in practice. A confirmed bound
1760× above anything the market did in a month is a guard rail, not a
distribution parameter. Confirmation was always about provenance.

The original entry follows.

---

### C1 (original) `[BLOCKER — downgraded]` The funding-rate protocol clamp (§1.5)

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

### C4 `[RESOLVED 2026-08-03 — and it inverts §5.2]` `webData2` vs `webData3`

**Measured, four probes, one address, one session:**

| subscription | `user` | venue |
|---|---|---|
| `webData2` | absent | rejected — `Error parsing JSON into valid websocket request` |
| `webData3` | absent | rejected — same message |
| **`webData3`** | **present** | **ACCEPTED** — `{"channel":"subscriptionResponse","data":{"method":"subscribe","subscription":{"type":"webData3","user":"0x963a…"}}}` |
| **`webData2`** | **present** | **rejected** — same parse error, same payload shape |

Two results, and the second is the one nobody was looking for.

**`webData3` exists.** Settled by an explicit `subscriptionResponse` echoing
the subscription back, which is as unambiguous as this venue gets. Both
subscriptions are keyed on `user`, and that is why the bare probe told us
nothing: a missing required field and an unknown type produce the identical
parse error.

**`webData2` does NOT work on this venue** — refused with a well-formed
`user`, in the exact payload shape `webData3` accepted a moment earlier. The
shape is therefore not in question. §5.2 names `webData2` as the documented
subscription and this repository repeated that in nine places; anything built
to that letter would subscribe to something the venue rejects.

**Nothing is broken today, and the reason is not reassuring.** `webData2`
appears nowhere in executable code — `grep -rn webData2` finds only comments,
docs and this entry. The "shard planner" that every one of those comments
deferred to (*"non-blocking: the planner uses `webData2`"*) **is not in the
tree either**: `grep -rn shard` finds only the comments that invoke it. So
the sentence that made C4 safe to leave open was not a fact about running
code — it was a fact about a component nobody had written, and it happened to
be pointing at the broken one of the two.

**What changed in the checker.** `check_webdata3` now probes BOTH and reports
the comparison, because the one-sided question is what hid this: while it
asked only "does `webData3` exist?", every answer could be waved off with the
assumption about `webData2` that nobody tested. It also refuses to conclude
anything when run without `--address`, since that is precisely the evidence
that produced a wrong refutation below.

**For whoever builds the planner:** subscribe to `webData3` with a `user`,
and re-run `verify --probe-ws --address 0x…` first, because this is one
measurement on one day and the venue moved once already.

---

**The retracted reading, kept because the mistake is the instructive part.**
A 2026-08-03 probe was read as refuting `webData3`, and that reading was
wrong.
Recorded because the mistake is more instructive than the result. The
operator ran the bash snippet this entry used to print and got:

```
{"channel":"error","data":"Error parsing JSON into valid websocket request: {\"method\": \"subscribe\", \"subscription\": {\"type\": \"webData3\"}}"}
```

This entry's stated criterion was "an error response means it does not
exist", and that criterion is too coarse. `Error parsing JSON into valid
websocket request` is a complaint about the REQUEST, and it has two
explanations that the message does not distinguish:

  - the subscription type is unknown, or
  - a required field is missing. `webData2` is keyed on `user`, and the probe
    above sends no `user` at all.

The refutation was argued from "the envelope is the same shape as the
`trades` subscription E4 confirmed live, so only the `type` value differs".
That is false on inspection: `trades` carries a second field (`coin`), and
this payload carries none. A payload with a missing required field and a
payload with an unknown type both fail to parse, and this venue's parse error
does not say which.

**The register row and the probe were consistent, and the register was the
more likely reading — as the table above then confirmed.** It records
`verify --probe-ws` acknowledging the subscription on 2026-07-31, and
`_probe_subscription` attaches `"user": address` when an address is supplied.
`webData3` existing and requiring `user` explained both observations at once;
"it does not exist" explained one and contradicted the other. Overturning the
better-supported reading on weaker evidence was the actual error, and the
lesson is cheaper than the finding: the control experiment cost thirty
seconds and was skipped because the wrong answer looked conclusive.

**The control that settled it** was to send `webData2` — assumed to exist —
with no `user`. It produced the same parse error, which is what made the bare
probe uninformative. The follow-up, `webData2` WITH a `user`, is the one
nobody had thought to run, and it is where the real finding was.
`scripts/probe_webdata.py` runs all four.

Note that `check_webdata3` returns PASS for both answers — existence and
non-existence — because C4 is non-blocking and a rejection is a real answer.
That is right, and it is also how a reader who scans statuses rather than
detail text can come away believing the opposite of what was measured. This
entry's contradiction lasted one commit; that property is permanent.

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

### C6 `[RESOLVED 2026-08-03]` Shadow cron vs. rate limit (§3.3 vs §5.3)

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
- *Fixed (2026-08-03), the residual the fix above left behind:* the
  verification harness. The two-process arithmetic was corrected for the two
  processes that are *scheduled*, and `market/verify.py` — run by hand, so
  never counted — kept `InfoClient()`'s default budget, which is the
  interactive one: all 1200/min, no reserve. A verification run launched
  while the stack was up therefore published 900 serving plus 300 shadow plus
  1200 here on one egress IP, a worse combined ceiling (2400) than the one
  this entry had just fixed (1500). It was found by reading, not by an
  incident, which is the point: a ceiling nobody reaches is invisible from
  inside, and the operational answer at the time — "stop the shadow
  containers before verifying" — was a precondition nothing enforced.
  Now the harness takes the background reserve and, when `$SHADOW_DSN` (or
  `--journal`) reaches the calibration database, charges the SAME ledger the
  sweep and the resolver share, as `actor='verify'`. Two consequences worth
  stating:
    - *it is paced.* Its 300/min share is less than one run costs — a
      50-address frame sweep alone is 1000 weight — so an unpaced charge
      would have turned the reserve into UNCHECKABLE results about requests
      that never left the process, which is precisely the defect
      `_is_self_inflicted` documents having already happened once. `charge`
      now waits out a spent window (`PacedBudget` in `market/info.py`),
      bounded at 10 minutes so a pool somebody is genuinely holding still
      surfaces. A full run costs about four minutes instead of one. This is
      the *fourth* place "a rate limit is a pace, not an error" had to be
      fixed (sweep, resolver, bundle build, harness); wrapping the budget
      rather than the caller is what stops there being a fifth.
    - *the fallback is loud.* An unreachable DSN, a sqlite journal or a
      missing psycopg all mean this run cannot see the other jobs' spending,
      and none of them is a reason to refuse to verify — so it falls back to
      a private 300/min window and prints which pool it actually joined.
      Believing you share a pool while holding a private one is the C6 defect
      itself; it must not be recoverable by staying quiet.
  The §5.3 weight constants were the last thing outstanding here, and the
  table below closes them against the published page. What this entry cannot
  close by reading is whether the venue enforces what it publishes; the
  client handles a 429 either way, and over-charging is no longer the
  blanket description of our error.

**The weight table, read from the primary source 2026-08-03** — and the
secondary-source note this replaces had it half right, in the reassuring
half. That note said `clearinghouseState` was probably weight 2 rather than
20, judged the error safe because over-charging self-limits harder than the
venue asks, and left the flat rate alone. The published page
(`.../for-developers/api/rate-limits-and-user-limits`) confirms the 2 — and
shows the flat rate is wrong in the OTHER direction too:

| request | published | this build charged | direction |
|---|---|---|---|
| `clearinghouseState` | **2** | 20 | over — safe, and 10x the sweep's real cost |
| `l2Book`, `allMids`, `orderStatus`, `spotClearinghouseState`, `exchangeStatus` | 2 | 20 | over |
| `userRole` | 60 | 20 | **under** (not called here) |
| `candleSnapshot` | 20 **+1 per 60 items** | 20 | **under** |
| `fundingHistory`, `userFunding`, `userFills`, … | 20 **+1 per 20 items** | 20 | **under** |
| everything else documented | 20 | 20 | correct |

**The under-charges are the finding.** A 90-day hourly candle snapshot is
2160 items — 20 + 36 = ~56 weight, charged as 20. A 30-day funding history is
720 items — also ~56, charged as 20. The bundle build fetches one of each per
coin, so on a 3-coin universe it spent ~356 weight while recording 140.
Against a 300/minute shadow reserve that is not a rounding error: the build
alone overshot the pool and ate into the share §5.3 sets aside for
interactive users, and **nothing could observe it**, because the budget only
ever knew what it charged itself. The serving engine did the same on every
five-minute rebuild.

Same shape as the per-process reserve this entry already fixed, one level
down: an accounting error invisible from inside the accounting.

**Taken.** `INFO_REQUEST_WEIGHTS` gives each type its published weight, and
`info_response_surcharge` bills the per-item extra once the response is in
hand — it cannot be known before, being a function of how many items came
back. That charge is *recorded, never refused* (`charge_incurred`): the
request is already on the wire and the venue has already counted it, so
refusing would discard a paid-for response while pretending it was free would
understate the window. It overshoots and the next charge waits — one request
late, which is the best available answer for a cost nobody can know in
advance.

**One reading had to be chosen, and the deployment settled it.** "An
additional rate limit weight per 20 items returned" admits +1 per 20 items or
+20 per 20 items. It is +1: under the alternative a single 90-day
`candleSnapshot` would cost 2160 weight against a 1200/minute limit and could
never succeed from any process, and this build has made exactly that call on
every rebuild for weeks without a 429.

**What it changes, now that it is confirmed rather than rumoured:**

- B1's reachable-window table. §3.3's floor of 200 addresses was quoted at
  8000 weight / 27 minutes; at weight 2 it is 400 weight and **under two
  minutes**. The sweep was never the constraint it was sized against.
- This entry's own case for sharing the pool rather than halving it, which
  argued "150/min would put 200 addresses at 53 minutes, past the resolver's
  50-minute ceiling". False now by an order of magnitude. The shared ledger
  is still right — two processes that cannot see each other's spending is a
  correctness problem at any weight, and the per-item surcharge is a second
  reason — but this justification for it is retired rather than reused.
- B6's universe scale, which is budget-constrained. Each added coin costs one
  candle snapshot plus one funding history per rebuild: ~112 weight, not the
  ~40 previously quoted. Widening is far cheaper than the old sweep
  arithmetic implied and dearer than the old per-coin figure did.

**One websocket limit worth carrying to whoever builds the shard planner**,
from the same page: *maximum of 10 unique users across user-specific
websocket subscriptions*. `webData3` is user-keyed (C4), so a planner
following C4's finding cannot watch more than 10 addresses per IP by
subscription — against §3.3's 200 a day. A design constraint on a component
that does not exist yet, recorded so it is met at design time rather than
discovered at address 11.


### C7 `[REFUTED 2026-08-03]` Is the Info API case-sensitive on `user`?

**Settled, and the original claim was wrong.** The repro below was run from
the operator's own machine against the same address the 2026-07-30 commit
named:

```
200 0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed
200 0x5aaeb6053f3e94c9b9a09f33669435e7ef1beaed
```

Two 200s, which this entry's own pre-registered criterion reads as
refutation. The venue is case-insensitive on `user`; the 422 that motivated
the original claim was something else — a malformed request in that one
invocation, or a transient response the second attempt happened to clear —
and is now unrepeatable evidence for anything, exactly the status this entry
gave it before the repro ran.

**Nothing built on the claim needs to change.** The section below already
established this in advance of the result: `normalise_address` defends two
properties of *this codebase* — the §5.1 silent-empty-state footgun and the
journal's case-sensitive `UNIQUE` constraint — neither of which depends on
what the venue does with case. Both stand exactly as written. The rest of
this entry is kept rather than deleted, because it separately documents a
real defect class (an unrepeated observation without a captured command line
being stated as protocol fact) that is worth a future reader seeing regardless
of which way this particular claim resolved.

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

Settled from a network where the API is reachable, by
`scripts/check_case_sensitivity.py` — a script rather than a shell one-liner
because the bash version printed here originally could not run on the
operator's PowerShell at all, and a repro nobody can execute settles nothing.
Two different status codes would confirm case sensitivity; two 200s refute
it. The result is recorded at the top of this entry.

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

### D3 `[RESOLVED]` "Effective leverage" is unsigned (§4.1)

§4.1 defines it as `sigma(equity 24h) / sigma(BTC 24h)` and the UI string was
"your book moves like BTC with leverage X". Those did not match: the ratio of
volatilities is unsigned and direction-free, so a market-neutral book with
large idiosyncratic variance read as "BTC with leverage 3" while having no
BTC exposure at all — and a short book read identically to a long one.

**Shipped 2026-08-02** (`apps/web/app/page.tsx`, panel 2). The unsigned ratio
now carries its own caveat verbatim in the UI — "This ratio has no direction:
a market-neutral book with large idiosyncratic variance scores the same as an
outright long" — displayed beside a *separately labelled* signed figure,
"Beta to {factor_coin} — the number that carries direction", with an explicit
long/short readout gated on `direction_detectable`. Nothing in the original
entry was left open to confirm: it named the fix (return the ratio and the
beta; do not let the ratio's copy claim direction) and the shipped copy is
that fix, not a paraphrase of it.

### D4 `[RESOLVED — confirmed 2026-08-03]` §6's 60-second staleness rule vs §2.1's 5-minute rebuild

The global correlation matrix is rebuilt every 5 minutes by design. If the
60-second staleness rule applied to it, every result would be permanently
stale and the execution button permanently disabled. The two clocks are
therefore separated: the 60 s / 5 min contract governs the *book and mark
price* inputs (WebSocket-driven, sub-second in normal operation), and matrix
age carries its own longer threshold (`MATRIX_HIDE_AFTER_MS = 960_000`),
exposed separately on the health endpoint as §6 requires.

**Confirmed by the specification owner on 2026-08-03**, asked because this is
the one rule §6 says not to soften for conversion — so a split adopted on
engineering grounds alone would have been exactly the softening it forbids.
What the confirmation covers is the *partition*, not a relaxation: neither
clock was widened, and each input is still judged against its own. The
degradation contract takes the worst verdict across all inputs
(`guardInputs()`), so a stale book still blocks execution however fresh the
matrix is.

Worth restating why the split is not a loophole. A 16-minute matrix threshold
sounds permissive next to 60 seconds until you note it is barely three
rebuild cycles: it fires when the rebuild loop has *failed*, which is the
condition it exists to detect, and it cannot fire in normal operation because
normal operation refreshes it every 5 minutes. The 60-second rule is doing
the opposite job — catching a feed that stopped ticking seconds ago — and one
threshold cannot do both without either blocking permanently or never firing.

### D6 `[RESOLVED — confirmed 2026-08-03]` §4.2's overlap rule is the wrong significance test

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
rather than silently resolved.

**Confirmed by the specification owner on 2026-08-03**: the paired test is
the one the UI acts on. Note what the confirmation does *not* do — it does
not delete §4.2's rule from the output. `marginal_intervals_overlap` is still
computed and returned, because the disagreement counter is only meaningful
while both readings exist, and a departure from the specification that erases
the thing it departed from cannot be audited afterwards.

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

**float32 path generation: DECIDED 2026-08-04 — declined, and not on the
precision argument.** It was the last entry on the "before the shadow clock
starts" list that was a decision rather than a measurement, so it was blocking
the window from starting. It turns out not to be a trade at all: the saving
does not exist.

The claim was "roughly a 2x saving". Measured on the fixture universe at
20 000 paths, path generation is **26-31% of a request**, so Amdahl bounds any
generator-only change:

| positions | request | generation | generator 2x faster | generator FREE |
|---|---|---|---|---|
| 2 | 425.7 ms | 131.6 ms (30.9%) | 1.18x | 1.45x |
| 4 | 523.8 ms | 134.8 ms (25.7%) | 1.15x | 1.35x |

A 2x request-level saving is unreachable *however* the generator is written.
And the decisive number: 523.8 ms / 1.35 = **388 ms**, so even a generator
that cost nothing at all still misses §2.6's 300 ms budget. The precision
would have bought nothing.

Measured directly as well, transcribing the generator with every intermediate
in float32: **0.96x — 4% slower**, because the quantile map dominates and runs
off a float64 table. The saving is not merely bounded, it is absent.

The precision cost, measured on the same paths for completeness, is small:
relative error in the terminal price after the 24-step cumulative sum is
3.7e-8 median / 1.9e-7 max, the signed mean is 1.2 SE from zero (so the
rounding has no measurable direction), and over 24 random books P(liq) and
VaR@95 were **unchanged on every one**. So this is not a §10 refusal — §10
would have permitted it. It is refused because it is a cost with no benefit.

What this does NOT fix: §2.6 is still missed by 1.4-1.8x at the book size §0
targets, and the binding cost is the ~74% of a request that is not path
generation. Reported rather than worked around, per §9.

Still open, none taken unilaterally because they trade against things the
specification cares about:
- production hardware, which this shared build container is not;
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
| C4 | ~~Does the `webData3` subscription exist?~~ **RESOLVED 2026-07-31**: it does. `verify --probe-ws` opened `wss://api.hyperliquid.xyz/ws`, sent the subscribe, and the venue acknowledged it. Note what this does and does not establish — the subscription **exists**; nothing here says what it carries or that it is preferable to `webData2`, which §5.2 specifies. **Superseded 2026-08-03**: probing BOTH with a `user` found `webData3` accepted and `webData2` REJECTED in the same payload shape, so the alternative this row treats as the safe default is the one that does not work. Suitability — what `webData3` carries — is still not answered. See C4. This check was UNCHECKABLE with the reason "this harness speaks only the Info POST API", which stopped being true when `collect_addresses` shipped and held a live socket to the same venue for 1 110 frames. | closed; see C4 for the 2026-08-03 inversion |
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

**The SHORT case is now measured too (same day).** A live isolated short was
found by scanning the sampling frame (index 4 of 515):
`szi=-0.00024, marginUsed=0.623304, uPnL=0.24828, rawUsd=15.665304`. Both
predictions hold exactly:

- cash sign: `rawUsd == marginUsed + positionValue` for a short, i.e. the
  general form is `marginUsed - sign(size)*positionValue`. A long BORROWS
  dollars to hold the asset (negative cash); a short HOLDS dollars against an
  asset it owes (positive cash, larger than the pocket). The long form alone
  does NOT generalise -- a first version of the regression test asserted it
  for both sides and failed, which is the test doing its job;
- the fix: collateral `0.375024`, and reconstructing §1.1 from it gives
  `liquidationPx = 64466.271605` against the venue's `64466.2716049383` --
  agreement to ten significant figures.

That also quantifies the near-miss: `rawUsd/collateral ~= 42x`. A short would
have parsed silently at forty-two times its true margin.

Both responses are now regression tests (`TestLiveIsolatedMargin`), verbatim,
with the venue's `liquidationPx` as the oracle rather than numbers anyone
typed. E6 is closed on both sides.


| E5 | ~~Live API access — every §5.1 parser written against fixtures, never exercised against the live schema~~ **RESOLVED 2026-07-29** by `python -m risk_engine.market.verify` run from a network where the API is reachable (it is still 403 at the build-environment proxy). All three parsers PASS on live mainnet: `meta` → 177 assets, 34 with multiple margin tiers; `candleSnapshot` → 720 hourly BTC returns, 0 gaps, hourly vol 0.00363; `clearinghouseState` → 10 positions, cross collateral $3,957,459.72. The documented response shapes were correct **for those three reads, and NOT for `clearinghouseState`'s isolated-margin fields** — see E6, opened 2026-08-03: `leverage.rawUsd` is not the pocket's collateral, and reading it as such dropped whole live accounts. E5's PASS was real but shallow: a parser that RETURNS a book is not a parser that returns the RIGHT book, and nothing compared the parsed numbers against the venue's own `liquidationPx`. | closed, but see E6 |
| E6 | Historical liquidation frequencies by nominal leverage for Baseline A's `P(liq)` arm (B3) | Baseline A's probability output |
