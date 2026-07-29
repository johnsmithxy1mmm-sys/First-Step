"""How much of a day's VaR breaches are one event (OPEN-QUESTIONS B1).

`validation/power.py` measured what the shadow window has to be for §0.3's
criterion to mean anything, and found the answer turns almost entirely on one
number: the intra-day correlation of breach indicators. At breach-ICC 0.05
the specified 21 days is roughly enough; at 0.20 it needs 60; at 0.40, 180.
Nothing in the specification supplies that number and no simulation can
invent it — it is a property of how the market moves, and it takes data.

**A pilot for this does not burn the §3.3 counter.** Changing the
distribution resets the validation window (§3.3, §10), which is why the real
counter must not start until A1, A8, C1, C2 and C5 are settled. This is
different: what makes two addresses breach together on one day is the common
market move, not the model version. A new version shifts the VaR levels; it
does not change whether BTC fell 8% that day. The estimate survives a version
change to first order, so a fortnight spent measuring it is not thrown away.

## Why this does not estimate the breach ICC directly

The obvious estimator — ANOVA moments on the breach indicators — is
unbiased and useless at pilot length. Measured: at 14 days x 200 addresses
the standard deviation of the estimate across datasets is 0.117 on a true
value of 0.20, and an interval with correct coverage spans roughly [0, 0.85].
Thirty days barely improves it. That is not a defect of the estimator; it is
that a breach is a 5% event, so a day of 200 addresses carries about ten
breaches, and ten events say very little about how correlated they were.

Two intervals were tried and rejected before this one, both measured rather
than assumed:

  - the day-clustered percentile bootstrap covers 43% of the time at 14 days
    against a nominal 95%, because resampling 14 clusters cannot see the
    sampling variability of a variance-component ratio;
  - the normal-theory F interval covers 66% at breach-ICC 0.20 and 50% at
    0.40, because at a 5% base rate the between-day sum of squares is far
    from chi-square.

Both fail *low*, which would size the window too short — §10's forbidden
direction reached by arithmetic. `breach_icc_confidence_set` therefore
inverts the test against the validated generator instead, which covers
correctly (92-100%) and is honest about being wide.

## What it estimates instead

The PIT values are available for *every* observation, not just the 5% that
breach, and they carry the same co-movement. Standardising them to
`Phi^-1(PIT)` puts them on a latent Gaussian scale where a day's common
shock is an ordinary intra-class correlation, estimable from all 2 800
observations rather than from ten breaches. Measured, that estimator recovers
a true latent 0.30 as 0.3044 with sd 0.091, against sd 0.079 on a breach ICC
of 0.106 — roughly 2.5x the relative precision.

`breach_icc_from_latent` then maps it back: two addresses breach together
when both latent values fall below the 5% quantile, which is a bivariate
orthant probability. The map compresses — measured, both columns:

    latent rho   breach ICC (gaussian)   breach ICC (t, df=4 — the default)
      0.00            0.000                   0.077
      0.05            0.012                   0.095
      0.15            0.041                   0.130
      0.30            0.098                   0.195
      0.50            0.204                   0.301

The copula is the assumption, and it is stated rather than hidden. The
engine simulates a t-copula (§2.3) — one chi-square mixing draw shared
across assets per step (`sim/paths.py`) — so `copula="t"` is the default,
and it carries a consequence the Gaussian column hides: **a floor**. Under
a shared-mixing t, zero correlation is not independence; a volatile day
inflates everyone together, so the breach ICC is ~0.077 at rho=0. The floor
was checked against reality's stand-in, not just the map: simulating actual
shared-mixing t days at rho=0 realises an empirical breach ICC of 0.079.
Two things follow. A pilot measuring a tiny latent correlation still sizes
a substantial window — that is the tail dependence talking, and it is
correct, not conservatism stacked on top. And the map's low end cannot
distinguish quiet markets from moderately clustered ones; the latent
interval printed alongside is what preserves that distinction.

The composition — Gaussian-scores ANOVA feeding a t-parameterised map — was
measured against data generated from the true shared-mixing t world rather
than assumed compatible. Sixteen replications per rho: the mean bias is
within +/-0.007 across rho 0 to 0.5 (standard errors 0.003-0.012), i.e.
approximately unbiased, with per-pilot scatter up to +/-0.05 at high rho.
The scatter is real sampling noise and it is what the *interval* absorbs;
sizing reads ci_high, not the point. `test_clustering.py` pins the
near-unbiasedness so a refactor cannot silently push it in the direction
that shortens the window.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

#: §0.3's target breach rate, and the quantile the map is taken at.
NOMINAL_BREACH_RATE = 0.05
#: Matches the copula the engine actually simulates (§2.3, A9).
DEFAULT_COPULA_DF = 4.0


def _icc_from_counts(sums: np.ndarray, counts: np.ndarray) -> tuple[float, float]:
    """(raw ICC, effective cluster size) by one-way ANOVA moments, for 0/1 data.

    Raw: it may be negative, and clamping belongs to the caller so the sign
    stays visible. A negative value means the days varied *less* than
    binomial sampling alone would give.
    """
    keep = counts > 0
    c = counts[keep].astype(np.float64)
    s = sums[keep].astype(np.float64)
    k = c.size
    total = float(c.sum())
    if k < 2 or total <= k:
        return float("nan"), float("nan")
    rates = s / c
    grand = float(s.sum()) / total
    # y is 0/1, so the within-cluster sum of squares collapses to n_i p_i(1-p_i).
    msw = float((c * rates * (1.0 - rates)).sum()) / (total - k)
    msb = float((c * (rates - grand) ** 2).sum()) / (k - 1)
    n0 = (total - float((c**2).sum()) / total) / (k - 1)
    denominator = msb + (n0 - 1.0) * msw
    if denominator <= 0:
        return 0.0, n0
    return (msb - msw) / denominator, n0


def _icc_continuous(values: np.ndarray, day_index: np.ndarray, n_days: int) -> tuple[float, float]:
    """One-way ANOVA intra-class correlation for continuous values."""
    counts = np.bincount(day_index, minlength=n_days).astype(np.float64)
    sums = np.bincount(day_index, weights=values, minlength=n_days)
    keep = counts > 0
    counts, sums = counts[keep], sums[keep]
    k = counts.size
    total = float(counts.sum())
    if k < 2 or total <= k:
        return float("nan"), float("nan")
    means = sums / counts
    grand = float(sums.sum()) / total
    wss = float(((values - means[day_index]) ** 2).sum())
    bss = float((counts * (means - grand) ** 2).sum())
    msw = wss / (total - k)
    msb = bss / (k - 1)
    n0 = (total - float((counts**2).sum()) / total) / (k - 1)
    denominator = msb + (n0 - 1.0) * msw
    if denominator <= 0:
        return 0.0, n0
    return (msb - msw) / denominator, n0


@dataclass(frozen=True, slots=True)
class LatentCorrelation:
    """Intra-day correlation on the `Phi^-1(PIT)` scale, with its interval."""

    point: float
    ci_low: float
    ci_high: float
    n_days: int
    n_observations: int
    effective_cluster_size: float
    raw: float
    #: PIT values at exactly 0 or 1, clipped before the probit. A few are
    #: rounding; a lot means the predictive distribution is misfitted, and
    #: that has to be visible rather than silently absorbed into the clamp.
    n_clipped: int = 0

    @property
    def clipped_fraction(self) -> float:
        return self.n_clipped / self.n_observations if self.n_observations else 0.0

    def summary(self) -> str:
        base = (
            f"latent intra-day correlation {self.point:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}]\n"
            f"  from {self.n_observations} PIT values over {self.n_days} days"
        )
        if self.n_clipped:
            base += (
                f"\n  {self.n_clipped} PIT values ({self.clipped_fraction:.1%}) were "
                "degenerate (exactly 0 or 1) and clipped. More than a percent or so "
                "means the predictive distribution is misfitted -- fix that before "
                "trusting this estimate."
            )
        return base


def _simulated_icc_draws(
    rho: float, counts: np.ndarray, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """ANOVA ICC draws for the Gaussian latent world with these EXACT day sizes.

    The day sizes are the real ones, not their mean. This matters and was
    measured before being fixed: simulating equal days at the mean size gave
    the confidence set 80% coverage against a nominal 95% on lumpy designs
    (20/400 alternating), and it failed *low* — the ceiling the window is
    sized from came out too small. Lumpy days are not a corner case here:
    the shadow cron's weight-budget governor truncates sweeps by design
    (§5.3), so real pilots produce exactly this shape.

    Simulated through sufficient statistics rather than raw observations.
    Conditional on the day factors, the day means and the pooled within-day
    sum of squares are independent:

        m_j  = sqrt(rho) f_j + sqrt(1-rho) u_j / sqrt(n_j),   f, u ~ N(0,1)
        WSS  = (1-rho) * chi2(N - k)

    which is exact for the Gaussian latent world and makes the cost per draw
    O(days) instead of O(observations).
    """
    c = np.asarray(counts, dtype=np.float64)
    c = c[c > 0]
    k = c.size
    total = float(c.sum())
    factor = rng.standard_normal((n_sims, k))
    mean_noise = rng.standard_normal((n_sims, k)) / np.sqrt(c)
    m = np.sqrt(rho) * factor + np.sqrt(1.0 - rho) * mean_noise
    wss = (1.0 - rho) * rng.chisquare(total - k, size=n_sims)
    grand = (m * c).sum(axis=1) / total
    bss = (c * (m - grand[:, None]) ** 2).sum(axis=1)
    msw = wss / (total - k)
    msb = bss / (k - 1)
    n0 = (total - float((c**2).sum()) / total) / (k - 1)
    denominator = msb + (n0 - 1.0) * msw
    return np.where(denominator > 0, (msb - msw) / np.maximum(denominator, 1e-300), 0.0)


def _latent_confidence_set(
    observed: float,
    counts: np.ndarray,
    rng: np.random.Generator,
    grid: np.ndarray,
    n_sims: int,
    alpha: float,
) -> tuple[float, float]:
    """Invert the test: candidates that could have produced `observed`.

    Built for the same reason the breach version is. The day-clustered
    bootstrap on this statistic covers about 75% at 14 days against a
    nominal 95% -- much better than the 43% it manages on breach indicators,
    but still failing *low*, and a ceiling that is too low shortens the
    window. This construction has correct coverage because the acceptance
    region is computed under each candidate rather than resampled from one
    dataset, and because each region is simulated with the pilot's actual
    day sizes (see `_simulated_icc_draws`).
    """
    included: list[float] = []
    for candidate in grid:
        rho = float(np.clip(candidate, 0.0, 0.999))
        draws = _simulated_icc_draws(rho, counts, n_sims, rng)
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        if lo <= observed <= hi:
            included.append(rho)
    if not included:
        return 0.0, float(grid[-1])
    return min(included), max(included)


def estimate_latent_correlation(
    pit: np.ndarray,
    days: np.ndarray,
    rng: np.random.Generator | None = None,
    n_boot: int = 2_000,
    alpha: float = 0.05,
    clip: float = 1e-6,
    method: str = "inverted",
    grid: np.ndarray | None = None,
) -> LatentCorrelation:
    """Intra-day correlation of `Phi^-1(PIT)`.

    Uses every observation rather than the 5% that breach, which is what
    makes it estimable at pilot length at all.

    `method="inverted"` (the default) builds the interval by inverting the
    test against the latent factor model, which has correct coverage.
    `method="bootstrap"` is the day-clustered percentile bootstrap; it is
    kept because it is assumption-free about the *shape* of the day effect,
    but it covers about 75% at 14 days against a nominal 95%, and it fails
    low, so it must not be used for sizing.
    """
    p = np.asarray(pit, dtype=np.float64)
    d = np.asarray(days)
    if p.size != d.size:
        raise ValueError("PIT and day arrays must be the same length")
    if p.size == 0:
        raise ValueError("no observations")
    unique_days, day_index = np.unique(d, return_inverse=True)
    n_days = unique_days.size
    if n_days < 2:
        raise ValueError(
            f"need at least two days to estimate an intra-day correlation; got {n_days}"
        )

    # A PIT of exactly 0 or 1 is an infinity on the latent scale. Clipping is
    # the standard handling and the bias is negligible at these clip levels,
    # but a *lot* of clipped values means the distribution is misfitted, not
    # that the correlation is high -- so it is counted and surfaced.
    n_clipped = int((p <= clip).sum() + (p >= 1.0 - clip).sum())
    z = stats.norm.ppf(np.clip(p, clip, 1.0 - clip))

    raw, n0 = _icc_continuous(z, day_index, n_days)
    if not np.isfinite(raw):
        raise ValueError("too few observations per day to estimate a correlation")
    point = float(min(max(raw, 0.0), 1.0))

    rng = rng or np.random.default_rng(0)
    if method == "inverted":
        if grid is None:
            grid = np.round(
                np.concatenate([np.arange(0.0, 0.40, 0.02), np.arange(0.40, 1.0, 0.05)]), 3
            )
        counts = np.bincount(day_index, minlength=n_days)
        lo, hi = _latent_confidence_set(
            point, counts, rng, grid, n_sims=max(n_boot // 2, 500), alpha=alpha
        )
    elif method == "bootstrap":
        groups = [z[day_index == i] for i in range(n_days)]
        draws = np.empty(n_boot)
        for t in range(n_boot):
            pick = rng.integers(0, n_days, size=n_days)
            values = np.concatenate([groups[i] for i in pick])
            index = np.concatenate([
                np.full(groups[i].size, j, dtype=np.int64) for j, i in enumerate(pick)
            ])
            draw, _ = _icc_continuous(values, index, n_days)
            draws[t] = draw if np.isfinite(draw) else np.nan
        finite = draws[np.isfinite(draws)]
        if finite.size < n_boot // 10:
            raise ValueError("bootstrap failed on most resamples")
        lo = float(np.clip(np.quantile(finite, alpha / 2), 0.0, 1.0))
        hi = float(np.clip(np.quantile(finite, 1 - alpha / 2), 0.0, 1.0))
    else:
        raise ValueError(f"unknown method {method!r}; use 'inverted' or 'bootstrap'")
    return LatentCorrelation(
        point=point,
        ci_low=min(lo, point),
        ci_high=max(hi, point),
        n_days=n_days,
        n_observations=int(p.size),
        effective_cluster_size=float(n0),
        raw=float(raw),
        n_clipped=n_clipped,
    )


def breach_icc_from_latent(
    rho: float,
    p: float = NOMINAL_BREACH_RATE,
    copula: str = "t",
    df: float = DEFAULT_COPULA_DF,
    n_mc: int = 400_000,
    seed: int = 0,
) -> float:
    """Breach-indicator ICC implied by a latent intra-day correlation.

    Two addresses breach together when both latent values fall below the `p`
    quantile, so the joint rate is an orthant probability and

        ICC = (P(both breach) - p^2) / (p(1 - p)).

    `copula="gaussian"` is the smallest defensible answer. `copula="t"` is
    the default because it is what the engine simulates (§2.3): tail
    dependence makes joint breaches more likely at the same correlation, so
    it maps to a higher ICC and a longer window. Between two stated
    assumptions, §10 permits the one that does not shorten validation.

    The t map has a FLOOR: at rho=0 it returns ~0.077, not 0, because a
    shared-mixing t is not independent at zero correlation — a fat-tailed
    day inflates every address at once. This matches what actual t-world
    simulation realises (0.079 measured at rho=0), so it is the model's
    physics, not an artifact. Consequence: under the default map the low
    end is dominated by tail dependence rather than by the measured
    correlation, and two pilots with latent 0.03 and 0.15 can size to the
    same window. Read the latent interval when the distinction matters.
    """
    rho = float(np.clip(rho, 0.0, 0.999))
    if copula == "gaussian":
        z = stats.norm.ppf(p)
        joint = float(
            stats.multivariate_normal.cdf([z, z], mean=[0.0, 0.0], cov=[[1.0, rho], [rho, 1.0]])
        )
    elif copula == "t":
        # No closed form worth trusting here; the orthant probability is
        # simulated from the same law the engine draws paths from.
        rng = np.random.default_rng(seed)
        chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
        normals = rng.standard_normal((n_mc, 2)) @ chol.T
        scale = np.sqrt(df / rng.chisquare(df, size=(n_mc, 1)))
        t_draws = normals * scale
        threshold = stats.t.ppf(p, df)
        joint = float(((t_draws[:, 0] < threshold) & (t_draws[:, 1] < threshold)).mean())
    else:
        raise ValueError(f"unknown copula {copula!r}; use 'gaussian' or 't'")
    return float(np.clip((joint - p * p) / (p * (1.0 - p)), 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class BreachIcc:
    point: float
    ci_low: float
    ci_high: float
    latent: LatentCorrelation
    copula: str
    #: Direct ANOVA estimate on the breach indicators, for comparison. No
    #: interval: see the module docstring on why one is not cheap here.
    direct_point: float | None
    breach_rate: float
    n_observations: int
    n_days: int

    @property
    def design_effect(self) -> float:
        return 1.0 + (self.latent.effective_cluster_size - 1.0) * self.point

    @property
    def effective_sample_size(self) -> float:
        return self.n_observations / max(self.design_effect, 1e-12)

    def summary(self) -> str:
        direct = (
            f"{self.direct_point:.3f}" if self.direct_point is not None else "n/a"
        )
        out = (
            f"breach ICC {self.point:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}]  "
            f"({self.copula}-copula map)\n"
            f"  latent correlation {self.latent.point:.3f} "
            f"[{self.latent.ci_low:.3f}, {self.latent.ci_high:.3f}]\n"
            f"  direct estimate    {direct} "
            f"(unstable at this length; shown as a sanity check, not for sizing)\n"
            f"  breach rate        {self.breach_rate:.4f}\n"
            f"  design effect      {self.design_effect:.1f}x\n"
            f"  effective n        {self.effective_sample_size:.0f} "
            f"(the naive interval assumes {self.n_observations})"
        )
        if self.latent.n_clipped:
            out += (
                f"\n  WARNING: {self.latent.n_clipped} PIT values "
                f"({self.latent.clipped_fraction:.1%}) were degenerate (0 or 1). "
                "That is a misfitted predictive distribution, not clustering."
            )
        return out


def estimate_breach_icc(
    pit: np.ndarray,
    breached: np.ndarray,
    days: np.ndarray,
    rng: np.random.Generator | None = None,
    n_boot: int = 2_000,
    copula: str = "t",
    df: float = DEFAULT_COPULA_DF,
) -> BreachIcc:
    """Breach ICC via the latent scale, with the direct estimate alongside."""
    latent = estimate_latent_correlation(pit, days, rng, n_boot=n_boot)

    b = np.asarray(breached, dtype=np.float64)
    d = np.asarray(days)
    _, day_index = np.unique(d, return_inverse=True)
    n_days = int(day_index.max()) + 1
    counts = np.bincount(day_index, minlength=n_days)
    sums = np.bincount(day_index, weights=b, minlength=n_days)
    direct_raw, _ = _icc_from_counts(sums, counts)
    direct = float(min(max(direct_raw, 0.0), 1.0)) if np.isfinite(direct_raw) else None

    def to_breach(r: float) -> float:
        return breach_icc_from_latent(r, copula=copula, df=df)

    return BreachIcc(
        point=to_breach(latent.point),
        ci_low=to_breach(latent.ci_low),
        ci_high=to_breach(latent.ci_high),
        latent=latent,
        copula=copula,
        direct_point=direct,
        breach_rate=float(b.mean()),
        n_observations=int(b.size),
        n_days=latent.n_days,
    )


def breach_icc_confidence_set(
    breached: np.ndarray,
    days: np.ndarray,
    rng: np.random.Generator | None = None,
    grid: np.ndarray | None = None,
    n_sims: int = 200,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Assumption-free interval for the breach ICC, by inverting the test.

    Slow and wide, and both are the point. It makes no copula assumption:
    for each candidate ICC it simulates the estimator's distribution under
    the validated beta-binomial generator with the observed day sizes, and
    keeps the candidates that could plausibly have produced the observed
    estimate. Coverage is 92-100% against a nominal 95%, where the bootstrap
    manages 43%.

    Use it to see what the breach data alone supports — which at pilot
    length is very little — not to size a window.
    """
    from risk_engine.validation.power import simulate_days

    b = np.asarray(breached, dtype=np.float64)
    d = np.asarray(days)
    _, day_index = np.unique(d, return_inverse=True)
    n_days = int(day_index.max()) + 1
    counts = np.bincount(day_index, minlength=n_days)
    sums = np.bincount(day_index, weights=b, minlength=n_days)
    observed, _ = _icc_from_counts(sums, counts)
    if not np.isfinite(observed):
        raise ValueError("cannot estimate an ICC from these observations")

    rng = rng or np.random.default_rng(0)
    if grid is None:
        grid = np.round(
            np.concatenate([np.arange(0.0, 0.30, 0.02), np.arange(0.30, 0.95, 0.05)]), 3
        )
    per_day = round(float(counts.mean()))
    rate = float(b.mean())

    included = []
    for candidate in grid:
        draws = np.empty(n_sims)
        for m in range(n_sims):
            s2, c2 = simulate_days(n_days, per_day, rate, float(candidate), rng)
            draws[m] = _icc_from_counts(s2, c2)[0]
        finite = draws[np.isfinite(draws)]
        if finite.size < 2:
            continue
        lo, hi = np.quantile(finite, [alpha / 2, 1 - alpha / 2])
        if lo <= observed <= hi:
            included.append(float(candidate))
    if not included:
        return 0.0, float(grid[-1])
    return min(included), max(included)


@dataclass(frozen=True, slots=True)
class WindowRecommendation:
    icc_used: float
    basis: str
    addresses_per_day: int
    target_power: float
    detect_rate: float
    days_required: int | None
    searched: list[tuple[int, float]]

    def summary(self) -> str:
        head = (
            f"sizing off breach ICC {self.icc_used:.3f} ({self.basis}), "
            f"{self.addresses_per_day} addresses/day\n"
            f"  target: {self.target_power:.0%} power to detect a true breach rate of "
            f"{self.detect_rate:.0%} against the claimed 5%\n"
        )
        rows = "\n".join(
            f"    {days:4d} days   power {power:.0%}" for days, power in self.searched
        )
        if self.days_required is None:
            tail = (
                f"\n  no window in the searched grid reaches {self.target_power:.0%}. "
                "At this much clustering the VaR criterion is not the right gate; "
                "see OPEN-QUESTIONS B1 for the alternatives."
            )
        else:
            tail = f"\n  -> {self.days_required} days"
        return head + rows + tail


def recommend_window(
    icc: BreachIcc,
    addresses_per_day: int = 200,
    target_power: float = 0.8,
    detect_rate: float = 0.10,
    day_grid: tuple[int, ...] = (21, 30, 45, 60, 90, 120, 180),
    basis: str = "ci_high",
    n_trials: int = 200,
    n_boot: int = 400,
    seed: int = 0,
) -> WindowRecommendation:
    """Smallest window in `day_grid` reaching `target_power` at this ICC.

    Runs the power simulation at the measured value rather than reading B1's
    table, because the table was computed on a grid the measurement will not
    land on.

    Defaults to the upper end of the interval, not the point estimate. Sizing
    off the middle is wrong half the time in the direction that shortens the
    window, and a window that is too short yields a gate that passes without
    establishing anything.

    Power is simulated at the requested `detect_rate` itself, never snapped
    to a nearby tabulated one: a stricter target (closer to the nominal 5%)
    is *harder* to detect and needs a longer window, and quantising it to an
    easier column was measured to under-size the window (audit F-1).
    """
    from risk_engine.validation.power import NOMINAL_BREACH_RATE as NOMINAL
    from risk_engine.validation.power import power_at

    if not NOMINAL < detect_rate < 1.0:
        raise ValueError(
            f"detect_rate must be a true breach rate above the nominal {NOMINAL:.0%} "
            f"and below 1; got {detect_rate}. The gate exists to catch a model that "
            "understates tail risk, so the alternative worth power against is a "
            "rate above the claimed one."
        )
    value = {"point": icc.point, "ci_high": icc.ci_high, "ci_low": icc.ci_low}[basis]

    searched: list[tuple[int, float]] = []
    required: int | None = None
    for i, days in enumerate(day_grid):
        power = power_at(
            days, addresses_per_day, value, detect_rate, n_trials, n_boot, seed + i
        )
        searched.append((days, power))
        if required is None and power >= target_power:
            required = days
            break

    return WindowRecommendation(
        icc_used=value,
        basis=basis,
        addresses_per_day=addresses_per_day,
        target_power=target_power,
        detect_rate=detect_rate,
        days_required=required,
        searched=searched,
    )
