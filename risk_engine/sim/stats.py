"""Interval estimation and scoring rules.

§2.5 forbids ever returning a point estimate without an interval, and §4
makes that unconstructible in the type system. This module supplies the
intervals.

Wilson intervals are used for probabilities rather than the Wald
(`p +- z sqrt(p(1-p)/n)`) interval, because the quantity most often reported
here is a small liquidation probability, and Wald intervals around small p
famously extend below zero and undercover.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

Z95 = 1.959963984540054


def wilson_interval(successes: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Two-sided Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def wilson_half_width(successes: int, n: int, z: float = Z95) -> float:
    lo, hi = wilson_interval(successes, n, z)
    return 0.5 * (hi - lo)


def paths_needed_for_half_width(p: float, half_width: float, z: float = Z95) -> int:
    """A path count whose interval half-width at probability `p` meets the target.

    Sufficient, not provably minimal: the width depends on the integer
    success count, so rounding `p * n` makes it very slightly non-monotone in
    `n` and a handful of smaller counts may also fit. Sufficiency is the
    property that matters -- it is what stops the engine returning a number
    that violates §2.5.

    §2.5 requires the half-width on P(liq) to stay under 2 pp. §2.6 says to
    cut paths when the latency budget is missed. Those conflict, and this
    function is where the conflict is resolved in favour of §2.5: it returns
    the floor below which paths must not be cut, whatever the clock says
    (OPEN-QUESTIONS D1).
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be a probability")
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    p = min(max(p, 1e-4), 1 - 1e-4)

    # The normal approximation is a starting guess, not the answer: at small p
    # it understates n, and returning a count whose actual Wilson width still
    # exceeds the target would break the one rule §2.5 is unambiguous about.
    # The Wilson half-width is monotone decreasing in n, so double until it
    # fits and then bisect for the smallest n that does.
    def fits(n: int) -> bool:
        return wilson_half_width(round(p * n), n, z) <= half_width

    n = max(2, int(np.ceil(z * z * p * (1 - p) / (half_width * half_width))))
    if not fits(n):
        lo = n
        hi = n * 2
        while not fits(hi):
            lo, hi = hi, hi * 2
            if hi > 1 << 34:  # pragma: no cover - unreachable for sane targets
                raise ValueError(f"half_width {half_width} unreachable at p={p}")
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if fits(mid):
                hi = mid
            else:
                lo = mid
        n = hi
    return n


def bootstrap_ci(
    samples: np.ndarray,
    statistic,
    rng: np.random.Generator,
    n_boot: int = 400,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap interval for an arbitrary statistic."""
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n == 0:
        raise ValueError("cannot bootstrap an empty sample")
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([statistic(x[i]) for i in idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def value_at_risk(losses_or_pnl: np.ndarray, level: float = 0.95) -> float:
    """VaR as a positive loss number at `level` (0.95 -> the 5% worst tail)."""
    pnl = np.asarray(losses_or_pnl, dtype=np.float64)
    return float(-np.quantile(pnl, 1.0 - level))


def conditional_value_at_risk(pnl: np.ndarray, level: float = 0.95) -> float:
    """Mean loss conditional on being in the worst (1 - level) tail, positive."""
    x = np.asarray(pnl, dtype=np.float64)
    cutoff = np.quantile(x, 1.0 - level)
    tail = x[x <= cutoff]
    if tail.size == 0:  # pragma: no cover - only with a degenerate sample
        return float(-cutoff)
    return float(-tail.mean())


def tail_size(n: int, level: float = 0.95) -> int:
    """How many of `n` samples make up the worst (1 - level) tail."""
    return max(1, round((1.0 - level) * n))


def conditional_value_at_risk_rows(pnl: np.ndarray, level: float = 0.95) -> np.ndarray:
    """CVaR of every row of a 2-D sample, as positive losses.

    Exists for the bootstrap in `pre_trade_delta`, where the scalar version
    called in a Python loop dominated the §2.6 latency budget: 200
    replicates x 2 books was 166 ms of a 300 ms allowance, because each call
    fully sorted 20 000 numbers. `np.partition` finds the tail in linear time
    and does every replicate in one pass.

    Selects exactly the `tail_size(n)` smallest values, so it can differ
    marginally from the quantile-and-mask scalar version when the sample has
    ties on the cutoff. `test_pre_trade_delta.py` pins the two together.
    """
    x = np.asarray(pnl)
    if x.ndim != 2:
        raise ValueError(f"expected a 2-D sample, got {x.shape}")
    k = tail_size(x.shape[1], level)
    part = np.partition(x, k - 1, axis=1)[:, :k]
    # Accumulate in float64 even when the input is float32: the caller may
    # narrow the sample to make the partition cheaper, but the tail mean
    # itself should not lose digits.
    return -part.mean(axis=1, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class PredictiveDistribution:
    """A predicted distribution stored as a quantile function.

    Quantiles rather than raw samples because this is what goes into the
    calibration journal (§3.4): 1001 numbers per prediction is a row a
    database can hold for years, 20 000 is not, and every metric §3.3 asks
    for -- PIT, CRPS, VaR breach -- is computable from the quantile function.
    """

    levels: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        if self.levels.shape != self.values.shape:
            raise ValueError("levels and values must have the same shape")
        # Finiteness before monotonicity (audit A-07): NaN makes every
        # comparison False, so a NaN quantile function would otherwise pass
        # the non-decreasing check and poison the calibration journal.
        if not np.isfinite(self.values).all():
            raise ValueError("quantile values must be finite")
        if np.any(np.diff(self.values) < 0):
            raise ValueError("quantile function must be non-decreasing")

    @classmethod
    def from_samples(cls, samples: np.ndarray, n_levels: int = 1001) -> PredictiveDistribution:
        levels = np.linspace(0.0, 1.0, n_levels)
        return cls(levels=levels, values=np.quantile(np.asarray(samples, float), levels))

    def quantile(self, q: float | np.ndarray) -> float | np.ndarray:
        return np.interp(q, self.levels, self.values)

    def cdf_interval(self, x: float) -> tuple[float, float]:
        """(P(X < x), P(X <= x)) -- distinct exactly where the law has an atom.

        The liquidation model writes a wiped account to exactly zero equity
        (§1.6), so the predicted equity-change distribution carries a mass
        point at total loss. A single-valued CDF cannot represent that
        honestly; both one-sided limits can.
        """
        v, lv = self.values, self.levels
        il = int(np.searchsorted(v, x, side="left"))
        ir = int(np.searchsorted(v, x, side="right"))
        if il < ir:
            # x sits on a flat run of the quantile function: an atom.
            return float(lv[il]), float(lv[ir - 1])
        if il == 0:
            return 0.0, 0.0
        if il == len(v):
            return 1.0, 1.0
        t = (x - v[il - 1]) / (v[il] - v[il - 1])
        c = float(lv[il - 1] + t * (lv[il] - lv[il - 1]))
        return c, c

    def cdf(self, x: float) -> float:
        """Midpoint CDF; for atom-aware work use `cdf_interval` or `pit`."""
        lo, hi = self.cdf_interval(x)
        return 0.5 * (lo + hi)

    def pit(self, actual: float, u: float) -> float:
        """§3.3: randomized probability integral transform.

        Uniform on [0, 1] under a correctly calibrated model *including*
        models whose predicted distribution carries atoms. The naive
        `cdf(actual)` is not: every realised liquidation maps to the same
        deterministic value, and the KS test then rejects a perfectly
        calibrated model with certainty (audit A-02, reproduced at
        p = 1e-104). `u` must be an independent U(0,1) draw; the resolver
        derives it deterministically from the prediction id so every journal
        row stays reproducible.
        """
        if not 0.0 <= u <= 1.0:
            raise ValueError(f"u must be in [0, 1], got {u}")
        lo, hi = self.cdf_interval(actual)
        return lo + u * (hi - lo)

    def crps(self, actual: float) -> float:
        """Continuous ranked probability score, lower is better.

        Computed from the quantile function via the pinball identity
        `CRPS = 2 * integral_0^1 QL_tau dtau`, which is exact for this
        representation rather than a sample approximation of it.
        """
        q = self.values
        tau = self.levels
        pinball = np.where(actual < q, 1.0 - tau, -tau) * (q - actual)
        return float(2.0 * np.trapezoid(pinball, tau))

    def var(self, level: float = 0.95) -> float:
        return float(-self.quantile(1.0 - level))

    def cvar(self, level: float = 0.95) -> float:
        mask = self.levels <= (1.0 - level)
        if mask.sum() < 2:  # pragma: no cover - only for absurdly coarse grids
            return float(-self.quantile(1.0 - level))
        return float(-np.trapezoid(self.values[mask], self.levels[mask]) / (1.0 - level))


def ks_uniformity(pit_values: np.ndarray) -> tuple[float, float]:
    """Kolmogorov-Smirnov statistic and p-value against U(0, 1).

    The p-value assumes independent observations. Shadow observations on the
    same calendar day share one market and are strongly dependent, so this
    p-value is anti-conservative and must be read alongside the day-clustered
    version (OPEN-QUESTIONS B1).
    """
    x = np.asarray(pit_values, dtype=np.float64)
    res = stats.kstest(x, "uniform")
    return float(res.statistic), float(res.pvalue)


def clustered_bootstrap_ci(
    values: np.ndarray,
    cluster_ids: np.ndarray,
    statistic,
    rng: np.random.Generator,
    n_boot: int = 2000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Interval that resamples whole clusters, not individual observations.

    §0.3 and §3.3 ask for a binomial interval around the VaR breach rate.
    That interval assumes independent observations; 300 addresses observed on
    the same day are not independent, because one market move drives all of
    them. Resampling days keeps the dependence intact and is the interval the
    gate should actually be read from (OPEN-QUESTIONS B1).
    """
    v = np.asarray(values)
    ids = np.asarray(cluster_ids)
    uniq = np.unique(ids)
    if uniq.size < 2:
        raise ValueError("need at least two clusters to bootstrap over them")
    groups = [v[ids == u] for u in uniq]
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(groups), size=len(groups))
        out[b] = statistic(np.concatenate([groups[i] for i in pick]))
    lo, hi = np.quantile(out, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)
