"""Fast monotone map from a copula variate to a standardised marginal.

Path generation needs `t_ppf(t_cdf(x, nu_copula), nu_asset)` on every drawn
number. At 20 000 paths x 24 steps x 8 assets that is ~4M evaluations, and
scipy's `t.cdf`/`t.ppf` on that volume costs seconds — an order of magnitude
over the whole 300 ms pre-trade budget in §2.6.

The composition is a fixed monotone odd function of one variable for each
(source df, target df) pair, so it is tabulated once and evaluated by
interpolation. The table is built in *probability* space (log-spaced down to
1e-13) so the far tail is resolved, and interpolated in log-log space, where
the relationship is smooth and nearly linear.

Past the last node the continuation is done in log-survival space rather
than log-log space: `log y` is asymptotically linear in `u = -log S(x)`
(slope `1/nu` for a t target), and `u` is available exactly from scipy's
`logsf` for either source family. Log-log extrapolation would be correct for
a t source, whose survival is a power law, but badly wrong for a Gaussian
source, whose `u` grows like `x^2/2` — and the Gaussian source is exactly
what the independence baseline and several benchmarks use.

Accuracy is asserted against scipy in `tests/test_quantile_map.py` — this is
a numerical shortcut on the path that produces every risk number, so it
carries its own error budget.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

#: Deepest probability resolved by the table. With 4M draws the deepest
#: survival probability actually observed is ~1e-7, so 1e-13 leaves six
#: decades of headroom before extrapolation is used at all.
_MIN_TAIL_P = 1e-13
#: Accurate nodes, spaced geometrically in probability.
#:
#: 8000, not the original 3000. Probability spacing is coarse in log|x| near
#: the median -- `p` moves geometrically while x moves almost linearly in
#: (0.5 - p) -- so consecutive nodes there straddled decades of log|x|, and
#: the linear interpolation across that gap WAS the whole error budget:
#: 1.28e-5 at target df 2.1, against the 1e-5 this module asserts, peaking at
#: |x| ~ 0.006. df 2.1 is the §2.2 clamp floor and a live serving
#: configuration; the accuracy test's grid stopped at 2.5 and never saw it.
#:
#: This is the node count and not `_N_GRID` because the error is interpolation
#: between NODES -- measured insensitive to the resample grid, which was tried
#: first at 2x and 4x for no change at all. Nodes are temporary (only
#: `grid_log_y` is retained), so this costs ~5 ms of one-time build per map
#: and no memory. Worst case over every shipped pair falls to 1.4e-6.
_N_NODES = 8000
#: Uniform resampling grid. Dense enough that linear interpolation in
#: log-log space stays far inside the accuracy budget asserted in
#: tests/test_quantile_map.py, and cheap because lookup is O(1).
_N_GRID = 32768


def _dist(df: float | None):
    return stats.norm if df is None else stats.t(df=df)


def _unit_variance_scale(df: float | None) -> float:
    """Divisor taking a raw t to unit variance. Undefined at df <= 2."""
    if df is None:
        return 1.0
    if df <= 2.0:
        raise ValueError(f"df={df} has infinite variance; the clamp in §2.2 exists for this")
    return float(np.sqrt(df / (df - 2.0)))


@dataclass(frozen=True, slots=True)
class QuantileMap:
    """`apply(x)` sends a copula variate to a unit-variance marginal variate."""

    source_df: float | None
    target_df: float | None
    linear_slope: float | None  # set when the map is exactly linear
    #: Uniformly spaced grid in log|x|. Uniform on purpose: `np.interp` on a
    #: non-uniform grid costs a binary search per point, which measured at
    #: 30 ms per asset per request -- a third of §2.6's entire online budget
    #: spent looking up array indices. With constant spacing the index is
    #: arithmetic, and the same interpolation costs a few milliseconds.
    grid_u0: float
    grid_inv_step: float
    grid_log_y: np.ndarray
    log_x_max: float
    small_slope: float
    #: Last tabulated -log(survival) and d(log y)/du there, for the tail.
    last_u: float
    tail_slope_u: float

    @classmethod
    def build(cls, source_df: float | None, target_df: float | None) -> QuantileMap:
        tgt_scale = _unit_variance_scale(target_df)
        if source_df == target_df:
            # Same family and shape: the composition is the identity, and all
            # that remains is the standardisation. Skipping the table here is
            # what makes the common case (one df for the whole book) free.
            return cls(source_df, target_df, 1.0 / tgt_scale,
                       0.0, 0.0, np.empty(0), 0.0, 0.0, 0.0, 0.0)

        src, tgt = _dist(source_df), _dist(target_df)
        # Accurate nodes first, spaced by probability so the far tail is
        # resolved; these are exact scipy quantiles.
        p = np.logspace(np.log10(0.4999), np.log10(_MIN_TAIL_P), _N_NODES)
        xs = src.isf(p)
        ys = tgt.isf(p) / tgt_scale
        if not (np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)):
            raise RuntimeError("quantile grid is not monotone; the table would be wrong")
        log_x, log_y = np.log(xs), np.log(ys)
        u = -np.log(p)  # -log S(x) at each node, exact by construction

        # Resample onto a uniform grid in log|x|. log y is smooth and very
        # nearly linear in log x, so the resampling error at this spacing is
        # orders of magnitude below the 1e-5 the accuracy tests demand.
        grid_u = np.linspace(log_x[0], log_x[-1], _N_GRID)
        grid_log_y = np.interp(grid_u, log_x, log_y)
        step = (log_x[-1] - log_x[0]) / (_N_GRID - 1)
        return cls(
            source_df=source_df,
            target_df=target_df,
            linear_slope=None,
            grid_u0=float(log_x[0]),
            grid_inv_step=float(1.0 / step),
            grid_log_y=grid_log_y,
            log_x_max=float(log_x[-1]),
            small_slope=float(ys[0] / xs[0]),
            last_u=float(u[-1]),
            tail_slope_u=float((log_y[-1] - log_y[-2]) / (u[-1] - u[-2])),
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.linear_slope is not None:
            return x * self.linear_slope
        ax = np.abs(x)
        x_min = np.exp(self.grid_u0)
        # Branch-free over the whole array. Splitting out the near-zero and
        # far-tail cases with boolean masks costs two gathers and a scatter
        # over millions of elements, to serve regions that hold almost no
        # draws; `np.where` on the result is materially cheaper.
        la = np.log(np.maximum(ax, x_min))
        t = (la - self.grid_u0) * self.grid_inv_step
        np.clip(t, 0.0, self.grid_log_y.size - 1.000001, out=t)
        i = t.astype(np.intp)
        frac = t - i
        lo = self.grid_log_y[i]
        ly = lo + frac * (self.grid_log_y[i + 1] - lo)
        out = np.exp(ly)
        # Near zero the map is linear, with the slope of its first segment.
        np.copyto(out, ax * self.small_slope, where=ax < x_min)
        # Past the last node, continue in log-survival space, where the
        # relationship is asymptotically linear for both source families.
        # Reached with probability ~1e-13 per draw, so the exact logsf call
        # here costs nothing in aggregate.
        over = la > self.log_x_max
        if over.any():
            u = -_dist(self.source_df).logsf(ax[over])
            out[over] = np.exp(
                self.grid_log_y[-1] + self.tail_slope_u * (u - self.last_u)
            )
        return np.copysign(out, x)


class QuantileMapCache:
    """Maps are pure functions of (source df, target df); build each once.

    "Once" only holds if the key space is finite, and a raw MLE float is not:
    each distinct df mints a permanent ~266 KB table in a process that lives
    for months. `marginals.fit_df` rounds to `DF_DECIMALS` for this reason,
    but the rounding is repeated here so the bound is a property of the cache
    rather than of one caller's discipline -- a second caller passing a raw
    float would otherwise reopen the leak silently.

    Rounding the KEY only. The map itself is built for the rounded df, which
    is the df the caller then gets back; nothing is silently fitted at one
    shape and evaluated at another.
    """

    #: Matches `marginals.DF_DECIMALS`. Not imported, to keep `sim` from
    #: depending on `model`; the test suite pins the two together.
    KEY_DECIMALS = 1

    def __init__(self) -> None:
        self._cache: dict[tuple[float | None, float | None], QuantileMap] = {}

    @classmethod
    def _key_df(cls, df: float | None) -> float | None:
        return None if df is None else round(float(df), cls.KEY_DECIMALS)

    def get(self, source_df: float | None, target_df: float | None) -> QuantileMap:
        key = (self._key_df(source_df), self._key_df(target_df))
        hit = self._cache.get(key)
        if hit is None:
            hit = QuantileMap.build(key[0], key[1])
            self._cache[key] = hit
        return hit

    def __len__(self) -> int:
        return len(self._cache)


MAP_CACHE = QuantileMapCache()
