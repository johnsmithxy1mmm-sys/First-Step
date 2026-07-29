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
_N_NODES = 3000


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
    log_x: np.ndarray
    log_y: np.ndarray
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
                       np.empty(0), np.empty(0), 0.0, 0.0, 0.0)

        src, tgt = _dist(source_df), _dist(target_df)
        p = np.logspace(np.log10(0.4999), np.log10(_MIN_TAIL_P), _N_NODES)
        xs = src.isf(p)
        ys = tgt.isf(p) / tgt_scale
        if not (np.all(np.diff(xs) > 0) and np.all(np.diff(ys) > 0)):
            raise RuntimeError("quantile grid is not monotone; the table would be wrong")
        log_x, log_y = np.log(xs), np.log(ys)
        u = -np.log(p)  # -log S(x) at each node, exact by construction
        return cls(
            source_df=source_df,
            target_df=target_df,
            linear_slope=None,
            log_x=log_x,
            log_y=log_y,
            small_slope=float(ys[0] / xs[0]),
            last_u=float(u[-1]),
            tail_slope_u=float((log_y[-1] - log_y[-2]) / (u[-1] - u[-2])),
        )

    def apply(self, x: np.ndarray) -> np.ndarray:
        if self.linear_slope is not None:
            return x * self.linear_slope
        ax = np.abs(x)
        out = np.empty_like(ax)
        small = ax <= np.exp(self.log_x[0])
        out[small] = ax[small] * self.small_slope
        big = ~small
        if big.any():
            vals = ax[big]
            la = np.log(vals)
            ly = np.interp(la, self.log_x, self.log_y)
            # np.interp clamps at the last node; continue in log-survival
            # space, where the relationship is asymptotically linear for both
            # source families. Reached with probability ~1e-13 per draw, so
            # the cost of the exact logsf call here is irrelevant.
            over = la > self.log_x[-1]
            if over.any():
                u = -_dist(self.source_df).logsf(vals[over])
                ly[over] = self.log_y[-1] + self.tail_slope_u * (u - self.last_u)
            out[big] = np.exp(ly)
        return np.copysign(out, x)


class QuantileMapCache:
    """Maps are pure functions of (source df, target df); build each once."""

    def __init__(self) -> None:
        self._cache: dict[tuple[float | None, float | None], QuantileMap] = {}

    def get(self, source_df: float | None, target_df: float | None) -> QuantileMap:
        key = (source_df, target_df)
        hit = self._cache.get(key)
        if hit is None:
            hit = QuantileMap.build(source_df, target_df)
            self._cache[key] = hit
        return hit


MAP_CACHE = QuantileMapCache()
