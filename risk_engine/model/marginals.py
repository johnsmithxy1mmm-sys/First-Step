"""Student-t marginals (§2.2).

One t per asset, degrees of freedom by maximum likelihood on a rolling
window, clamped to [2.1, 30] with every clamp logged.

The clamp is not cosmetic. Below df = 2 the variance is infinite, so scaling
the marginal to a target volatility is undefined and every downstream
number — the Cholesky, the CVaR, the bridge variance — becomes meaningless.
Above df = 30 the t is indistinguishable from a normal at any sample size we
have, so the upper clamp costs nothing and keeps the quantile maps well
conditioned.

Scale and shape are estimated from different windows on purpose: the
volatility comes from the EWMA moments (§2.1, half-life 20 days, so it
tracks the current regime), while df comes from an unweighted MLE over a
longer window, because tail shape is the parameter you have the least data
about and the one that moves the slowest.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import optimize, special

from risk_engine.observability.metrics import METRICS, Metrics

DF_MIN = 2.1
DF_MAX = 30.0
#: MLE search bounds, wider than the clamp so that a clamp is detectable.
_SEARCH_LO, _SEARCH_HI = 1.05, 200.0


@dataclass(frozen=True, slots=True)
class MarginalSpec:
    """A standardised Student-t scaled to `step_vol`."""

    asset: str
    df: float
    step_vol: float
    df_raw: float
    clamped: bool

    def __post_init__(self) -> None:
        if not (DF_MIN <= self.df <= DF_MAX):
            raise ValueError(f"{self.asset}: df {self.df} outside the clamp [{DF_MIN}, {DF_MAX}]")
        if self.step_vol <= 0:
            raise ValueError(f"{self.asset}: step_vol must be positive")

    @property
    def scale(self) -> float:
        """Multiplier taking a *standardised* t (unit variance) to `step_vol`."""
        return self.step_vol


def _neg_log_likelihood(df: float, x: np.ndarray) -> float:
    """Profile NLL of a zero-location t, with the scale profiled out by MLE.

    For fixed df the scale MLE solves a fixed-point equation; a few
    iterations of the standard EM update converge to plenty of precision for
    a shape parameter we then clamp to one decimal place anyway.
    """
    n = x.size
    s2 = float(np.mean(x**2))
    for _ in range(50):
        wts = (df + 1.0) / (df + x**2 / s2)
        new = float(np.mean(wts * x**2))
        if abs(new - s2) <= 1e-12 * max(s2, 1e-300):
            s2 = new
            break
        s2 = new
    scale = np.sqrt(s2)
    z = x / scale
    ll = n * (
        special.gammaln(0.5 * (df + 1.0))
        - special.gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
        - np.log(scale)
    )
    ll -= 0.5 * (df + 1.0) * np.log1p(z**2 / df).sum()
    return -ll


def fit_df(returns: np.ndarray) -> tuple[float, float, bool]:
    """Return (clamped_df, raw_mle_df, was_clamped)."""
    x = np.asarray(returns, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size < 100:
        raise ValueError(f"need >= 100 observations to fit a tail index, got {x.size}")
    res = optimize.minimize_scalar(
        _neg_log_likelihood,
        bounds=(_SEARCH_LO, _SEARCH_HI),
        args=(x,),
        method="bounded",
        options={"xatol": 1e-3},
    )
    raw = float(res.x)
    clamped = float(np.clip(raw, DF_MIN, DF_MAX))
    return clamped, raw, clamped != raw


def fit_marginal(
    asset: str,
    returns: np.ndarray,
    step_vol: float,
    metrics: Metrics | None = None,
) -> MarginalSpec:
    metrics = metrics or METRICS
    df, raw, clamped = fit_df(returns)
    if clamped:
        metrics.incr("df_clamps")
        metrics.df_clamps.append({"asset": asset, "df_raw": raw, "df_used": df})
    return MarginalSpec(asset=asset, df=df, step_vol=step_vol, df_raw=raw, clamped=clamped)
