"""Correlated price paths from a t-copula with Student-t marginals (§2.3).

Dependence and marginals are separated on purpose: the copula carries joint
tail behaviour (assets crashing together), the marginals carry each asset's
own tail thickness. A multivariate-t with a single df would force both to be
the same number.

Common random numbers
---------------------
`BaseRandomness` exists so that two configurations can be driven by
*identical* draws. Several §3.1 benchmarks compare probabilities whose true
difference is smaller than the Monte Carlo noise on either one; comparing
independent runs there tests the random number generator, not the model.
With shared draws, monotonicity in leverage is exact pathwise rather than
statistical, which is what §3.1.5 actually asks for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from risk_engine.model.drift import DriftConvention, log_drift_per_step
from risk_engine.sim.quantile_map import MAP_CACHE


@dataclass(frozen=True, slots=True)
class PathSpec:
    """Everything needed to draw paths for one universe of assets."""

    coins: tuple[str, ...]
    step_vol: np.ndarray  # (A,) per-hour log-return vol
    corr: np.ndarray  # (A, A), already PD
    marginal_df: tuple[float | None, ...]  # None -> Gaussian marginal
    copula_df: float | None  # None -> Gaussian copula (the §3.2 baseline uses this)
    drift: DriftConvention = DriftConvention.ZERO_LOG_RETURN

    def __post_init__(self) -> None:
        a = len(self.coins)
        if self.step_vol.shape != (a,):
            raise ValueError(f"step_vol {self.step_vol.shape} vs {a} coins")
        if self.corr.shape != (a, a):
            raise ValueError(f"corr {self.corr.shape} vs {a} coins")
        if len(self.marginal_df) != a:
            raise ValueError(f"{len(self.marginal_df)} marginals vs {a} coins")

    @property
    def n_assets(self) -> int:
        return len(self.coins)


@dataclass(frozen=True, slots=True)
class BaseRandomness:
    """Raw draws, reusable across configurations that share a shape.

    `chi` is the copula's mixing variable and is therefore tied to the copula
    df it was drawn for; `copula_df` records that so reuse across a different
    copula cannot happen silently.
    """

    z: np.ndarray  # (P, S, A) iid standard normal
    chi: np.ndarray | None  # (P, S, 1) chi2_nu / nu
    copula_df: float | None
    bridge_cross: np.ndarray  # (P, S)
    bridge_iso: np.ndarray  # (P, S, I)

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.z.shape


def draw_base_randomness(
    n_paths: int,
    n_steps: int,
    n_assets: int,
    n_isolated: int,
    copula_df: float | None,
    rng: np.random.Generator,
) -> BaseRandomness:
    z = rng.standard_normal((n_paths, n_steps, n_assets))
    chi = None
    if copula_df is not None:
        chi = rng.chisquare(copula_df, size=(n_paths, n_steps, 1)) / copula_df
    return BaseRandomness(
        z=z,
        chi=chi,
        copula_df=copula_df,
        bridge_cross=rng.random((n_paths, n_steps)),
        bridge_iso=rng.random((n_paths, n_steps, n_isolated)),
    )


def generate_log_returns(spec: PathSpec, base: BaseRandomness) -> np.ndarray:
    """(P, S, A) correlated log returns."""
    if base.copula_df != spec.copula_df:
        raise ValueError(
            f"randomness was drawn for copula df {base.copula_df}, spec wants {spec.copula_df}; "
            "the mixing variable is not interchangeable across copulas"
        )
    if base.z.shape[2] != spec.n_assets:
        raise ValueError(f"randomness has {base.z.shape[2]} assets, spec has {spec.n_assets}")

    chol = np.linalg.cholesky(spec.corr)
    # `y` is freshly allocated by the matmul and is not visible to the caller,
    # so every step from here on works in place on it. `base.z` is never
    # touched -- it is shared across configurations by design.
    y = base.z @ chol.T
    if spec.copula_df is not None:
        # Elliptical t: a Gaussian vector divided by an independent
        # sqrt(chi2/nu). One mixing draw per (path, step) is what couples the
        # assets in the tail -- a shared shock, not per-asset noise.
        y /= np.sqrt(base.chi)

    # One scratch column shared by every asset: `apply` is memory-bound and
    # allocating its working buffer per asset measured as real time at
    # (20 000 x 24) elements. Column `a` is only written after `apply` has
    # finished reading it, and the columns are disjoint, so mapping in place
    # is safe and saves a second (P, S, A) array.
    scratch = np.empty(y.shape[:2], dtype=np.float64)
    for a, df in enumerate(spec.marginal_df):
        col = MAP_CACHE.get(spec.copula_df, df).apply(y[:, :, a], out=scratch)
        # Vol-scale the column while it is still hot, instead of sweeping the
        # whole (P, S, A) block afterwards. Same bits: the map's last act is a
        # `copysign`, and multiplying a signed magnitude by a positive vol
        # gives what scaling the block later would have given.
        col *= spec.step_vol[a]
        y[:, :, a] = col
    drift = np.asarray(log_drift_per_step(spec.drift, spec.step_vol))
    # ZERO_LOG_RETURN -- the default -- has drift exactly zero, and adding a
    # zero vector to (P, S, A) is a full pass over the array for nothing.
    if drift.any():
        y += drift[None, None, :]
    return y


def generate_price_paths(
    spec: PathSpec, spot: np.ndarray, base: BaseRandomness
) -> np.ndarray:
    """(P, S+1, A) price paths; column 0 is `spot`."""
    # `r` is this function's private array (generate_log_returns builds it
    # fresh every call), so the cumulative sum and the exponential both run in
    # place, and the scaling by spot writes straight into the output block
    # instead of through a full-size temporary.
    r = generate_log_returns(spec, base)
    np.cumsum(r, axis=1, out=r)
    np.exp(r, out=r)
    prices = np.empty((r.shape[0], r.shape[1] + 1, r.shape[2]), dtype=np.float64)
    prices[:, 0, :] = spot[None, :]
    np.multiply(r, spot[None, None, :], out=prices[:, 1:, :])
    return prices


def lower_tail_dependence(copula_df: float, rho: float) -> float:
    """Model-implied coefficient of tail dependence for a t-copula.

    lambda = 2 * T_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ), and it is
    symmetric between the tails -- which is precisely the limitation §2.3
    asks to be diagnosed against the data.
    """
    from scipy import stats

    if not -1.0 < rho < 1.0:
        rho = float(np.clip(rho, -0.999999, 0.999999))
    arg = -np.sqrt((copula_df + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * stats.t(df=copula_df + 1.0).cdf(arg))
