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

    def independent(self) -> PathSpec:
        """Baseline B (§3.2): identical marginals, dependence removed."""
        return PathSpec(
            coins=self.coins,
            step_vol=self.step_vol,
            corr=np.eye(self.n_assets),
            marginal_df=self.marginal_df,
            copula_df=self.copula_df,
            drift=self.drift,
        )


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
    y = base.z @ chol.T
    if spec.copula_df is not None:
        # Elliptical t: a Gaussian vector divided by an independent
        # sqrt(chi2/nu). One mixing draw per (path, step) is what couples the
        # assets in the tail -- a shared shock, not per-asset noise.
        y = y / np.sqrt(base.chi)

    out = np.empty_like(y)
    for a, df in enumerate(spec.marginal_df):
        out[:, :, a] = MAP_CACHE.get(spec.copula_df, df).apply(y[:, :, a])
    out *= spec.step_vol[None, None, :]
    out += np.asarray(log_drift_per_step(spec.drift, spec.step_vol))[None, None, :]
    return out


def generate_price_paths(
    spec: PathSpec, spot: np.ndarray, base: BaseRandomness
) -> np.ndarray:
    """(P, S+1, A) price paths; column 0 is `spot`."""
    r = generate_log_returns(spec, base)
    log_paths = np.cumsum(r, axis=1)
    prices = np.empty((r.shape[0], r.shape[1] + 1, r.shape[2]), dtype=np.float64)
    prices[:, 0, :] = spot[None, :]
    np.exp(log_paths, out=log_paths)
    prices[:, 1:, :] = spot[None, None, :] * log_paths
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
