"""Exponentially weighted second moments (§2.1).

Half-life 20 days on hourly log returns, i.e. 480 steps.

Returns are treated as zero-mean. Subtracting a sample mean would inject
exactly the drift estimate that §2.4 forbids, at the cost of extra variance
in the estimator and no benefit at hourly frequency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HOURS_PER_DAY = 24
DEFAULT_HALF_LIFE_HOURS = 20 * HOURS_PER_DAY


@dataclass(frozen=True, slots=True)
class EwmaMoments:
    cov: np.ndarray  # (A, A) per-step covariance of log returns
    vol: np.ndarray  # (A,) per-step volatility
    corr: np.ndarray  # (A, A)
    n_eff: float  # effective sample size of the weighting scheme
    assets: tuple[str, ...]


def ewma_weights(n: int, half_life: float = DEFAULT_HALF_LIFE_HOURS) -> np.ndarray:
    """Weights for `n` observations, oldest first, summing to 1."""
    if n < 2:
        raise ValueError("need at least two observations")
    lam = 0.5 ** (1.0 / half_life)
    ages = np.arange(n - 1, -1, -1, dtype=np.float64)  # newest has age 0
    w = lam**ages
    return w / w.sum()


def effective_sample_size(weights: np.ndarray) -> float:
    """(sum w)^2 / sum w^2 — the `n` that goes into the Ledoit-Wolf formulas.

    The published shrinkage intensity assumes i.i.d. sampling with equal
    weights. Exponential weighting has no closed-form equivalent, so n_eff is
    substituted for n. This is an adaptation, recorded in OPEN-QUESTIONS A5.
    """
    return float(weights.sum() ** 2 / (weights**2).sum())


def ewma_moments(
    returns: np.ndarray,
    assets: tuple[str, ...],
    half_life: float = DEFAULT_HALF_LIFE_HOURS,
) -> EwmaMoments:
    """`returns` is (T, A), oldest row first, with no missing values.

    Assets whose history is too short to sit in this matrix do not get a row
    here: they go through the young-asset gate in `correlation.py` instead
    (§2.1), because a short-sample correlation estimate for a new listing is
    worse than an explicitly conservative imputed one.
    """
    returns = np.asarray(returns, dtype=np.float64)
    if returns.ndim != 2:
        raise ValueError(f"returns must be (T, A), got {returns.shape}")
    if returns.shape[1] != len(assets):
        raise ValueError(f"{returns.shape[1]} columns vs {len(assets)} asset names")
    if not np.isfinite(returns).all():
        raise ValueError("returns contain NaN/inf; align the window before calling")

    w = ewma_weights(returns.shape[0], half_life)
    cov = np.einsum("t,ti,tj->ij", w, returns, returns, optimize=True)
    cov = 0.5 * (cov + cov.T)
    vol = np.sqrt(np.diag(cov))
    if (vol <= 0).any():
        dead = [assets[i] for i in np.flatnonzero(vol <= 0)]
        raise ValueError(f"zero variance for {dead}; a constant price series is not tradable")
    corr = cov / np.outer(vol, vol)
    np.fill_diagonal(corr, 1.0)
    return EwmaMoments(
        cov=cov, vol=vol, corr=corr, n_eff=effective_sample_size(w), assets=assets
    )
