"""Ledoit-Wolf shrinkage toward a constant-correlation target (§2.1).

Mandatory, not optional: an unshrunk EWMA correlation matrix over 50 assets
estimated from a few hundred effective observations has badly dispersed
eigenvalues, and the small ones are exactly what a Cholesky-driven path
generator amplifies. Tail estimates come out systematically too low.

The target keeps every variance and replaces every correlation with the
average correlation. The shrinkage intensity is the Ledoit-Wolf (2003)
constant-correlation estimator with two changes, both forced by §2.1's
requirement that the moments be exponentially weighted:

  - sample averages `(1/n) sum_t` become weighted sums `sum_t w_t`
  - the sample size `n` in the intensity becomes the effective sample size

This is an adaptation of the published estimator, not the published
estimator. See OPEN-QUESTIONS A5.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class ShrinkageResult:
    cov: np.ndarray
    intensity: float
    mean_correlation: float


def ledoit_wolf_constant_correlation(
    returns: np.ndarray, weights: np.ndarray, n_eff: float
) -> ShrinkageResult:
    """Shrink the weighted covariance of `returns` (T, A) toward equicorrelation."""
    x = np.asarray(returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    t, a = x.shape
    if w.shape != (t,):
        raise ValueError(f"weights {w.shape} do not match returns {x.shape}")
    if a < 2:
        return ShrinkageResult(cov=np.einsum("t,ti,tj->ij", w, x, x), intensity=0.0,
                               mean_correlation=0.0)

    s = np.einsum("t,ti,tj->ij", w, x, x, optimize=True)
    s = 0.5 * (s + s.T)
    var = np.diag(s).copy()
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)

    off = ~np.eye(a, dtype=bool)
    r_bar = float(corr[off].mean())

    # Target F: same variances, all correlations equal to r_bar.
    f = r_bar * np.outer(sd, sd)
    np.fill_diagonal(f, var)

    # pi: summed asymptotic variance of the entries of S.
    x2 = x * x
    pi_mat = np.einsum("t,ti,tj->ij", w, x2, x2, optimize=True) - s**2
    pi = float(pi_mat.sum())

    # rho: covariance between the estimation error of S and of F.
    # theta_ii_ij = sum_t w_t (x_ti^2 - s_ii)(x_ti x_tj - s_ij)
    term = np.einsum("t,ti,ti,tj->ij", w, x2, x, x, optimize=True) - var[:, None] * s
    ratio = np.outer(1.0 / sd, sd)  # ratio[i, j] = sd_j / sd_i
    rho = float(np.diag(pi_mat).sum())
    cross = 0.5 * r_bar * (ratio * term + ratio.T * term.T)
    rho += float(cross[off].sum())

    gamma = float(((f - s) ** 2).sum())
    if gamma <= 0:
        intensity = 0.0
    else:
        intensity = float(np.clip((pi - rho) / gamma / n_eff, 0.0, 1.0))

    shrunk = intensity * f + (1.0 - intensity) * s
    # The target shares S's diagonal, so variances are untouched by
    # construction; enforce it against floating point drift.
    np.fill_diagonal(shrunk, var)
    return ShrinkageResult(cov=shrunk, intensity=intensity, mean_correlation=r_bar)
