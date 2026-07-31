"""t-copula fitting and the tail-asymmetry diagnostic (§2.3).

§2.3 mandates the diagnostic, and for a good reason: a t-copula is
*symmetric*. Its coefficient of tail dependence is identical in the upper and
lower tails by construction. Crypto perp returns are not symmetric -- assets
crash together harder than they rally together -- so the model is
structurally incapable of representing the very asymmetry that matters most
to a leveraged long book.

`diagnose_tail_asymmetry` measures the empirical dependence in each tail and
compares the lower one against what the fitted copula implies. If the model
understates the lower tail, §2.3 calls that a blocking defect requiring a
switch to a skewed-t. **That switch is not implemented in Phase 1.** The
diagnostic therefore reports the defect rather than papering over it, and
`assert_lower_tail_not_understated` turns it into a hard failure, because §9
requires an unmet criterion to stop and be reported rather than be worked
around.

The caller is `service.state._checked_tail_diagnostics`, run from both bundle
builders. **Naming it matters**: this paragraph made the same claim in the
same present tense while nothing outside the test suite called either
function (OPEN-QUESTIONS A10), so the model read as guarded by an assertion
that production never reached. A docstring cannot establish that a check
runs; a named caller can be grepped for, and `TestTheDiagnosticRunsOnTheShippedPath`
fails if it goes away.

§2.3 also does not say how the copula's degrees of freedom are chosen.
`fit_copula_df` maximises the copula likelihood on pseudo-observations over
a grid, holding the correlation matrix fixed at the shrunk estimate -- the
standard two-stage (IFM) approach. Recorded as OPEN-QUESTIONS A9.

**`fit_copula_df` has no shipped caller either**, and unlike the diagnostic
that is not yet fixed: both bundles pass a hardcoded `copula_df=4.0`. Wiring
a check that can only refuse changes no output; replacing 4.0 with a fitted
value changes every output, which is a distribution change and resets the
§3.3 shadow counter. Tracked under A9 as a decision to take deliberately.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from risk_engine.sim.paths import lower_tail_dependence

COPULA_DF_GRID = np.concatenate([np.arange(2.5, 12.1, 0.5), np.arange(13.0, 30.1, 1.0)])


def pseudo_observations(returns: np.ndarray) -> np.ndarray:
    """Rank-transform each column to (0, 1), the standard copula input."""
    x = np.asarray(returns, dtype=np.float64)
    n = x.shape[0]
    ranks = np.argsort(np.argsort(x, axis=0), axis=0) + 1
    return ranks / (n + 1.0)


def t_copula_loglik(u: np.ndarray, corr: np.ndarray, df: float) -> float:
    d = u.shape[1]
    q = stats.t(df=df).ppf(u)
    inv = np.linalg.inv(corr)
    sign, logdet = np.linalg.slogdet(corr)
    if sign <= 0:  # pragma: no cover - guarded upstream by the PD projection
        raise ValueError("correlation matrix is not positive definite")
    maha = np.einsum("ni,ij,nj->n", q, inv, q, optimize=True)

    joint = (
        special.gammaln(0.5 * (df + d))
        - special.gammaln(0.5 * df)
        - 0.5 * d * np.log(df * np.pi)
        - 0.5 * logdet
        - 0.5 * (df + d) * np.log1p(maha / df)
    )
    marginal = (
        special.gammaln(0.5 * (df + 1.0))
        - special.gammaln(0.5 * df)
        - 0.5 * np.log(df * np.pi)
        - 0.5 * (df + 1.0) * np.log1p(q**2 / df)
    )
    return float(joint.sum() - marginal.sum())


def fit_copula_df(
    returns: np.ndarray, corr: np.ndarray, grid: np.ndarray = COPULA_DF_GRID
) -> float:
    """Two-stage MLE: correlation held at the shrunk estimate, df profiled."""
    u = pseudo_observations(returns)
    lls = [t_copula_loglik(u, corr, float(df)) for df in grid]
    return float(grid[int(np.argmax(lls))])


def empirical_tail_dependence(
    u: np.ndarray, v: np.ndarray, threshold: float = 0.05
) -> tuple[float, float]:
    """(lower, upper) empirical tail-dependence estimates at `threshold`."""
    lower_n = int((u <= threshold).sum())
    upper_n = int((u > 1 - threshold).sum())
    lower = float(((u <= threshold) & (v <= threshold)).sum() / lower_n) if lower_n else np.nan
    upper = (
        float(((u > 1 - threshold) & (v > 1 - threshold)).sum() / upper_n) if upper_n else np.nan
    )
    return lower, upper


def model_tail_dependence_at_threshold(
    df: float, rho: float, threshold: float, n_sim: int = 200_000, seed: int = 0
) -> float:
    """P(V <= q | U <= q) under a t-copula, at the *same finite* q as the data.

    The closed form `lower_tail_dependence` is the asymptotic coefficient,
    the limit as q -> 0. At a workable threshold like q = 0.05 the finite
    estimate is materially larger -- for nu=4, rho=0.6 it is about 0.40
    against an asymptotic 0.31. Comparing an empirical estimate at q = 0.05
    against the asymptotic number therefore flags a *correctly specified*
    model as understating its lower tail, which is exactly the false alarm
    that would get the §2.3 gate disabled the first time it was inconvenient.

    Simulated rather than derived from a bivariate t CDF because the
    comparison is then literally the same statistic applied to model draws
    and to data.
    """
    rng = np.random.default_rng(seed)
    chol = np.linalg.cholesky(np.array([[1.0, rho], [rho, 1.0]]))
    z = rng.standard_normal((n_sim, 2)) @ chol.T
    w = rng.chisquare(df, size=(n_sim, 1)) / df
    t = z / np.sqrt(w)
    u = stats.t(df=df).cdf(t)
    lower, _ = empirical_tail_dependence(u[:, 0], u[:, 1], threshold)
    return lower


@dataclass(frozen=True, slots=True)
class TailDiagnostic:
    pair: tuple[str, str]
    threshold: float
    empirical_lower: float
    empirical_upper: float
    #: Same statistic, same threshold, under the fitted copula.
    model_at_threshold: float
    #: The asymptotic coefficient, reported for context only. Never compared
    #: against a finite-threshold empirical estimate -- see the function above.
    model_asymptotic: float
    n_lower_exceedances: int

    @property
    def asymmetry(self) -> float:
        """Positive means the data crash together harder than they rally."""
        return self.empirical_lower - self.empirical_upper

    def understates_lower_tail(self, margin: float = 0.05) -> bool:
        """True when the symmetric copula sits below the observed lower tail.

        `margin` keeps sampling noise from tripping the flag; with a 5%
        threshold on a few thousand hourly observations the standard error on
        the empirical estimate is a few percentage points.
        """
        return self.empirical_lower - self.model_at_threshold > margin

    def __str__(self) -> str:
        return (
            f"{self.pair[0]}/{self.pair[1]} @ q={self.threshold}: "
            f"lower={self.empirical_lower:.3f} upper={self.empirical_upper:.3f} "
            f"model@q={self.model_at_threshold:.3f} "
            f"model_asymptotic={self.model_asymptotic:.3f} "
            f"asymmetry={self.asymmetry:+.3f} (n={self.n_lower_exceedances})"
        )


def diagnose_tail_asymmetry(
    returns: np.ndarray,
    assets: tuple[str, ...],
    corr: np.ndarray,
    copula_df: float,
    threshold: float = 0.05,
    n_sim: int = 200_000,
) -> list[TailDiagnostic]:
    """One diagnostic per asset pair, worst lower-tail gap first."""
    u = pseudo_observations(returns)
    out: list[TailDiagnostic] = []
    for i in range(len(assets)):
        for j in range(i + 1, len(assets)):
            lower, upper = empirical_tail_dependence(u[:, i], u[:, j], threshold)
            rho = float(corr[i, j])
            out.append(
                TailDiagnostic(
                    pair=(assets[i], assets[j]),
                    threshold=threshold,
                    empirical_lower=lower,
                    empirical_upper=upper,
                    model_at_threshold=model_tail_dependence_at_threshold(
                        copula_df, rho, threshold, n_sim=n_sim, seed=i * 1000 + j
                    ),
                    model_asymptotic=lower_tail_dependence(copula_df, rho),
                    n_lower_exceedances=int((u[:, i] <= threshold).sum()),
                )
            )
    return sorted(out, key=lambda d: d.model_at_threshold - d.empirical_lower)


def assert_lower_tail_not_understated(
    diagnostics: list[TailDiagnostic], margin: float = 0.05
) -> None:
    """§2.3: understating the lower tail is a blocking defect.

    Raises rather than warns. The remedy §2.3 names is a skewed-t, which
    Phase 1 does not implement -- so the honest response to this firing is to
    stop and report it, not to continue with a model known to understate the
    risk the product exists to measure (§9, §10).
    """
    bad = [d for d in diagnostics if d.understates_lower_tail(margin)]
    if bad:
        lines = "\n  ".join(str(d) for d in bad)
        raise ValueError(
            "the t-copula understates lower-tail dependence for:\n  "
            f"{lines}\n"
            "§2.3 classifies this as a blocking defect and prescribes a skewed-t, "
            "which is not implemented. Report this rather than proceeding."
        )
