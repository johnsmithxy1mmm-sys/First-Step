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

`fit_copula_df` IS the shipped path as of 0.3.0. Both bundle builders reach
it through `service/state.py::_fitted_copula_df`; the hardcoded
`copula_df=4.0` this paragraph used to describe is gone, and on the fixture
the fitted value is 6.5. Replacing the constant changed every output, which
is a distribution change, so it was taken as a MINOR version bump that resets
the §3.3 shadow counter -- deliberately, before the clock started. A9.

(This paragraph asserted the opposite for a release after it stopped being
true, which is the defect class this file's other notes exist to catch: prose
in the present tense describing wiring that has changed. It is cheap to write
and expensive to trust.)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import special, stats

from risk_engine.sim.paths import lower_tail_dependence

COPULA_DF_GRID = np.concatenate([np.arange(2.5, 12.1, 0.5), np.arange(13.0, 30.1, 1.0)])

#: A pair becomes a DEMAND on the tail floor only when its lower-tail
#: shortfall is at least this many standard errors (A11 finding 3, measured:
#: an unconditional criterion at these sample sizes has no null and installs
#: a remedy on zero-signal data ~84% of the time; conditioning restores the
#: nominal rate). Pairs below this are noise to re-test, not to fit.
TAIL_DEMAND_SIGMAS = 2.0

#: The floor targets the demand's one-sided 95% lower confidence bound,
#: `empirical_lower - 1.645*SE`, not its point estimate. Measured (A11
#: proposal, 2026-08-05): chasing the point on the live ETH/SOL reading needs
#: df ~ 1.05 -- a near-Cauchy copula that would wreck the body fit and every
#: other pair to chase one number carrying a 0.04 SE. The bound is what the
#: data insists on at 95%; §10's heavier-tail preference is served by the
#: DEMAND threshold being conditional, not by over-fitting the point.
TAIL_ONE_SIDED_Z = 1.645


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

        The effective margin is `max(margin, TAIL_ONE_SIDED_Z * SE)`: the
        absolute `margin` binds at large n, the SE-scaled term at small n.

        The fixed 0.05 alone was 1.13 SE at the live window's n=108 — a
        criterion with essentially no null, firing readily on noise (the same
        defect A11 finding 3 measured for the skew margin: ~84% false-positive
        rate on zero-signal data). The SE term gives the gate a nominal
        one-sided 5% level where the data are thin, and hands back to the
        absolute floor as n grows and 0.05 becomes a real signal. This is
        also what makes the A11 tail floor coherent with the gate: the floor
        raises the model to the demand's one-sided 95% lower bound, so a
        floored bundle's residual shortfall is at most `z*SE` — inside this
        margin by construction (up to Monte-Carlo noise between two estimates
        of model@q; a borderline flip re-fires the gate and the next build
        floors deeper, which is self-correcting rather than silent).
        """
        if not np.isfinite(self.empirical_lower):
            # NaN means the window held no lower-tail exceedances at all, so
            # there is no measurement to compare. `NaN > margin` is False,
            # which would let a §2.3 BLOCKING gate pass vacuously — the guard
            # reporting "not understated" on the strength of no evidence. An
            # unmeasurable tail is not a safe tail; say so, and let
            # `assert_lower_tail_not_understated` refuse.
            return True
        se = self.lower_standard_error
        effective = max(margin, TAIL_ONE_SIDED_Z * se) if np.isfinite(se) else margin
        return self.empirical_lower - self.model_at_threshold > effective

    @property
    def lower_standard_error(self) -> float:
        """SE of the empirical lower estimate, as a binomial proportion.

        Reported because the margin is not many of these. On the live mainnet
        window (2160 hourly observations, so n=108 exceedances at q=0.05) the
        SE is ~0.042 and the 0.05 margin is 1.13 SE, which means the gate
        fires readily on noise. That is the intended direction -- §10 makes a
        false alarm cheaper than a miss -- but an operator reading a refusal
        needs to be able to tell a 2.6-sigma signal from a 1.4-sigma one.
        """
        n = self.n_lower_exceedances
        p = self.empirical_lower
        if n <= 0 or not np.isfinite(p):
            return float("nan")
        return float(np.sqrt(max(p * (1.0 - p), 0.0) / n))

    @property
    def lower_excess_sigmas(self) -> float:
        """How far past the model the lower tail sits, in standard errors."""
        se = self.lower_standard_error
        if not np.isfinite(se) or se <= 0:
            return float("nan")
        return float((self.empirical_lower - self.model_at_threshold) / se)

    @property
    def upper_also_understated(self) -> bool:
        """True when the model is under the UPPER tail too.

        This is the distinction §2.3's prescribed remedy turns on. A skewed-t
        heavies one tail at the other's expense, so it is the right fix only
        when the lower tail is heavier than the upper. When BOTH tails run
        above the model the family or its df is too thin all round, and adding
        skew would fit one tail by making the other worse.
        """
        if not (np.isfinite(self.empirical_upper) and np.isfinite(self.empirical_lower)):
            return False
        return self.empirical_upper >= self.empirical_lower

    def __str__(self) -> str:
        sig = self.lower_excess_sigmas
        sigma = f" {sig:+.1f}sigma" if np.isfinite(sig) else ""
        shape = " [UPPER TAIL ALSO UNDER-MODELLED]" if self.upper_also_understated else ""
        return (
            f"{self.pair[0]}/{self.pair[1]} @ q={self.threshold}: "
            f"lower={self.empirical_lower:.3f} upper={self.empirical_upper:.3f} "
            f"model@q={self.model_at_threshold:.3f} "
            f"model_asymptotic={self.model_asymptotic:.3f} "
            f"asymmetry={self.asymmetry:+.3f} (n={self.n_lower_exceedances}"
            f"{sigma}){shape}"
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


@dataclass(frozen=True, slots=True)
class TailFloor:
    """What the conditional tail floor decided, kept for logging and audit."""

    df_ml: float
    df: float
    #: The pairs that were demands (>= TAIL_DEMAND_SIGMAS over the model).
    demands: tuple[TailDiagnostic, ...]
    #: False when even the grid's heaviest df cannot reach every demand's
    #: bound; the §2.3 gate then keeps firing and recording mode continues,
    #: which is the correct escalation rather than a silent best-effort.
    covered: bool

    @property
    def floored(self) -> bool:
        return self.df < self.df_ml


def tail_floor_df(
    diagnostics: list[TailDiagnostic],
    assets: tuple[str, ...],
    corr: np.ndarray,
    df_ml: float,
    grid: np.ndarray = COPULA_DF_GRID,
    demand_sigmas: float = TAIL_DEMAND_SIGMAS,
    one_sided_z: float = TAIL_ONE_SIDED_Z,
    n_sim: int = 400_000,
) -> TailFloor:
    """A11's adopted remedy (option A): `df* = min(df_ML, df_tail)`.

    `df_tail` is the LARGEST df on the grid whose model@q covers
    `empirical_lower - z*SE` for every pair whose shortfall is significant at
    `demand_sigmas`. Conditional exactly as finding 3 requires — sub-2-sigma pairs
    are noise to re-test, never demands — and one-sided exactly as finding 4
    requires: lambda_U is reported by the diagnostic, never constrained, because it
    is not monotone in the family's parameters and a two-sided criterion is
    unsatisfiable (measured: the live BTC/ETH upper tail is unreachable at any
    admissible skew, and for the df lever the whole grid moves lambda_U with λ_L).

    Why a floor under the ML fit rather than a tail-matched estimator
    outright: the likelihood sees every observation and the tail statistic
    sees ~n*q of them, so the ML fit is the better estimate everywhere the
    tail does not contradict it at 2 sigma. The floor binds only on measured,
    significant shortfall — which also means it cannot RAISE df: thin-tailed
    data leave the ML fit alone.

    model@q is monotone decreasing in df, so the first covering df walking
    the grid downward from `df_ml` is the largest one. Each check simulates
    (`n_sim` per pair per step, fixed seed) — an offline cost in a build that
    already simulates the diagnostic itself.

    When even the grid floor cannot cover every demand, the floor is applied
    anyway (`covered=False`): heavier is still nearer the data, and the §2.3
    gate stays lit, which keeps recording mode on. Chasing coverage OFF the
    grid is foreclosed — the measured reach of the family says the point
    estimate needs df ~ 1, a near-Cauchy copula (OPEN-QUESTIONS A11).
    """
    index = {a: i for i, a in enumerate(assets)}
    demands = tuple(
        d for d in diagnostics
        if d.understates_lower_tail()
        and np.isfinite(d.lower_excess_sigmas)
        and d.lower_excess_sigmas >= demand_sigmas
    )
    if not demands or len(assets) < 2:
        return TailFloor(df_ml=df_ml, df=df_ml, demands=(), covered=True)

    targets = []
    for d in demands:
        se = d.lower_standard_error
        i, j = index[d.pair[0]], index[d.pair[1]]
        targets.append((float(corr[i, j]), d.empirical_lower - one_sided_z * se))

    def covers(df: float) -> bool:
        return all(
            model_tail_dependence_at_threshold(
                df, rho, demands[0].threshold, n_sim=n_sim, seed=7
            ) >= target
            for rho, target in targets
        )

    candidates = sorted({float(g) for g in grid if g <= df_ml}, reverse=True)
    for df in candidates:
        if covers(df):
            return TailFloor(df_ml=df_ml, df=df, demands=demands, covered=True)
    floor = candidates[-1] if candidates else df_ml
    return TailFloor(df_ml=df_ml, df=floor, demands=demands, covered=False)


def assert_lower_tail_not_understated(
    diagnostics: list[TailDiagnostic], margin: float = 0.05
) -> None:
    """§2.3: understating the lower tail is a blocking defect.

    Raises rather than warns. §2.3 names a skewed-t as the remedy, and that
    prescription was REFUTED by measurement (OPEN-QUESTIONS A11, all four
    findings, 2026-08-04): not per-output conservative, correlation-destroying
    below nu=4, and unable to reach the observed upper tails at any admissible
    skew. The adopted remedy is the conditional df floor (`tail_floor_df`),
    which runs BEFORE this gate in the bundle build -- so this firing means
    the floor could not cover the demands within the family's grid, and the
    honest response is still to stop and report, not to continue with a model
    known to understate the risk the product exists to measure (§9, §10).
    """
    bad = [d for d in diagnostics if d.understates_lower_tail(margin)]
    if bad:
        lines = "\n  ".join(str(d) for d in bad)
        unmeasured = [d for d in bad if not np.isfinite(d.empirical_lower)]
        note = (
            "\nSome pairs have NO measurable lower tail in this window "
            f"({', '.join('/'.join(d.pair) for d in unmeasured)}): too few joint "
            "exceedances to estimate one. That is reported as a failure rather "
            "than a pass -- an unmeasurable tail is not evidence of a safe one."
            if unmeasured else ""
        )
        # §2.3 prescribes a skewed-t, and for a pair whose lower tail is
        # genuinely heavier than its upper that is the right remedy. It is NOT
        # the right remedy for a pair whose UPPER tail is also above the
        # model: skew buys one tail at the other's expense, so applying it
        # there would fit the lower tail by making the upper one worse. That
        # case says the family or its df is too thin all round.
        #
        # Observed live on mainnet 2026-08-03, which is why this is separated
        # rather than left implicit: BTC/ETH came back lower=0.694 against
        # upper=0.731 -- asymmetry NEGATIVE -- while still failing the gate,
        # so a run that read the message literally would have reached for the
        # wrong fix.
        symmetric = [d for d in bad if d.upper_also_understated]
        shape_note = (
            "\nNOT AN ASYMMETRY for "
            f"{', '.join('/'.join(d.pair) for d in symmetric)}: the upper tail is "
            "at or above the lower one, so the model is under BOTH tails there. "
            "A skewed-t is the wrong remedy for that -- it would fit the lower "
            "tail by worsening the upper. Those pairs point at the copula df or "
            "the elliptical family itself (OPEN-QUESTIONS A11)."
            if symmetric else ""
        )
        raise ValueError(
            "the t-copula understates lower-tail dependence for:\n  "
            f"{lines}\n"
            "§2.3 classifies this as a blocking defect. The adopted remedy -- the "
            "A11 conditional df floor -- runs before this gate, so this firing "
            "means the demands exceed what the family's grid can cover. Report "
            f"this rather than proceeding.{note}"
            f"{shape_note}"
        )
