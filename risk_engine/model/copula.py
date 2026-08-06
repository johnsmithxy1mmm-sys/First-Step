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

from dataclasses import dataclass, replace

import numpy as np
from scipy import special, stats

from risk_engine.model.psd import project_to_correlation
from risk_engine.sim.paths import lower_tail_dependence

COPULA_DF_GRID = np.concatenate([np.arange(2.5, 12.1, 0.5), np.arange(13.0, 30.1, 1.0)])

#: The floor targets the demand's one-sided 95% lower confidence bound,
#: `empirical_lower - 1.645*SE`, not its point estimate. Measured (A11
#: proposal, 2026-08-05): chasing the point on the live ETH/SOL reading needs
#: df ~ 1.05 -- a near-Cauchy copula that would wreck the body fit and every
#: other pair to chase one number carrying a 0.04 SE. The bound is what the
#: data insists on at 95%; §10's heavier-tail preference is served by the
#: DEMAND threshold being conditional, not by over-fitting the point.
TAIL_ONE_SIDED_Z = 1.645

#: A pair becomes a DEMAND on the tail floor when its lower-tail shortfall is
#: significant at this many standard errors -- THE SAME threshold the gate's
#: margin tests, and the sameness is load-bearing. As shipped in 0.5.0 this
#: was 2.0 against the gate's 1.645, which left a stuck band: a pair at, say,
#: 1.8 sigma kept the §2.3 gate lit (so recording mode continued and no
#: gate-day accrued) while never becoming a demand (so the floor never even
#: attempted a remedy). A gate that can fire on a pair the remedy is not
#: allowed to see is the A10 defect in a new coat. One threshold, shared:
#: every pair that can hold the gate open is a pair the floor tries to cover.
#: Conditionality (A11 finding 3) is preserved -- 1.645 one-sided IS a null,
#: which the old flat margin never had.
TAIL_DEMAND_SIGMAS = TAIL_ONE_SIDED_Z

#: The rho-lift never raises a pair past this. At 1.0 the pair is comonotone
#: and the matrix is singular -- Cholesky, which every serving slice relies
#: on, fails outright. Measured with the gate's estimator: model@q at 0.98 is
#: ~0.84-0.87 across the df grid (0.871 at df 2.5, 0.838 at df 30) -- far
#: above the live bounds (~0.69) but NOT unreachable by a pathological
#: reading, so the cap is a real boundary: bounds past it come back
#: uncovered and the gate stays lit. A pair whose MEASURED rho already
#: exceeds the cap is never cut down to it -- the search domain is
#: [rho_from, max(cap, rho_from)], see `_smallest_covering_rho`.
TAIL_RHO_CAP = 0.98

#: The lifted rho is resolved to this granularity by bisection. Finer than
#: the Monte-Carlo noise on model@q at the gate's n_sim (~0.005), so the
#: resolution is not what limits the lift's precision. What removes the
#: noise from the COHERENCE question is that the search scores candidate
#: rhos with the gate's own estimator -- same seed convention, same n_sim --
#: so "the lift covers" and "the gate passes" are the same number, not two
#: estimates of it. Two estimators here would open a band where the remedy
#: reports covered while the gate keeps firing: the 0.5.0 stuck band in a
#: new coat, with no next-build self-correction to close it.
TAIL_RHO_RESOLUTION = 0.001


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


def _gate_seed(i: int, j: int) -> int:
    """The seed `diagnose_tail_asymmetry` uses for the pair at indices (i, j).

    One place, because two conventions here is a defect class: any check that
    claims to predict the gate must draw the same sample the gate draws.
    """
    return min(i, j) * 1000 + max(i, j)


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
                        copula_df, rho, threshold, n_sim=n_sim, seed=_gate_seed(i, j)
                    ),
                    model_asymptotic=lower_tail_dependence(copula_df, rho),
                    n_lower_exceedances=int((u[:, i] <= threshold).sum()),
                )
            )
    return sorted(out, key=lambda d: d.model_at_threshold - d.empirical_lower)


def uncovered_at_gate(
    diagnostics,
    assets: tuple[str, ...],
    corr: np.ndarray,
    df: float,
    gate_n_sim: int = 200_000,
) -> tuple[tuple[str, str], ...]:
    """The pairs on which §2.3 will fire for a bundle serving `(corr, df)`.

    The gate's own computation, applied ahead of time: model@q re-simulated
    at each pair's SERVED entry with the same per-pair seed and n_sim
    `diagnose_tail_asymmetry` will use, scored through the same
    `understates_lower_tail` margin. The remedy chain's `covered` is this —
    a prediction of the gate, never a summary of its own search — because
    the PD projection can move entries the search never touched, and a pair
    that can hold the gate open while the remedy reports covered is the
    0.5.0 stuck band again. Also run by the embed path after a full-matrix
    re-projection (`service/state.py`), for the same reason.
    """
    index = {a: i for i, a in enumerate(assets)}
    corr = np.asarray(corr, dtype=np.float64)
    out = []
    for d in diagnostics:
        i, j = index[d.pair[0]], index[d.pair[1]]
        refit = replace(
            d,
            model_at_threshold=model_tail_dependence_at_threshold(
                df, float(corr[i, j]), d.threshold,
                n_sim=gate_n_sim, seed=_gate_seed(i, j),
            ),
        )
        if refit.understates_lower_tail():
            out.append(d.pair)
    return tuple(out)


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
    pair_seeds: dict | None = None,
) -> TailFloor:
    """The A11 df floor (option A, 0.5.0): `df* = min(df_ML, df_tail)`.

    Since 0.7.0 this is the SECOND stage of the adopted chain — it composes
    inside `tail_remedy_dependence` on the demands the rho-lift's cap cannot
    reach, on the already-lifted matrix. The contract below is unchanged.

    `pair_seeds` maps a demand's pair to the seed its model@q is scored
    with; the chain passes the GATE's per-pair seeds so the floor's search
    walks the same estimator the gate will apply (standalone default: the
    historical seed 7). Either way the chain's final `covered` verdict comes
    from `uncovered_at_gate` on the served matrix, never from this search.

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
        seed = 7 if pair_seeds is None else int(pair_seeds.get(d.pair, 7))
        targets.append(
            (seed, float(corr[i, j]), d.empirical_lower - one_sided_z * se)
        )

    def covers(df: float) -> bool:
        return all(
            model_tail_dependence_at_threshold(
                df, rho, demands[0].threshold, n_sim=n_sim, seed=seed
            ) >= target
            for seed, rho, target in targets
        )

    candidates = sorted({float(g) for g in grid if g <= df_ml}, reverse=True)
    for df in candidates:
        if covers(df):
            return TailFloor(df_ml=df_ml, df=df, demands=demands, covered=True)
    floor = candidates[-1] if candidates else df_ml
    return TailFloor(df_ml=df_ml, df=floor, demands=demands, covered=False)


@dataclass(frozen=True, slots=True)
class PairLift:
    """One demand pair's rho adjustment, kept for logging and audit."""

    pair: tuple[str, str]
    rho_from: float
    rho_to: float
    #: The demand's one-sided 95% lower confidence bound the lift targets.
    target: float
    #: False when even TAIL_RHO_CAP cannot reach the target at df_ml; the df
    #: floor then composes on top, and failing that the gate stays lit.
    covered: bool

    @property
    def lifted(self) -> bool:
        return self.rho_to > self.rho_from


@dataclass(frozen=True, slots=True)
class TailRemedy:
    """What the A11 remedy chain decided: the rho-lift, then the df floor.

    Adopted 2026-08-05 (OPEN-QUESTIONS A11, option R+H; 0.7.0), after the
    first live firing of the 0.5.0/0.6.0 df floor measured the df lever's
    reach and found it short of the asymmetric pair's bound at the grid wall
    -- and a per-output measurement found it barely moves P(liq) at the live
    correlations anyway. The rho-lever covers where df could not: the demand
    targets are attainable at the ML df with room to spare.
    """

    df_ml: float
    #: The df the bundle should serve: df_ml when the lift covers everything,
    #: the composed floor's value when it does not.
    df: float
    #: The correlation matrix the bundle should serve: demand pairs lifted,
    #: PD-projected. Identical to the input when there were no demands.
    corr: np.ndarray
    lifts: tuple[PairLift, ...]
    demands: tuple[TailDiagnostic, ...]
    #: The gate's verdict on the SERVED (corr, df), computed ahead of time by
    #: `uncovered_at_gate` with the gate's own estimator: False exactly when
    #: §2.3 will fire on this bundle -- never a summary of the search, which
    #: can disagree with the gate through PD-projection redistribution, an
    #: unmeasurable pair, or an unreachable bound.
    covered: bool
    #: The pairs §2.3 will fire on (empty iff `covered`). Named so the log
    #: can say WHICH pair holds the gate, including pairs no demand touched.
    uncovered_pairs: tuple[tuple[str, str], ...]
    #: True when the PD projection moved the lifted matrix -- coverage is
    #: re-verified on the projected entries, never assumed from the search.
    projection_moved: bool

    @property
    def lifted(self) -> bool:
        return any(lift.lifted for lift in self.lifts)

    @property
    def floored(self) -> bool:
        return self.df < self.df_ml


def _smallest_covering_rho(
    df: float,
    rho_from: float,
    target: float,
    threshold: float,
    cap: float,
    resolution: float,
    n_sim: int,
    seed: int,
) -> tuple[float, bool]:
    """The smallest rho in [rho_from, cap] whose model@q reaches `target`.

    model@q is monotone increasing in rho (more dependence, more joint
    exceedances), so bisection finds the smallest covering value; `resolution`
    bounds the interval, and the fixed seed makes the search deterministic.
    Never returns below `rho_from`: the lift raises the modelled co-crash or
    leaves it alone, it does not cut it.

    `seed` and `n_sim` must be the GATE's for this pair (see the caller): the
    search is answering "what will the gate compute", so it must compute the
    same thing.
    """
    # A measured rho already past the cap is NOT cut down to it: the lift's
    # one and only permitted direction is up. Returning `cap` here would
    # serve an entry BELOW the EWMA measurement -- understating measured
    # co-crash dependence, the §10 direction -- to satisfy a bound the pair
    # cannot meet anyway.
    cap = max(cap, rho_from)

    def model_at(rho: float) -> float:
        return model_tail_dependence_at_threshold(
            df, rho, threshold, n_sim=n_sim, seed=seed
        )

    if model_at(rho_from) >= target:
        return rho_from, True
    if model_at(cap) < target:
        return cap, False
    lo, hi = rho_from, cap
    while hi - lo > resolution:
        mid = 0.5 * (lo + hi)
        if model_at(mid) >= target:
            hi = mid
        else:
            lo = mid
    return hi, True


def tail_remedy_dependence(
    diagnostics: list[TailDiagnostic],
    assets: tuple[str, ...],
    corr: np.ndarray,
    df_ml: float,
    grid: np.ndarray = COPULA_DF_GRID,
    demand_sigmas: float = TAIL_DEMAND_SIGMAS,
    one_sided_z: float = TAIL_ONE_SIDED_Z,
    rho_cap: float = TAIL_RHO_CAP,
    rho_resolution: float = TAIL_RHO_RESOLUTION,
    gate_n_sim: int = 200_000,
    metrics=None,
) -> TailRemedy:
    """A11's adopted remedy chain (option R+H): lift rho, then floor df.

    For each pair whose lower-tail shortfall is significant at
    `demand_sigmas` -- the SAME demand set the floor uses, so nothing that
    can hold the §2.3 gate open is invisible to the remedy -- the pair's
    correlation entry is lifted to the smallest value whose model@q at the
    ML df covers the demand's one-sided 95% lower bound. The lifted matrix
    is PD-projected, coverage is RE-VERIFIED on the projected entries, and
    `tail_floor_df` composes on top for any demand the cap could not reach.
    Everything the floor's contract promised still holds: conditional (no
    demand leaves the matrix and the df untouched), one-sided (lifts only,
    never cuts; lambda_U reported, never constrained), recomputed from
    current readings on every build.

    Every coverage check in the chain uses the GATE's estimator,
    deliberately: the same per-pair seed convention and `gate_n_sim` as
    `diagnose_tail_asymmetry` -- in the lift search, in the composed floor's
    walk (via `pair_seeds`), and in the FINAL verdict, which is
    `uncovered_at_gate` over EVERY pair of the served matrix, demands and
    bystanders alike. Two estimators anywhere would open a band where the
    remedy reports covered while the gate keeps firing -- recording mode
    with no remedy left to engage, the 0.5.0 stuck band re-created by the
    very change that closed it. And verifying only the DEMAND pairs would
    miss the other half: the PD projection can redistribute a lift's
    distortion onto entries the search never touched, so a bystander pair
    can be pushed past its own margin (or below its measured correlation)
    by a remedy that never looked at it. The final verdict looks at all of
    them, on the entries that will actually be served.

    Why rho and not (only) df, measured 2026-08-05: at the live readings the
    df lever hits the grid wall short of the asymmetric pair's bound
    (model@q ~0.679 at the 2.5 floor vs a 0.6913 target), while rho* = 0.907
    at the ML df 5.0 covers it with room to spare, costs +0.027 nats/obs in
    the body against +0.10 for chasing the point estimate, and -- per the
    per-output table in OPEN-QUESTIONS A11 -- is the lever that actually
    moves P(liq), where the df floor moved it by <= 0.1 pp. The lift's
    per-output signs vary by book shape (same-sign books rise, hedged books
    fall), which is disclosed there rather than assumed away; the direction
    of every move is TOWARD the measured dependence.

    The body cost is the honest price: the EWMA estimate is the better body
    fit, and the lift overweights body dependence on lifted pairs to buy tail
    coverage inside a one-parameter family. The crash-regime redesign that
    would confine the lift to where the evidence is remains recorded in A11
    as the sharper, larger, deferred model.
    """
    index = {a: i for i, a in enumerate(assets)}
    corr = np.asarray(corr, dtype=np.float64)
    demands = tuple(
        d for d in diagnostics
        if d.understates_lower_tail()
        and np.isfinite(d.lower_excess_sigmas)
        and d.lower_excess_sigmas >= demand_sigmas
    )
    if not demands or len(assets) < 2:
        # No demand does NOT mean the gate is dark: an unmeasurable pair
        # (NaN empirical lower, or a zero-SE reading whose excess is not
        # finite) fires `understates_lower_tail` while carrying no bound the
        # remedy could target. `covered` must say what the gate will say.
        firing = tuple(d.pair for d in diagnostics if d.understates_lower_tail())
        return TailRemedy(
            df_ml=df_ml, df=df_ml, corr=corr, lifts=(), demands=(),
            covered=not firing, uncovered_pairs=firing,
            projection_moved=False,
        )

    lifted = np.array(corr, copy=True)
    lifts = []
    for d in demands:
        i, j = index[d.pair[0]], index[d.pair[1]]
        target = float(d.empirical_lower - one_sided_z * d.lower_standard_error)
        rho_to, reached = _smallest_covering_rho(
            df_ml, float(corr[i, j]), target, d.threshold,
            rho_cap, rho_resolution, gate_n_sim,
            seed=_gate_seed(i, j),
        )
        lifts.append(PairLift(
            pair=d.pair, rho_from=float(corr[i, j]), rho_to=rho_to,
            target=target, covered=reached,
        ))
        lifted[i, j] = lifted[j, i] = rho_to

    projection = project_to_correlation(lifted, metrics=metrics, label="tail_lift")
    final = np.asarray(projection.corr, dtype=np.float64)

    # Demand coverage is judged on the matrix that will actually be served,
    # never on the search result: the projection may pull a lifted entry
    # back down. Same estimator as the gate, for the same reason as in the
    # search. This selects the FLOOR's inputs; the chain's verdict comes
    # from the all-pairs check below.
    uncovered = []
    for d, lift in zip(demands, lifts, strict=True):
        i, j = index[d.pair[0]], index[d.pair[1]]
        reached = model_tail_dependence_at_threshold(
            df_ml, float(final[i, j]), d.threshold, n_sim=gate_n_sim,
            seed=_gate_seed(i, j),
        ) >= lift.target
        if not reached:
            uncovered.append(d)

    if uncovered:
        # The floor composes on the LIFTED matrix: lowering df raises model@q
        # for every pair at fixed rho, so pairs the lift already covers stay
        # covered and only the residual demands drive the search. It walks
        # the gate's own estimator (pair_seeds, gate_n_sim): a floor chosen
        # under a different seed can sit below the gate's reading of the
        # same point, and the boundary has zero slack.
        floor = tail_floor_df(
            uncovered, assets, final, df_ml,
            grid=grid, demand_sigmas=demand_sigmas,
            one_sided_z=one_sided_z, n_sim=gate_n_sim,
            pair_seeds={
                d.pair: _gate_seed(index[d.pair[0]], index[d.pair[1]])
                for d in uncovered
            },
        )
        df = floor.df
    else:
        df = df_ml

    # The verdict. Every pair of the served matrix, not just the demands:
    # the projection can redistribute the lift onto bystander pairs, and a
    # pair that can hold the gate open must never be invisible to the
    # remedy's report of itself.
    firing = uncovered_at_gate(diagnostics, assets, final, df, gate_n_sim)
    return TailRemedy(
        df_ml=df_ml, df=df, corr=final, lifts=tuple(lifts), demands=demands,
        covered=not firing, uncovered_pairs=firing,
        projection_moved=bool(projection.corrected),
    )


def assert_lower_tail_not_understated(
    diagnostics: list[TailDiagnostic], margin: float = 0.05
) -> None:
    """§2.3: understating the lower tail is a blocking defect.

    Raises rather than warns. §2.3 names a skewed-t as the remedy, and that
    prescription was REFUTED by measurement (OPEN-QUESTIONS A11, all four
    findings, 2026-08-04): not per-output conservative, correlation-destroying
    below nu=4, and unable to reach the observed upper tails at any admissible
    skew. The adopted remedy is the A11 chain (`tail_remedy_dependence`: the
    conditional rho-lift, then the df floor), which runs BEFORE this gate in
    the bundle build -- so this firing means the demands exceed what the
    lift's cap and the family's grid can cover together, and the honest
    response is still to stop and report, not to continue with a model known
    to understate the risk the product exists to measure (§9, §10).
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
            "A11 chain, rho-lift then df floor -- runs before this gate, so this "
            "firing means the chain could not produce a served matrix that passes "
            "it: an unreachable demand, an unmeasurable pair, or PD-projection "
            "redistribution (the remedy log names which, per pair). Report "
            f"this rather than proceeding.{note}"
            f"{shape_note}"
        )
