"""The five analytical benchmarks of §3.1 — the Phase 1 gate.

A Monte Carlo engine does not crash when it is wrong; it returns a confident
incorrect number. These five checks are the only thing standing between a
sign error in the liquidation condition and a user being told their book is
safe. They are run by `python -m risk_engine.validation.cli` and by pytest.

Three of them compare probabilities whose true difference is comparable to,
or smaller than, the Monte Carlo noise on either estimate. Those are run
under **common random numbers**: the configurations share one
`BaseRandomness`, so the comparison is pathwise rather than statistical.
Without that, §3.1.2's 0.3 pp tolerance is one standard error at 20 000
paths and §3.1.5's "strictly increasing over 20 grid points" is a coin flip
on a correct engine (OPEN-QUESTIONS A6).

Nothing here is tolerated by widening a threshold. If a benchmark fails, the
result carries the measured number and the CLI exits non-zero.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from itertools import pairwise

import numpy as np
from scipy import stats

from risk_engine.domain.types import AssetSpec, Book, MarginMode, MarginTier, Position
from risk_engine.liquidation.margin import liquidation_price
from risk_engine.model.drift import DriftConvention
from risk_engine.sim.engine import simulate_paths
from risk_engine.sim.paths import BaseRandomness, PathSpec, draw_base_randomness

#: Benchmarks use more paths than production. They are an offline gate, and
#: the tolerances in §3.1 are only meaningful once MC noise is well below
#: them; see the module docstring.
BENCHMARK_PATHS = 200_000
#: Correlations of exactly +/-1 make the correlation matrix singular and
#: Cholesky fails. Backing off by 1e-12 leaves an independent component of
#: size 1e-6 sigma, which is nine orders below any effect being measured.
UNIT_RHO = 1.0 - 1e-12


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    passed: bool
    detail: str
    measured: float | None = None
    expected: float | None = None
    tolerance: float | None = None

    def __str__(self) -> str:
        flag = "PASS" if self.passed else "FAIL"
        return f"[{flag}] {self.name}: {self.detail}"


def synthetic_spec(name: str, max_leverage: float = 20.0) -> AssetSpec:
    """A single-tier asset, for benchmarks only.

    Single tier on purpose: §3.1.1's closed form assumes a fixed barrier, and
    a tier boundary crossed on the way down would move it. The tier logic has
    its own tests in `test_liquidation.py`.
    """
    return AssetSpec(name, 5, max_leverage, (MarginTier(0.0, max_leverage),))


def _flat_spec(coins: tuple[str, ...], vol: float, rho: np.ndarray | None,
               df: float | None, copula_df: float | None) -> PathSpec:
    a = len(coins)
    corr = np.eye(a) if rho is None else rho
    return PathSpec(
        coins=coins,
        step_vol=np.full(a, vol),
        corr=corr,
        marginal_df=(df,) * a,
        copula_df=copula_df,
        drift=DriftConvention.ZERO_LOG_RETURN,
    )


def _p_liq(book: Book, specs: dict[str, AssetSpec], spec: PathSpec,
           spot: np.ndarray, base: BaseRandomness, use_bridge: bool = True) -> float:
    out = simulate_paths(book, specs, spec, spot, base, funding_paths=None,
                         use_bridge=use_bridge)
    hits = out.cross_liq
    if out.iso_liq.size:
        hits = hits | out.iso_liq.any(axis=1)
    return float(hits.mean())


# ---------------------------------------------------------------------------
# §3.1.1 — closed-form barrier probability
# ---------------------------------------------------------------------------

def gbm_first_passage(
    spot: float, barrier: float, step_vol: float, n_steps: int, drift_per_step: float = 0.0
) -> float:
    """P(min_{t<=T} S_t <= barrier) for a GBM, continuous monitoring.

    Standard reflection result on the log price, which is Brownian with drift
    `drift_per_step` and volatility `step_vol` in units of one step.
    """
    if barrier >= spot:
        raise ValueError("this form is for a down barrier")
    b = np.log(barrier / spot)
    t = float(n_steps)
    nu, s = drift_per_step, step_vol
    first = stats.norm.cdf((b - nu * t) / (s * np.sqrt(t)))
    second = np.exp(2 * nu * b / s**2) * stats.norm.cdf((b + nu * t) / (s * np.sqrt(t)))
    return float(first + second)


def benchmark_1_closed_form(
    n_paths: int = BENCHMARK_PATHS, seed: int = 11, tolerance: float = 0.005
) -> BenchmarkResult:
    """One position, no funding, Gaussian marginals, against the exact answer.

    The simulator monitors hourly and applies the Brownian-bridge correction,
    so the target is the *continuous* first-passage probability with no
    discrete-monitoring shift. That is the whole point of the correction: an
    uncorrected hourly simulator would land well below this number, and the
    gap would be a real understatement of risk, not a benchmark artefact.
    """
    # Collateral is chosen to put the barrier probability near 15%. At the
    # 0.5% probability a round-numbered book produces, a 0.5 pp tolerance is
    # 100x the quantity being measured and the benchmark would pass on an
    # engine that was wrong by an order of magnitude.
    coin, entry, size, collateral = "SYN", 100.0, 100.0, 915.0
    step_vol, n_steps = 0.01, 24
    specs = {coin: synthetic_spec(coin)}
    now = datetime.now(timezone.utc)
    book = Book("0xbench", collateral, (Position(coin, size, entry, MarginMode.CROSS, 10.0),), now)

    barrier = liquidation_price(book.positions[0], book, {coin: entry}, specs)
    expected = gbm_first_passage(entry, barrier, step_vol, n_steps)

    spec = _flat_spec((coin,), step_vol, None, df=None, copula_df=None)
    rng = np.random.default_rng(seed)
    base = draw_base_randomness(n_paths, n_steps, 1, 0, None, rng)
    got = _p_liq(book, specs, spec, np.array([entry]), base)

    diff = abs(got - expected)
    return BenchmarkResult(
        name="3.1.1 closed-form first passage",
        passed=diff <= tolerance,
        detail=(
            f"barrier={barrier:.4f} simulated={got:.5f} closed_form={expected:.5f} "
            f"|diff|={diff * 100:.3f}pp tol={tolerance * 100:.1f}pp"
        ),
        measured=got,
        expected=expected,
        tolerance=tolerance,
    )


def benchmark_1b_bridge_matters(
    n_paths: int = 100_000, seed: int = 12
) -> BenchmarkResult:
    """Not in §3.1, but it pins the claim §1 rests on.

    Hourly close-only monitoring must produce a *lower* P(liq) than
    continuous monitoring. If this ever fails, the bridge correction is not
    doing what OPEN-QUESTIONS A3 says it does, and the engine is understating
    risk in the way §10 prohibits.
    """
    coin, entry, size, collateral = "SYN", 100.0, 100.0, 915.0
    step_vol, n_steps = 0.01, 24
    specs = {coin: synthetic_spec(coin)}
    now = datetime.now(timezone.utc)
    book = Book("0xbench", collateral, (Position(coin, size, entry, MarginMode.CROSS, 10.0),), now)
    spec = _flat_spec((coin,), step_vol, None, df=None, copula_df=None)
    base = draw_base_randomness(n_paths, n_steps, 1, 0, None, np.random.default_rng(seed))

    with_bridge = _p_liq(book, specs, spec, np.array([entry]), base, use_bridge=True)
    without = _p_liq(book, specs, spec, np.array([entry]), base, use_bridge=False)
    return BenchmarkResult(
        name="3.1.1b discrete monitoring understates risk",
        passed=without < with_bridge,
        detail=f"close-only={without:.5f} < bridge-corrected={with_bridge:.5f}",
        measured=without,
        expected=with_bridge,
    )


# ---------------------------------------------------------------------------
# §3.1.2 — rho = 1 collapses to a single doubled position
# ---------------------------------------------------------------------------

def benchmark_2_perfect_correlation(
    n_paths: int = BENCHMARK_PATHS, seed: int = 21, tolerance: float = 0.003
) -> BenchmarkResult:
    entry, size, collateral = 100.0, 50.0, 915.0
    step_vol, n_steps = 0.01, 24
    now = datetime.now(timezone.utc)

    pair_specs = {"A1": synthetic_spec("A1"), "A2": synthetic_spec("A2")}
    pair = Book(
        "0xbench",
        collateral,
        (
            Position("A1", size, entry, MarginMode.CROSS, 10.0),
            Position("A2", size, entry, MarginMode.CROSS, 10.0),
        ),
        now,
    )
    single_specs = {"A1": synthetic_spec("A1")}
    single = Book(
        "0xbench", collateral,
        (Position("A1", 2 * size, entry, MarginMode.CROSS, 10.0),), now,
    )

    rho = np.array([[1.0, UNIT_RHO], [UNIT_RHO, 1.0]])
    pair_spec = _flat_spec(("A1", "A2"), step_vol, rho, df=4.0, copula_df=4.0)
    single_spec = _flat_spec(("A1",), step_vol, None, df=4.0, copula_df=4.0)

    base = draw_base_randomness(n_paths, n_steps, 2, 0, 4.0, np.random.default_rng(seed))
    # Common random numbers: the single-asset run reuses column 0 of the same
    # draws, so with rho = 1 both books see the identical price path.
    single_base = BaseRandomness(
        z=base.z[:, :, :1], chi=base.chi, copula_df=base.copula_df,
        bridge_cross=base.bridge_cross, bridge_iso=base.bridge_iso,
    )

    p_pair = _p_liq(pair, pair_specs, pair_spec, np.array([entry, entry]), base)
    p_single = _p_liq(single, single_specs, single_spec, np.array([entry]), single_base)

    diff = abs(p_pair - p_single)
    return BenchmarkResult(
        name="3.1.2 rho=1 equals one doubled position",
        passed=diff <= tolerance,
        detail=(
            f"two positions={p_pair:.5f} one doubled={p_single:.5f} "
            f"|diff|={diff * 100:.4f}pp tol={tolerance * 100:.1f}pp"
        ),
        measured=p_pair,
        expected=p_single,
        tolerance=tolerance,
    )


# ---------------------------------------------------------------------------
# §3.1.3 — monotone in correlation
# ---------------------------------------------------------------------------

def benchmark_3_correlation_monotonicity(
    n_paths: int = BENCHMARK_PATHS, seed: int = 31
) -> BenchmarkResult:
    entry, size, collateral = 100.0, 50.0, 1_200.0
    step_vol, n_steps = 0.01, 24
    now = datetime.now(timezone.utc)
    specs = {"A1": synthetic_spec("A1"), "A2": synthetic_spec("A2")}
    book = Book(
        "0xbench", collateral,
        (
            Position("A1", size, entry, MarginMode.CROSS, 10.0),
            Position("A2", size, entry, MarginMode.CROSS, 10.0),
        ),
        now,
    )
    base = draw_base_randomness(n_paths, n_steps, 2, 0, 4.0, np.random.default_rng(seed))
    spot = np.array([entry, entry])

    probs = {}
    for label, r in (("-1", -UNIT_RHO), ("0", 0.0), ("+1", UNIT_RHO)):
        corr = np.array([[1.0, r], [r, 1.0]])
        spec = _flat_spec(("A1", "A2"), step_vol, corr, df=4.0, copula_df=4.0)
        probs[label] = _p_liq(book, specs, spec, spot, base)

    ok = probs["-1"] < probs["0"] < probs["+1"]
    return BenchmarkResult(
        name="3.1.3 P(liq) monotone in correlation",
        passed=ok,
        detail=(
            f"rho=-1: {probs['-1']:.5f} < rho=0: {probs['0']:.5f} "
            f"< rho=+1: {probs['+1']:.5f}"
        ),
        measured=probs["+1"],
    )


# ---------------------------------------------------------------------------
# §3.1.4 — 1x leverage is safe
# ---------------------------------------------------------------------------

def benchmark_4_unlevered_is_safe(
    n_paths: int = BENCHMARK_PATHS, seed: int = 41, threshold: float = 0.001
) -> BenchmarkResult:
    """At 1x, equity is the position value and maintenance margin is a
    fraction of it, so the gap cannot close at any positive price. The check
    is structural rather than statistical, and it catches sign errors in the
    margin arithmetic that a probability comparison would not."""
    entry, size = 100.0, 100.0
    notional = entry * size
    step_vol, n_steps = 0.02, 24
    now = datetime.now(timezone.utc)
    specs = {"A1": synthetic_spec("A1"), "A2": synthetic_spec("A2")}

    cross = Book("0xbench", notional * 2,
                 (Position("A1", size, entry, MarginMode.CROSS, 1.0),
                  Position("A2", size, entry, MarginMode.CROSS, 1.0)), now)
    isolated = Book("0xbench", 0.0,
                    (Position("A1", size, entry, MarginMode.ISOLATED, 1.0, notional),), now)

    corr = np.array([[1.0, 0.8], [0.8, 1.0]])
    spec2 = _flat_spec(("A1", "A2"), step_vol, corr, df=4.0, copula_df=4.0)
    spec1 = _flat_spec(("A1",), step_vol, None, df=4.0, copula_df=4.0)
    base2 = draw_base_randomness(n_paths, n_steps, 2, 0, 4.0, np.random.default_rng(seed))
    base1 = draw_base_randomness(n_paths, n_steps, 1, 1, 4.0, np.random.default_rng(seed))

    p_cross = _p_liq(cross, specs, spec2, np.array([entry, entry]), base2)
    p_iso = _p_liq(isolated, specs, spec1, np.array([entry]), base1)
    worst = max(p_cross, p_iso)
    return BenchmarkResult(
        name="3.1.4 unlevered book does not liquidate",
        passed=worst < threshold,
        detail=f"cross 1x={p_cross:.6f} isolated 1x={p_iso:.6f} threshold={threshold}",
        measured=worst,
        tolerance=threshold,
    )


# ---------------------------------------------------------------------------
# §3.1.5 — monotone in leverage over a 20-point grid
# ---------------------------------------------------------------------------

def benchmark_5_leverage_monotonicity(
    n_paths: int = 40_000, seed: int = 51, n_grid: int = 20
) -> BenchmarkResult:
    """Two readings of "leverage", because §3.1.5 does not say which.

    For a cross book the leverage *slider* provably does not move the
    liquidation price (§1.2), so the only meaningful grid is over effective
    leverage: notional at fixed collateral. For an isolated position the set
    leverage does move it, so that grid is over `L` directly. Both must be
    strictly increasing (OPEN-QUESTIONS A7).

    The grid starts at 8x rather than at 1x. Below roughly 8x, liquidation
    inside 24 h requires a move the model assigns probability zero at an
    hourly vol of 1%, and "strictly increasing" cannot hold across a region
    where the true probability is flat at zero -- that would be a demand on
    the grid, not on the engine. The grid is chosen so every point is in the
    non-degenerate range; §3.1.4 separately pins the zero end.
    """
    entry, collateral = 100.0, 1_000.0
    step_vol, n_steps = 0.01, 24
    now = datetime.now(timezone.utc)
    specs = {"A1": synthetic_spec("A1", max_leverage=25.0)}
    spec = _flat_spec(("A1",), step_vol, None, df=4.0, copula_df=4.0)
    base = draw_base_randomness(n_paths, n_steps, 1, 1, 4.0, np.random.default_rng(seed))
    # Same draws for both grids; the cross books have no isolated pocket, so
    # they take a zero-width slice of the isolated bridge uniforms rather
    # than a separate draw, which keeps the two grids on common randomness.
    cross_base = replace(base, bridge_iso=base.bridge_iso[:, :, :0])
    spot = np.array([entry])

    grid = np.linspace(8.0, 24.0, n_grid)

    cross_probs = []
    for lev in grid:
        size = collateral * lev / entry
        book = Book("0xbench", collateral,
                    (Position("A1", size, entry, MarginMode.CROSS, 25.0),), now)
        cross_probs.append(_p_liq(book, specs, spec, spot, cross_base))

    iso_probs = []
    size = 100.0
    for lev in grid:
        margin = entry * size / lev
        book = Book("0xbench", 0.0,
                    (Position("A1", size, entry, MarginMode.ISOLATED, lev, margin),), now)
        iso_probs.append(_p_liq(book, specs, spec, spot, base))

    # The other half of A7, and the reason the cross grid is over EFFECTIVE
    # leverage: §1.2 says the slider does not move a cross liquidation price
    # at all. That is the claim that makes gridding the slider a test of the
    # spec's wording rather than of the engine, so it is asserted rather than
    # relied on. Measured invariant to 6 decimal places across 5x/10x/25x.
    fixed_size = collateral * 12.0 / entry
    slider_probs = [
        _p_liq(
            Book("0xbench", collateral,
                 (Position("A1", fixed_size, entry, MarginMode.CROSS, set_lev),), now),
            specs, spec, spot, cross_base,
        )
        for set_lev in (5.0, 10.0, 25.0)
    ]
    slider_ok = max(slider_probs) - min(slider_probs) == 0.0

    cross_ok = all(b > a for a, b in pairwise(cross_probs))
    iso_ok = all(b > a for a, b in pairwise(iso_probs))
    bad_cross = [i for i, (a, b) in enumerate(pairwise(cross_probs)) if b <= a]
    bad_iso = [i for i, (a, b) in enumerate(pairwise(iso_probs)) if b <= a]
    return BenchmarkResult(
        name="3.1.5 P(liq) strictly increasing in leverage",
        passed=cross_ok and iso_ok and slider_ok,
        detail=(
            f"cross(effective) {cross_probs[0]:.5f}..{cross_probs[-1]:.5f} "
            f"{'ok' if cross_ok else f'violations at {bad_cross}'}; "
            f"isolated(set L) {iso_probs[0]:.5f}..{iso_probs[-1]:.5f} "
            f"{'ok' if iso_ok else f'violations at {bad_iso}'}; "
            f"cross slider invariant (§1.2) {slider_probs[0]:.6f} "
            f"{'ok' if slider_ok else f'MOVED: {slider_probs}'}"
        ),
    )


ALL_BENCHMARKS = (
    benchmark_1_closed_form,
    benchmark_1b_bridge_matters,
    benchmark_2_perfect_correlation,
    benchmark_3_correlation_monotonicity,
    benchmark_4_unlevered_is_safe,
    benchmark_5_leverage_monotonicity,
)


def run_all() -> list[BenchmarkResult]:
    return [fn() for fn in ALL_BENCHMARKS]
