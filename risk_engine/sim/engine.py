"""Monte Carlo engine (§2.5, §2.6).

Structure follows the split in §2.6: everything expensive and shared is
offline (the EWMA, the shrinkage, the projection, the global Cholesky), and
the online path is a submatrix slice plus path generation. Each stage is
timed separately, because a total-latency number cannot tell you which half
regressed.

Path count is adaptive, not fixed. §2.5 wants 20 000 paths *and* a 95%
interval on P(liq) no wider than 2 pp; when those disagree the interval
wins and the run is extended (OPEN-QUESTIONS D1). A result that could not
reach the target interval is returned with `converged=False`.

Two corrections to what this docstring used to claim, both measured:

**Nothing in `tools/` refuses to publish it.** `portfolio_risk` and
`pre_trade_delta` return the full result with every point estimate populated
and expose `publishable` alongside; the service serves it at HTTP 200. The
refusal is real but lives at the TypeScript boundary
(`services/backend/src/risk/client.ts`, `staleness/contract.ts`), which is
also where §6's degradation contract lives, so a Python consumer added later
inherits no protection at all.

**At shipped defaults `converged=False` is unreachable.** With
`DEFAULT_PATHS = 20_000` the worst-case Wilson half-width over every possible
success count is 0.0069, well inside the 0.02 target, so the first pass always
converges and the escalation loop never reaches `MAX_PATHS`. The flag can only
fire on a caller-supplied path count below ~2 400. That is not a defect —
§2.5's rule being satisfied by construction is the desired outcome — but it
means `mc_non_convergence` is pinned at zero, and "the counter is quiet
because all is well" is indistinguishable from "the counter cannot fire".
Anyone tightening `DEFAULT_TARGET_HALF_WIDTH` or lowering `MAX_PATHS` re-opens
the path and should re-read the paragraph above about where the refusal is.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import (
    AssetSpec,
    Book,
    RiskEstimate,
    SimulationProvenance,
)
from risk_engine.liquidation.simulator import BridgeContext, LiquidationSimulator
from risk_engine.model.correlation import GlobalCorrelationMatrix
from risk_engine.model.drift import DriftConvention
from risk_engine.model.funding import Ar1Funding, FundingBounds, simulate_funding
from risk_engine.model.marginals import MarginalSpec
from risk_engine.observability.metrics import METRICS, Metrics, Timer
from risk_engine.sim.paths import (
    BaseRandomness,
    PathSpec,
    draw_base_randomness,
    generate_price_paths,
)
from risk_engine.sim.stats import (
    PredictiveDistribution,
    bootstrap_ci,
    conditional_value_at_risk,
    paths_needed_for_half_width,
    value_at_risk,
    wilson_half_width,
    wilson_interval,
)
from risk_engine.version import MODEL_VERSION

DEFAULT_PATHS = 20_000
DEFAULT_TARGET_HALF_WIDTH = 0.02  # §2.5: 2 percentage points
MAX_PATHS = 320_000
DEFAULT_CHUNK = 5_000
#: Threads used to walk path blocks concurrently (OPEN-QUESTIONS D7).
#: Threads rather than processes because numpy releases the GIL on the large
#: array operations that dominate, so the arrays stay shared and nothing has
#: to be pickled across a boundary.
#:
#: The default is 2, not the core count. Measured on a contended four-core
#: container with a six-position book (min of seven runs): serial 656 ms,
#: two threads 418 ms, four threads 648 ms. Beyond two, oversubscription
#: against whatever else shares the box costs more than the parallelism
#: buys, and the liquidation walk's per-step Python loop does not
#: parallelise anyway. The optimum is hardware-dependent -- tune
#: RISK_ENGINE_WORKERS on the deployment target rather than trusting this
#: number, which was measured somewhere else.
DEFAULT_WORKERS = max(1, min(int(os.environ.get("RISK_ENGINE_WORKERS", "2")), os.cpu_count() or 1))
#: Floats per block of price paths, roughly 32 MB. Multiplied by the worker
#: count for peak memory, which is why it is a per-block figure and not a
#: total.
CHUNK_FLOAT_BUDGET = 4_000_000
#: Below this a block costs more in per-block overhead than it saves.
MIN_CHUNK = 2_000


def plan_chunks(
    n_paths: int,
    per_path_floats: int,
    workers: int = DEFAULT_WORKERS,
    min_chunk: int = MIN_CHUNK,
    float_budget: int = CHUNK_FLOAT_BUDGET,
) -> list[int]:
    """Split `n_paths` into blocks that are both memory-bounded and parallel.

    Two constraints pull in opposite directions. Memory wants blocks no
    larger than the float budget. Parallelism wants at least one block per
    worker, which for a 20 000-path request over a handful of assets means
    *smaller* blocks than memory alone would pick -- the previous adaptive
    sizing produced a single block, which left three of four cores idle.

    Sizes are balanced rather than "full blocks plus a remainder", so no
    worker finishes early and waits on a straggler.
    """
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    memory_cap = max(1, float_budget // max(per_path_floats, 1))
    parallel_target = max(min_chunk, math.ceil(n_paths / max(workers, 1)))
    size = max(1, min(memory_cap, parallel_target, n_paths))
    n_chunks = math.ceil(n_paths / size)
    base, remainder = divmod(n_paths, n_chunks)
    return [base + (1 if i < remainder else 0) for i in range(n_chunks)]


def spawn_streams(seed: int, n: int) -> list[np.random.Generator]:
    """Independent, reproducible generators, one per block.

    `SeedSequence.spawn` gives streams that are statistically independent and
    a deterministic function of the seed, so a parallel run reproduces
    exactly and does not depend on the order blocks happen to finish in.
    Drawing blocks sequentially from one generator, as the serial version
    did, cannot be parallelised without exactly that dependence.
    """
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


@dataclass(frozen=True, slots=True)
class ModelBundle:
    """The offline half of §2.6, ready to be sliced."""

    matrix: GlobalCorrelationMatrix
    marginals: dict[str, MarginalSpec]
    funding: dict[str, Ar1Funding]
    funding_bounds: FundingBounds
    copula_df: float | None
    drift: DriftConvention = DriftConvention.ZERO_LOG_RETURN
    model_version: str = MODEL_VERSION
    #: §2.3's tail-asymmetry diagnostic, as measured on the returns this
    #: bundle was fitted from. Empty when the bundle was assembled by hand
    #: (tests, benchmarks) rather than from a return series.
    #:
    #: Carried on the bundle rather than discarded at build time because a
    #: passing diagnostic is a *result*, not the absence of one: it says the
    #: symmetric copula was checked against this particular market and found
    #: adequate, and that claim expires when the market does. `assert_...`
    #: below is what refuses; this is what lets an operator see the margin.
    tail_diagnostics: tuple = ()

    def path_spec(self, coins: tuple[str, ...], independent: bool = False) -> PathSpec:
        with Timer("slice_submatrix"):
            corr = np.eye(len(coins)) if independent else self.matrix.submatrix(coins)
            vol = self.matrix.volatilities(coins)
        missing = [c for c in coins if c not in self.marginals]
        if missing:
            raise KeyError(f"no fitted marginal for {missing}")
        # Audit A-03: rho=0 under a t-copula is NOT independence -- the shared
        # chi-square mixer still crashes every asset together (measured:
        # 12.95% joint lower-tail frequency where independence gives 5%).
        # §3.2's baseline B demands independent paths, and a Gaussian copula
        # with an identity matrix is the elliptical copula for which
        # uncorrelated actually means independent. Marginals are untouched.
        return PathSpec(
            coins=coins,
            step_vol=vol,
            corr=corr,
            marginal_df=tuple(self.marginals[c].df for c in coins),
            copula_df=None if independent else self.copula_df,
            drift=self.drift,
        )

    def funding_models(self, coins: tuple[str, ...]) -> list[Ar1Funding | None]:
        return [self.funding.get(c) for c in coins]


@dataclass(frozen=True, slots=True)
class RiskResult:
    p_liq_any: RiskEstimate
    p_liq_cross: RiskEstimate
    p_liq_isolated: dict[str, RiskEstimate]
    cvar_95_usd: RiskEstimate
    var_95_usd: float
    equity_change: PredictiveDistribution
    funding_cost: PredictiveDistribution
    start_equity: float
    n_paths: int
    converged: bool
    provenance: SimulationProvenance
    #: Kept out of the journal; used by max_safe_size and the benchmarks.
    raw_equity_change: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    factor_return: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))
    factor_coin: str | None = None

    @property
    def equity_return(self) -> PredictiveDistribution:
        v = self.equity_change
        return PredictiveDistribution(levels=v.levels, values=v.values / self.start_equity)


@dataclass(frozen=True, slots=True)
class _RawOutcome:
    """Concatenated per-path results before they become estimates."""

    cross_liq: np.ndarray
    iso_liq: np.ndarray
    equity_change: np.ndarray
    funding_paid: np.ndarray
    isolated_coins: tuple[str, ...]
    start_equity: float
    #: Simple return of the risk factor (BTC) on each path, for §4.1's
    #: effective leverage and beta. Empty when no factor column was requested.
    factor_return: np.ndarray = field(default_factory=lambda: np.empty(0))


def simulate_books_checkpointed(
    books: Sequence[Book],
    specs: dict[str, AssetSpec],
    path_spec: PathSpec,
    spot: np.ndarray,
    base: BaseRandomness,
    funding_paths: np.ndarray | None,
    horizons: tuple[int, ...],
    use_bridge: bool = True,
    factor_col: int | None = None,
) -> list[dict[int, _RawOutcome]]:
    """Several books walked over ONE set of price paths.

    This is what makes `pre_trade_delta` (§4.2) meaningful. The quantity the
    user is shown is a *difference* between two books that overlap almost
    entirely, and it is usually far smaller than the Monte Carlo error on
    either side. Simulating the two books independently would report mostly
    sampling noise. Walking both over identical prices makes the comparison
    pathwise: the difference is exactly the effect of the order.

    It is also half the work -- path generation dominates the online budget
    in §2.6, and it happens once here rather than once per book.
    """
    columns = {c: i for i, c in enumerate(path_spec.coins)}
    with Timer("generate_paths"):
        prices = generate_price_paths(path_spec, spot, base)
    bridge = (
        BridgeContext(step_vol=path_spec.step_vol, corr=path_spec.corr) if use_bridge else None
    )

    results: list[dict[int, _RawOutcome]] = []
    for book in books:
        sim = LiquidationSimulator(book, specs, columns)
        # The bridge uniforms come from `base`, so they are shared across
        # books and configurations; drawing them here would break common
        # random numbers. Each book takes as many isolated columns as it has
        # pockets, from the same pool.
        uniforms = None
        if use_bridge:
            # Keyed by COIN, not by the pocket's position in the book.
            # `Book.with_position` rebuilds the tuple as `(*others, merged)`,
            # so an order touching a coin moves that pocket to the end: a book
            # with isolated [SOL, ETH] becomes [ETH, SOL] after an order on
            # SOL. Slicing `[:n_iso]` positionally then fed column 0 to SOL in
            # the "before" book and to ETH in the "after" book, decoupling the
            # interior-hit draws for exactly the paired walk `pre_trade_delta`
            # exists to make pathwise. Costs variance, not bias -- the price
            # paths were always shared -- but it widens the very interval D6
            # measured as six times tighter.
            iso_cols = [columns[p.coin] for p in book.isolated_positions]
            if iso_cols and base.bridge_iso.shape[2] <= max(iso_cols):
                raise ValueError(
                    f"randomness has {base.bridge_iso.shape[2]} isolated columns, "
                    f"book needs a column for universe index {max(iso_cols)}"
                )
            uniforms = (base.bridge_cross, base.bridge_iso[:, :, iso_cols])
        with Timer("liquidation_walk"):
            outs = sim.run_checkpointed(
                prices, horizons, funding_paths=funding_paths, bridge=bridge,
                bridge_uniforms=uniforms,
            )
        per_horizon: dict[int, _RawOutcome] = {}
        for h, out in outs.items():
            factor = (
                prices[:, h, factor_col] / prices[:, 0, factor_col] - 1.0
                if factor_col is not None
                else np.empty(0)
            )
            per_horizon[h] = _RawOutcome(
                cross_liq=out.cross_liquidated,
                iso_liq=out.isolated_liquidated,
                equity_change=out.equity_change,
                funding_paid=out.funding_paid,
                isolated_coins=out.isolated_coins,
                start_equity=out.start_equity,
                factor_return=factor,
            )
        results.append(per_horizon)
    return results


def simulate_paths_checkpointed(
    book: Book,
    specs: dict[str, AssetSpec],
    path_spec: PathSpec,
    spot: np.ndarray,
    base: BaseRandomness,
    funding_paths: np.ndarray | None,
    horizons: tuple[int, ...],
    use_bridge: bool = True,
    factor_col: int | None = None,
) -> dict[int, _RawOutcome]:
    """One book, snapshotted at each horizon.

    Deterministic given `base`, which is what makes common random numbers
    possible for the benchmarks in §3.1. All horizons come from the SAME
    walk (audit A-10): alive sets only ever shrink, so the liquidation flags
    at a longer horizon are a pathwise superset of a shorter one.
    """
    return simulate_books_checkpointed(
        (book,), specs, path_spec, spot, base, funding_paths, horizons,
        use_bridge, factor_col,
    )[0]


def simulate_paths(
    book: Book,
    specs: dict[str, AssetSpec],
    path_spec: PathSpec,
    spot: np.ndarray,
    base: BaseRandomness,
    funding_paths: np.ndarray | None,
    use_bridge: bool = True,
    factor_col: int | None = None,
) -> _RawOutcome:
    """Single-horizon convenience wrapper; the §3.1 benchmarks use this."""
    n_steps = base.z.shape[1]
    return simulate_paths_checkpointed(
        book, specs, path_spec, spot, base, funding_paths, (n_steps,), use_bridge, factor_col
    )[n_steps]


def run_blocks(
    *,
    books: Sequence[Book],
    specs: dict[str, AssetSpec],
    bundle: ModelBundle,
    spec: PathSpec,
    spot_vec: np.ndarray,
    n_paths: int,
    horizons: tuple[int, ...],
    seed: int,
    n_iso: int,
    include_funding: bool,
    use_bridge: bool = True,
    factor_col: int | None = None,
    workers: int = DEFAULT_WORKERS,
) -> list[dict[int, _RawOutcome]]:
    """Draw and walk `n_paths` in parallel blocks, for every book at once.

    Every book sees the same paths (the common random numbers `pre_trade_delta`
    depends on), and every block runs on its own thread with its own
    independent stream. The result is deterministic in `seed` and independent
    of completion order, because each block's randomness is decided before
    any thread starts.
    """
    longest = max(horizons)
    per_path = (longest + 1) * spec.n_assets
    sizes = plan_chunks(n_paths, per_path, workers)
    streams = spawn_streams(seed, len(sizes))
    funding_models = bundle.funding_models(spec.coins) if include_funding else None

    def one_block(index: int) -> list[dict[int, _RawOutcome]]:
        size = sizes[index]
        rng = streams[index]
        base = draw_base_randomness(size, longest, spec.n_assets, n_iso, spec.copula_df, rng)
        funding = (
            simulate_funding(funding_models, bundle.funding_bounds, size, longest, rng)
            if include_funding
            else None
        )
        return simulate_books_checkpointed(
            books, specs, spec, spot_vec, base, funding, horizons, use_bridge, factor_col,
        )

    if len(sizes) == 1 or workers <= 1:
        results = [one_block(0)]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(sizes))) as pool:
            results = list(pool.map(one_block, range(len(sizes))))

    return [
        {h: _concat([block[b][h] for block in results]) for h in horizons}
        for b in range(len(books))
    ]


class MonteCarloEngine:
    def __init__(
        self,
        bundle: ModelBundle,
        specs: dict[str, AssetSpec],
        metrics: Metrics | None = None,
        workers: int = DEFAULT_WORKERS,
    ) -> None:
        self.bundle = bundle
        self.specs = specs
        self.metrics = metrics or METRICS
        self.workers = workers

    def run(
        self,
        book: Book,
        spot: dict[str, float],
        horizon_hours: int,
        n_paths: int = DEFAULT_PATHS,
        seed: int | None = None,
        independent: bool = False,
        include_funding: bool = True,
        use_bridge: bool = True,
        target_half_width: float = DEFAULT_TARGET_HALF_WIDTH,
        max_paths: int = MAX_PATHS,
        chunk_paths: int = DEFAULT_CHUNK,
        factor_coin: str | None = "BTC",
        now: datetime | None = None,
    ) -> RiskResult:
        """Single horizon. For several horizons use `run_horizons`, which
        walks them on shared paths."""
        return self.run_horizons(
            book, spot, (horizon_hours,), n_paths=n_paths, seed=seed,
            independent=independent, include_funding=include_funding,
            use_bridge=use_bridge, target_half_width=target_half_width,
            max_paths=max_paths, chunk_paths=chunk_paths, factor_coin=factor_coin,
            now=now,
        )[horizon_hours]

    def run_horizons(
        self,
        book: Book,
        spot: dict[str, float],
        horizons: tuple[int, ...],
        n_paths: int = DEFAULT_PATHS,
        seed: int | None = None,
        independent: bool = False,
        include_funding: bool = True,
        use_bridge: bool = True,
        target_half_width: float = DEFAULT_TARGET_HALF_WIDTH,
        max_paths: int = MAX_PATHS,
        chunk_paths: int = DEFAULT_CHUNK,
        factor_coin: str | None = "BTC",
        now: datetime | None = None,
    ) -> dict[int, RiskResult]:
        """Every horizon from one walk of shared paths (audit A-10).

        Convergence is judged on the horizon with the widest interval, so
        every returned result satisfies §2.5's 2 pp rule, and P(liq) is
        pathwise monotone in the horizon rather than monotone in expectation.
        """
        if not book.positions:
            raise ValueError("an empty book has no liquidation risk to estimate")
        if not horizons or any(h < 1 for h in horizons):
            raise ValueError(f"horizons must be positive step counts, got {horizons}")
        horizons = tuple(sorted(set(int(h) for h in horizons)))
        now = now or datetime.now(timezone.utc)
        # §2.5: a random seed in production, a fixed one in tests, and it is
        # written next to the result either way.
        if seed is None:
            seed = int(np.random.SeedSequence().entropy % (2**63))

        # §4.1 measures the book against BTC, so BTC is simulated even when
        # the user does not hold it: comparing two independently drawn runs
        # would put sampling noise into the ratio. A column with no position
        # costs one extra asset in the path draw and nothing in the walk.
        coins = book.coins
        factor_col = None
        if factor_coin is not None:
            if factor_coin not in coins:
                coins = (*coins, factor_coin)
            factor_col = coins.index(factor_coin)
        spec = self.bundle.path_spec(coins, independent=independent)
        missing_spot = [c for c in coins if c not in spot]
        if missing_spot:
            raise KeyError(f"no spot price for {missing_spot}")
        spot_vec = np.array([spot[c] for c in coins], dtype=np.float64)
        # One bridge column per COIN in the universe, not per pocket in this
        # book: the columns are keyed by universe index so the same coin gets
        # the same draws in every book of a paired walk (see
        # `simulate_books_checkpointed`). Unused columns cost one uniform per
        # path-step and nothing else.
        n_iso = len(coins)
        if include_funding:
            # Only coins the book actually holds; the factor column carries no
            # position, so it accrues no funding.
            for c in book.coins:
                if c not in self.bundle.funding:
                    raise KeyError(
                        f"no funding model for {c}; a book position without funding history "
                        "would silently understate cost (§1.5)"
                    )

        notes: list[str] = []
        # Audit F-7: the copula df is FITTED since A9 and refits on every
        # five-minute bundle rebuild, so `seed + model_version` no longer
        # reproduces a number on their own — the same seed under a df of 6.5
        # and 4.0 gives different tails. §2.5 says provenance is "everything
        # needed to reproduce", so the parameter rides with every result.
        # Read off the SPEC, not the bundle: baseline B simulates with
        # copula_df=None through this same engine, and recording the bundle's
        # value against the baseline's rows would attribute the wrong model.
        notes.append(f"copula_df={spec.copula_df}")
        target = n_paths
        raws: dict[int, _RawOutcome] = {}
        converged = False
        while True:
            raws = self._accumulate(
                book, spec, spot_vec, target, horizons, seed, n_iso,
                include_funding, use_bridge, chunk_paths, factor_col,
            )
            widest = max(
                wilson_half_width(int(_any_liq(r).sum()), target) for r in raws.values()
            )
            if widest <= target_half_width:
                converged = True
                break
            # Size the next attempt on the horizon that is hardest to resolve.
            worst_p = max(float(_any_liq(r).mean()) for r in raws.values())
            needed = paths_needed_for_half_width(worst_p, target_half_width)
            nxt = min(max(needed, target * 2), max_paths)
            if nxt <= target:
                notes.append(
                    f"path cap {max_paths} reached with interval half-width "
                    f"{widest:.4f} > {target_half_width}"
                )
                self.metrics.incr("mc_non_convergence")
                break
            self.metrics.incr("mc_path_escalations")
            target = nxt

        return {
            h: self._assemble(
                raws[h], seed, target, h, converged, now, tuple(notes), factor_coin
            )
            for h in horizons
        }

    def _accumulate(
        self, book, spec, spot_vec, n_paths, horizons, seed, n_iso,
        include_funding, use_bridge, chunk_paths, factor_col=None,
    ) -> dict[int, _RawOutcome]:
        blocks = run_blocks(
            books=(book,),
            specs=self.specs,
            bundle=self.bundle,
            spec=spec,
            spot_vec=spot_vec,
            n_paths=n_paths,
            horizons=horizons,
            seed=seed,
            n_iso=n_iso,
            include_funding=include_funding,
            use_bridge=use_bridge,
            factor_col=factor_col,
            workers=self.workers,
        )
        return blocks[0]

    def _assemble(
        self, raw, seed, n_paths, horizon_hours, converged, now, notes, factor_coin=None
    ) -> RiskResult:
        version = self.bundle.model_version
        rng = np.random.default_rng(seed ^ 0x5EED)

        def prob(flags: np.ndarray) -> RiskEstimate:
            k, n = int(flags.sum()), flags.size
            lo, hi = wilson_interval(k, n)
            point = k / n
            # The Wilson centre is shifted from k/n; keep the point estimate
            # inside its own interval without widening it dishonestly.
            return RiskEstimate(point, min(lo, point), max(hi, point), version, now)

        eq = raw.equity_change
        cvar = conditional_value_at_risk(eq, 0.95)
        c_lo, c_hi = bootstrap_ci(eq, lambda s: conditional_value_at_risk(s, 0.95), rng)
        # The bootstrap is on the PnL sample, so the interval is in PnL
        # orientation; CVaR is reported as a positive loss, hence the flip.
        cvar_est = RiskEstimate(cvar, min(c_lo, c_hi, cvar), max(c_lo, c_hi, cvar), version, now)

        iso = {}
        for i, coin in enumerate(raw.isolated_coins):
            iso[coin] = prob(raw.iso_liq[:, i])

        return RiskResult(
            p_liq_any=prob(_any_liq(raw)),
            p_liq_cross=prob(raw.cross_liq),
            p_liq_isolated=iso,
            cvar_95_usd=cvar_est,
            var_95_usd=value_at_risk(eq, 0.95),
            equity_change=PredictiveDistribution.from_samples(eq),
            funding_cost=PredictiveDistribution.from_samples(raw.funding_paid),
            start_equity=raw.start_equity,
            n_paths=n_paths,
            converged=converged,
            provenance=SimulationProvenance(
                seed=seed,
                n_paths=n_paths,
                horizon_hours=horizon_hours,
                model_version=version,
                computed_at=now,
                notes=notes,
            ),
            raw_equity_change=eq,
            factor_return=raw.factor_return,
            factor_coin=factor_coin,
        )


def _any_liq(raw: _RawOutcome) -> np.ndarray:
    if raw.iso_liq.size:
        return raw.cross_liq | raw.iso_liq.any(axis=1)
    return raw.cross_liq


def _concat(parts: list[_RawOutcome]) -> _RawOutcome:
    if len(parts) == 1:
        return parts[0]
    return _RawOutcome(
        cross_liq=np.concatenate([p.cross_liq for p in parts]),
        iso_liq=np.concatenate([p.iso_liq for p in parts]),
        equity_change=np.concatenate([p.equity_change for p in parts]),
        funding_paid=np.concatenate([p.funding_paid for p in parts]),
        isolated_coins=parts[0].isolated_coins,
        start_equity=parts[0].start_equity,
        factor_return=np.concatenate([p.factor_return for p in parts]),
    )
