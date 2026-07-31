"""§4.2 `pre_trade_delta` — what a proposed order does to the book's risk.

Two design decisions carry this tool.

**Common random numbers.** The "before" and "after" books differ by one
order and are otherwise identical, so the effect being measured is usually
far smaller than the Monte Carlo error on either side. Two independent runs
would report mostly sampling noise: at 20 000 paths the standard error on a
10% probability is 0.21 pp, while a realistic order might move it by 0.5 pp.
Both books are therefore walked over one identical set of price paths, and
the difference becomes pathwise -- the order's effect and nothing else.

**Overlapping intervals are not the right significance test.** §4.2 says
that when the "before" and "after" intervals overlap, the UI must report
statistical indistinguishability rather than drawing an arrow. Applied to
CRN-paired estimates that rule is wrong, and wrong in the dangerous
direction: two marginal intervals can overlap heavily while the paired
difference is overwhelmingly significant. Non-overlap implies significance,
but overlap does not imply insignificance -- a standard and much-repeated
statistical error. Following it literally would tell a user "no detectable
change" about an order that provably raises their liquidation probability,
which is the §10-forbidden direction.

So this tool reports both:

- `marginal_intervals_overlap` -- exactly what §4.2 asks for;
- `change`, a confidence interval on the *paired difference*, and
  `distinguishable`, read off that interval.

`distinguishable` is the answer the UI should act on. When the two disagree,
`overlap_rule_would_mislead` is set so the discrepancy is visible rather
than silently resolved. See OPEN-QUESTIONS D6.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import (
    AssetSpec,
    Book,
    MarginMode,
    Position,
    RiskEstimate,
    SimulationProvenance,
)
from risk_engine.observability.metrics import METRICS, Metrics, Timer
from risk_engine.sim.engine import (
    DEFAULT_PATHS,
    DEFAULT_TARGET_HALF_WIDTH,
    DEFAULT_WORKERS,
    MAX_PATHS,
    ModelBundle,
    _RawOutcome,
    _any_liq,
    run_blocks,
)
from risk_engine.sim.stats import (
    Z95,
    conditional_value_at_risk,
    conditional_value_at_risk_rows,
    paths_needed_for_half_width,
    wilson_half_width,
    wilson_interval,
)

#: §2.6's online budget for a pre-trade request.
LATENCY_BUDGET_MS = 300.0
#: Bootstrap replicates for the paired CVaR difference. CVaR is a tail
#: functional, so unlike a mean it cannot be differenced per path; the
#: interval has to be resampled. Kept modest to stay inside the budget.
CVAR_BOOTSTRAP_REPS = 200


@dataclass(frozen=True, slots=True)
class ProposedOrder:
    """A hypothetical order, in the terms §4.2 states: asset, direction,
    size, leverage, margin mode."""

    coin: str
    size: float  # signed: positive buy/long, negative sell/short
    leverage: float
    mode: MarginMode = MarginMode.CROSS
    #: Isolated only. Defaults to notional / leverage, i.e. the margin the
    #: venue would move into the pocket at the requested leverage.
    isolated_margin: float | None = None

    def __post_init__(self) -> None:
        if self.size == 0:
            raise ValueError("a zero-size order is not an order")
        if self.leverage <= 0:
            raise ValueError("leverage must be positive")

    def to_position(self, execution_price: float) -> Position:
        iso = None
        if self.mode is MarginMode.ISOLATED:
            iso = self.isolated_margin
            if iso is None:
                iso = abs(self.size) * execution_price / self.leverage
        return Position(
            coin=self.coin,
            size=self.size,
            entry_price=execution_price,
            mode=self.mode,
            leverage=self.leverage,
            isolated_margin=iso,
        )

    def describe(self) -> str:
        side = "buy" if self.size > 0 else "sell"
        return f"{side} {abs(self.size):g} {self.coin} at {self.leverage:g}x {self.mode.value}"


@dataclass(frozen=True, slots=True)
class DeltaEstimate:
    """Before, after, and the paired change between them."""

    before: RiskEstimate
    after: RiskEstimate
    change: RiskEstimate

    @property
    def distinguishable(self) -> bool:
        """Whether the change is significant, from the PAIRED interval.

        This is the question the UI needs answered, and the marginal
        intervals cannot answer it.
        """
        return not (self.change.ci_low <= 0.0 <= self.change.ci_high)

    @property
    def marginal_intervals_overlap(self) -> bool:
        """§4.2's literal rule. Reported, but not the decision."""
        return self.before.overlaps(self.after)

    @property
    def overlap_rule_would_mislead(self) -> bool:
        """The paired test finds a real change that §4.2's rule would hide."""
        return self.distinguishable and self.marginal_intervals_overlap

    @property
    def direction(self) -> int:
        if not self.distinguishable:
            return 0
        return 1 if self.change.point > 0 else -1


@dataclass(frozen=True, slots=True)
class PreTradeDelta:
    order: ProposedOrder
    execution_price: float
    p_liq: DeltaEstimate
    cvar_95_usd: DeltaEstimate
    horizon_hours: int
    n_paths: int
    converged: bool
    provenance: SimulationProvenance
    latency_ms: dict[str, float] = field(default_factory=dict)
    #: Assets in the order that the book did not already hold. §4.2 requires
    #: these to cost a submatrix slice, not a cold start.
    new_assets: tuple[str, ...] = ()

    @property
    def total_latency_ms(self) -> float:
        return self.latency_ms.get("total", 0.0)

    @property
    def within_budget(self) -> bool:
        return self.total_latency_ms <= LATENCY_BUDGET_MS

    @property
    def publishable(self) -> bool:
        """§2.5's flag. NOT a refusal — see `PortfolioRisk.publishable`: the
        withholding happens in the Node backend, not here, and this object
        carries a full set of numbers regardless of what this returns.
        """
        return self.converged

    def summary(self) -> str:
        p = self.p_liq
        if not p.distinguishable:
            verdict = "no statistically distinguishable change in P(liq)"
        else:
            arrow = "raises" if p.direction > 0 else "lowers"
            verdict = (
                f"{arrow} P(liq) by {abs(p.change.point) * 100:.2f}pp "
                f"[{p.change.ci_low * 100:+.2f}, {p.change.ci_high * 100:+.2f}]"
            )
        return f"{self.order.describe()}: {verdict}"


def _paired_proportion_change(
    before: np.ndarray, after: np.ndarray, version: str, now: datetime
) -> RiskEstimate:
    """Interval on `P(after) - P(before)` for paired binary outcomes.

    Each path falls into one of four cells. Only the discordant pairs carry
    information about the difference, which is why the paired interval is so
    much tighter than differencing two marginal ones -- the concordant paths
    cancel exactly instead of contributing noise twice.
    """
    n = before.size
    n01 = int((~before & after).sum())  # the order caused a liquidation
    n10 = int((before & ~after).sum())  # the order prevented one
    diff = (n01 - n10) / n
    # Variance of the paired difference (Agresti, categorical data analysis).
    var = (n01 + n10 - (n01 - n10) ** 2 / n) / (n * n)
    half = Z95 * float(np.sqrt(max(var, 0.0)))
    if n01 == 0 and n10 == 0:
        # Every path agrees: the order changed nothing that these paths can
        # see. The honest interval is the degenerate one at zero, not a
        # fabricated width.
        return RiskEstimate(0.0, 0.0, 0.0, version, now)
    return RiskEstimate(diff, diff - half, diff + half, version, now)


def _paired_cvar_change(
    before: np.ndarray, after: np.ndarray, seed: int, version: str, now: datetime,
    level: float = 0.95, reps: int = CVAR_BOOTSTRAP_REPS,
) -> tuple[RiskEstimate, RiskEstimate, RiskEstimate]:
    """(before, after, change) CVaR estimates, bootstrapped on shared indices.

    Resampling the same path indices for both books preserves the pairing,
    so the interval on the change reflects the order's effect rather than
    the common market variation both books are exposed to.
    """
    cvar_before = conditional_value_at_risk(before, level)
    cvar_after = conditional_value_at_risk(after, level)
    rng = np.random.default_rng(seed ^ 0xC7A2)
    n = before.size
    # Resampled in batches: the vectorised tail selection is what keeps this
    # inside §2.6's budget, but a single (reps, n) index matrix would be
    # 32 MB per book, so it is built a slice at a time.
    batch = max(1, min(reps, 4_000_000 // max(n, 1)))
    # float32 for the resampled copies: the partition is ~3x faster on half
    # the bytes, and a bootstrap *interval* does not need the seventh
    # significant digit. The point estimates below stay in float64.
    b32, a32 = before.astype(np.float32), after.astype(np.float32)
    b_parts, a_parts = [], []
    for start in range(0, reps, batch):
        idx = rng.integers(0, n, size=(min(batch, reps - start), n))
        b_parts.append(conditional_value_at_risk_rows(b32[idx], level))
        a_parts.append(conditional_value_at_risk_rows(a32[idx], level))
    b, a = np.concatenate(b_parts), np.concatenate(a_parts)

    def est(point: float, samples: np.ndarray) -> RiskEstimate:
        lo, hi = np.quantile(samples, [0.025, 0.975])
        return RiskEstimate(point, min(lo, point), max(hi, point), version, now)

    d = a - b
    lo, hi = np.quantile(d, [0.025, 0.975])
    point = cvar_after - cvar_before
    change = RiskEstimate(point, min(lo, point), max(hi, point), version, now)
    return est(cvar_before, b), est(cvar_after, a), change


def pre_trade_delta(
    book: Book,
    order: ProposedOrder,
    spot: dict[str, float],
    bundle: ModelBundle,
    specs: dict[str, AssetSpec],
    horizon_hours: int = 24,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = None,
    include_funding: bool = True,
    target_half_width: float = DEFAULT_TARGET_HALF_WIDTH,
    max_paths: int = MAX_PATHS,
    workers: int = DEFAULT_WORKERS,
    now: datetime | None = None,
    metrics: Metrics | None = None,
) -> PreTradeDelta:
    """Risk of the book with and without the proposed order."""
    metrics = metrics or METRICS
    now = now or datetime.now(timezone.utc)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**63))
    if order.coin not in spot:
        raise KeyError(f"no spot price for {order.coin}")

    with Timer("pre_trade_total", metrics) as total:
        execution_price = spot[order.coin]
        after_book = book.with_position(order.to_position(execution_price))
        new_assets = tuple(c for c in after_book.coins if c not in book.coins)

        # The union universe. §4.2's cold-start requirement is satisfied by
        # construction: an asset the user does not hold is already in the
        # global matrix, so adding it costs a wider submatrix slice and
        # nothing else -- no estimation, no rebuild.
        coins = tuple(dict.fromkeys(book.coins + after_book.coins))
        if not coins:
            raise ValueError("both books are empty; there is no risk to compare")
        spec = bundle.path_spec(coins)
        missing_spot = [c for c in coins if c not in spot]
        if missing_spot:
            raise KeyError(f"no spot price for {missing_spot}")
        spot_vec = np.array([spot[c] for c in coins], dtype=np.float64)

        if include_funding:
            for c in coins:
                if c not in bundle.funding:
                    raise KeyError(
                        f"no funding model for {c}; a position without funding history "
                        "would silently understate cost (§1.5)"
                    )
        n_iso = max(len(book.isolated_positions), len(after_book.isolated_positions))

        target = n_paths
        converged = False
        notes: list[str] = []
        before_raw = after_raw = None
        while True:
            before_raw, after_raw = _accumulate_pair(
                book, after_book, specs, bundle, spec, spot_vec, target,
                horizon_hours, seed, n_iso, include_funding, workers,
            )
            widest = max(
                wilson_half_width(int(_any_liq(r).sum()), target)
                for r in (before_raw, after_raw)
            )
            if widest <= target_half_width:
                converged = True
                break
            worst_p = max(float(_any_liq(r).mean()) for r in (before_raw, after_raw))
            nxt = min(
                max(paths_needed_for_half_width(worst_p, target_half_width), target * 2),
                max_paths,
            )
            if nxt <= target:
                notes.append(
                    f"path cap {max_paths} reached with interval half-width "
                    f"{widest:.4f} > {target_half_width}"
                )
                metrics.incr("mc_non_convergence")
                break
            metrics.incr("mc_path_escalations")
            target = nxt

        version = bundle.model_version
        liq_before, liq_after = _any_liq(before_raw), _any_liq(after_raw)

        def prob(flags: np.ndarray) -> RiskEstimate:
            k, n = int(flags.sum()), flags.size
            lo, hi = wilson_interval(k, n)
            point = k / n
            return RiskEstimate(point, min(lo, point), max(hi, point), version, now)

        p_liq = DeltaEstimate(
            before=prob(liq_before),
            after=prob(liq_after),
            change=_paired_proportion_change(liq_before, liq_after, version, now),
        )
        with Timer("cvar_bootstrap", metrics):
            cb, ca, cc = _paired_cvar_change(
                before_raw.equity_change, after_raw.equity_change, seed, version, now
            )
        cvar = DeltaEstimate(before=cb, after=ca, change=cc)

    latency = {
        "total": total.elapsed_ms,
        **{
            stage: metrics.percentiles(stage).get("p50", 0.0)
            for stage in ("slice_submatrix", "generate_paths", "liquidation_walk")
        },
    }
    if total.elapsed_ms > LATENCY_BUDGET_MS:
        metrics.incr("pre_trade_budget_exceeded")

    if p_liq.overlap_rule_would_mislead:
        metrics.incr("overlap_rule_would_mislead")

    return PreTradeDelta(
        order=order,
        execution_price=execution_price,
        p_liq=p_liq,
        cvar_95_usd=cvar,
        horizon_hours=horizon_hours,
        n_paths=target,
        converged=converged,
        provenance=SimulationProvenance(
            seed=seed, n_paths=target, horizon_hours=horizon_hours,
            model_version=version, computed_at=now, notes=tuple(notes),
        ),
        latency_ms=latency,
        new_assets=new_assets,
    )


def _accumulate_pair(
    book, after_book, specs, bundle, spec, spot_vec, n_paths, horizon_hours,
    seed, n_iso, include_funding, workers,
) -> tuple[_RawOutcome, _RawOutcome]:
    before, after = run_blocks(
        books=(book, after_book),
        specs=specs,
        bundle=bundle,
        spec=spec,
        spot_vec=spot_vec,
        n_paths=n_paths,
        horizons=(horizon_hours,),
        seed=seed,
        n_iso=n_iso,
        include_funding=include_funding,
        workers=workers,
    )
    return before[horizon_hours], after[horizon_hours]
