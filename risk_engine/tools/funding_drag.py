"""§4.4 `funding_drag` — what holding this book costs in funding.

A distribution, not a point. §4.4 is explicit about that, and the reason is
that the AR(1) with its protocol clamp (§1.5) produces a genuinely wide
spread over a week: quoting a median alone would let a user plan around a
number the model gives maybe even odds of beating.

Two things this deliberately does not do:

It does not net funding against price PnL. Funding is a contractual cash
flow and the only non-zero drift the model permits (§2.4); mixing it with
the price distribution would hide it inside a much larger number, which is
exactly how a cost that compounds hourly goes unnoticed for a week.

It does not condition funding on the price path. Rates and returns are
drawn independently, which is a known simplification with an unresolved
sign (OPEN-QUESTIONS A8) -- in reality funding tracks the perp-spot premium,
so a falling market pushes it negative. The independence is stated on every
result rather than buried, because a user planning a week-long hold is
exactly who it misleads.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from risk_engine.domain.types import AssetSpec, Book, RiskEstimate, SimulationProvenance
from risk_engine.sim.engine import (
    DEFAULT_PATHS,
    DEFAULT_WORKERS,
    ModelBundle,
    run_blocks,
)
from risk_engine.sim.stats import PredictiveDistribution, bootstrap_ci

DAY_HOURS = 24
WEEK_HOURS = 24 * 7


@dataclass(frozen=True, slots=True)
class FundingDrag:
    """Cost of carry over `horizon_hours`, positive meaning the user pays."""

    address: str
    horizon_hours: int
    #: Whole-book cost. The distribution is the answer; `expected` is a
    #: convenience for a headline and carries its interval like everything
    #: else (§4).
    total: PredictiveDistribution
    expected: RiskEstimate
    #: Per position, so a user can see which leg is bleeding.
    per_position: dict[str, PredictiveDistribution]
    #: Funding as a share of starting equity, which is the number that
    #: actually decides whether a hold is viable.
    share_of_equity: PredictiveDistribution
    start_equity: float
    provenance: SimulationProvenance
    #: OPEN-QUESTIONS A8, restated on every result rather than in a footnote.
    caveats: tuple[str, ...] = (
        "Funding is simulated independently of price. In reality it tracks the "
        "perp-spot premium, so a falling market pushes it negative; the sign of "
        "the resulting bias differs between longs and shorts and is not "
        "conservative either way (OPEN-QUESTIONS A8).",
        "Rates are clamped to the protocol bound, whose value is taken from "
        "documentation and has not been verified against the live API "
        "(OPEN-QUESTIONS C1).",
    )

    def quantile(self, q: float) -> float:
        return float(self.total.quantile(q))

    def summary(self) -> str:
        med = self.quantile(0.5)
        lo, hi = self.quantile(0.05), self.quantile(0.95)
        share = float(self.share_of_equity.quantile(0.5))
        direction = "costs" if med >= 0 else "earns"
        return (
            f"over {self.horizon_hours}h this book {direction} "
            f"${abs(med):,.0f} in funding (5th-95th: ${lo:,.0f} to ${hi:,.0f}), "
            f"{share * 100:+.2f}% of equity at the median"
        )


def funding_drag(
    book: Book,
    spot: dict[str, float],
    bundle: ModelBundle,
    specs: dict[str, AssetSpec],
    horizon_hours: int = WEEK_HOURS,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = None,
    workers: int = DEFAULT_WORKERS,
    now: datetime | None = None,
) -> FundingDrag:
    """Distribution of funding cost over the holding horizon (§4.4)."""
    if not book.positions:
        raise ValueError("an empty book pays no funding")
    now = now or datetime.now(timezone.utc)
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**63))

    coins = book.coins
    missing = [c for c in coins if c not in bundle.funding]
    if missing:
        raise KeyError(
            f"no funding model for {missing}; a held position without funding history "
            "would silently understate cost (§1.5)"
        )
    missing_spot = [c for c in coins if c not in spot]
    if missing_spot:
        raise KeyError(f"no spot price for {missing_spot}")

    spec = bundle.path_spec(coins)
    spot_vec = np.array([spot[c] for c in coins], dtype=np.float64)

    # The whole-book figure comes from one walk of the real book. The
    # per-position split comes from walking each position alone on the SAME
    # paths, so the parts are measured under identical market outcomes and
    # sum to something a user can reconcile against the total.
    singles = [
        Book(book.address, book.cross_collateral if p.mode.value == "cross" else 0.0,
             (p,), book.captured_at)
        for p in book.positions
    ]
    blocks = run_blocks(
        books=(book, *singles),
        specs=specs,
        bundle=bundle,
        spec=spec,
        spot_vec=spot_vec,
        n_paths=n_paths,
        horizons=(horizon_hours,),
        seed=seed,
        n_iso=max(len(b.isolated_positions) for b in (book, *singles)),
        include_funding=True,
        workers=workers,
    )
    whole = blocks[0][horizon_hours]
    paid = whole.funding_paid

    version = bundle.model_version
    mean = float(paid.mean())
    lo, hi = bootstrap_ci(paid, lambda s: float(s.mean()), np.random.default_rng(seed ^ 0xF00D))
    expected = RiskEstimate(mean, min(lo, mean), max(hi, mean), version, now)

    per_position = {
        position.coin: PredictiveDistribution.from_samples(
            blocks[i + 1][horizon_hours].funding_paid
        )
        for i, position in enumerate(book.positions)
    }
    start_equity = whole.start_equity
    return FundingDrag(
        address=book.address,
        horizon_hours=horizon_hours,
        total=PredictiveDistribution.from_samples(paid),
        expected=expected,
        per_position=per_position,
        share_of_equity=PredictiveDistribution.from_samples(paid / max(start_equity, 1e-9)),
        start_equity=start_equity,
        provenance=SimulationProvenance(
            seed=seed, n_paths=n_paths, horizon_hours=horizon_hours,
            model_version=version, computed_at=now,
        ),
    )
