"""§4.4 `funding_drag` — what holding this book costs in funding.

A distribution, not a point. §4.4 is explicit about that, and the reason is
the AR(1) with its protocol clamp (§1.5): the rate is persistent, so a hold is
dominated by wherever the rate happens to start, and a short one gives it no
time to average out. Narrowing the default to a day therefore does not make a
point estimate defensible -- on the fitted fixtures the 5th-95th band at 24h
is about as wide as the median itself, wider in relative terms than the same
band over a week, because mean reversion has had less of the horizon to work
in. A median alone would let a user plan around a figure the model gives maybe
even odds of beating.

The default horizon is a day, not a week, and that is a risk decision rather
than a convenience; the reasoning is on `horizon_hours` below and in
OPEN-QUESTIONS A8.

Two things this deliberately does not do:

It does not net funding against price PnL. Funding is a contractual cash
flow and the only non-zero drift the model permits (§2.4); mixing it with
the price distribution would hide it inside a much larger number, which is
exactly how a cost that compounds hourly goes unnoticed over a hold.

It does not condition funding on the price path. Rates and returns are
drawn independently, which is a known simplification with an unresolved
sign (OPEN-QUESTIONS A8) -- in reality funding tracks the perp-spot premium,
so a falling market pushes it negative. The error that leaves grows with the
horizon, which is why the default is a day: over 24h it is small beside the
estimation error on the rate itself, over a week it is not. The independence
is stated on every result rather than buried, because a user planning a long
hold is exactly who it misleads.
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
#: Nothing defaults to a week any more (OPEN-QUESTIONS A8), but this stays
#: exported: it is the named long horizon a caller passes to reach past the
#: default, so it documents that the week is still supported rather than gone,
#: and removing a public module constant would break an importer with an
#: ImportError at a call site that has nothing to do with the decision.
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
    #: The horizon clause is not padding: the default is now 24h but a week is
    #: still reachable by explicit argument, so one static string is attached
    #: to results at horizons whose bias magnitudes differ. A caveat that named
    #: only the side would read as horizon-independent and let a caller who
    #: asked for 168h believe the wording was written for their case.
    caveats: tuple[str, ...] = (
        "Funding is simulated independently of price. In reality it tracks the "
        "perp-spot premium, so a falling market pushes it negative; the sign of "
        "the resulting bias differs by side and by horizon and is not "
        "conservative either way (OPEN-QUESTIONS A8). Over the 24h default it "
        "is small beside the estimation error on the rate; it grows with the "
        "horizon, so a longer hold is indicative rather than calibrated.",
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
    # A day, not a week (OPEN-QUESTIONS A8, decided 2026-07-30). Funding and
    # price are drawn independently, and the sign of that bias is unresolved:
    # under independence a long keeps paying funding through a crash in which
    # the real rate would have turned negative, and a short keeps receiving it
    # through a rally, so the error is not conservative for either side and
    # §10 cannot be satisfied by arguing it errs the safe way. What bounds it
    # instead is the horizon: over 24h the unmodelled correlation has one day
    # to compound and the error sits below the estimation error on the rate;
    # over 168h it does not. The real fix is a joint model of rate and return,
    # which changes the predicted distribution and is therefore a §2 change --
    # a MINOR/MAJOR bump, which resets the §3.3 shadow counter (see
    # risk_engine/version.py) and discards every validation day accumulated so
    # far. Narrowing the default is the honest interim position rather than a
    # workaround: it neither hides the bias nor pays that price. A week stays
    # reachable by passing horizon_hours=WEEK_HOURS; the change is that a
    # caller has to ask, and the docstring says what asking accepts.
    horizon_hours: int = DAY_HOURS,
    n_paths: int = DEFAULT_PATHS,
    seed: int | None = None,
    workers: int = DEFAULT_WORKERS,
    now: datetime | None = None,
) -> FundingDrag:
    """Distribution of funding cost over the holding horizon (§4.4).

    `horizon_hours` defaults to 24h. Longer horizons remain reachable -- pass
    `horizon_hours=WEEK_HOURS` for a week -- but beyond roughly a day the
    independence of funding and price (OPEN-QUESTIONS A8) degrades and the
    result should be read as indicative rather than as a calibrated
    distribution. It is not merely wider than it should be: the bias has a
    sign that differs between longs and shorts, so a week-long figure is not
    safe in either direction and must not be presented to a user as one.
    """
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
