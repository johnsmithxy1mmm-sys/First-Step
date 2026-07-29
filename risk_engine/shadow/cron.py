"""Shadow prediction cron (§3.3).

Runs daily from the end of Phase 1. For each tracked address it writes three
predictions of the 24 h equity-change distribution -- the model and both
baselines -- and the resolver fills in the realised value a day later. While
Phases 2 and 3 are being built read-only, the validation window accumulates
on its own.

Why the distribution and not just P(liq): a liquidation is a rare binary
event, and a realistic sample will never have the statistical power to fill
calibration bins for it. The equity change, by contrast, is known for every
address after 24 h no matter what happened, so every observation carries
information. P(liq) is then checked as a consistent by-product of the same
distribution rather than as a separate rare-event test.

Rate limits (§5.3): the job holds a `WeightBudget` with a large reserved
fraction, so it refuses to spend into the headroom that live users need.
Running out of budget mid-sweep truncates the sweep and is reported -- it is
not an error, and it must not be retried into the reserve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Protocol

import numpy as np

from risk_engine.domain.types import AssetSpec, Book
from risk_engine.market.info import RateLimitExceeded, WeightBudget
from risk_engine.shadow.journal import (
    VARIANT_BASELINE_A,
    VARIANT_BASELINE_B,
    VARIANT_MODEL,
    CalibrationJournal,
)
from risk_engine.sim.engine import ModelBundle, MonteCarloEngine
from risk_engine.sim.stats import wilson_interval
from risk_engine.validation.baselines import NaiveBaseline, run_baseline_b
from risk_engine.version import DISTRIBUTION_VERSION

#: §5.3: keep three quarters of the weight budget for interactive traffic.
SHADOW_RESERVED_FRACTION = 0.75


class SnapshotProvider(Protocol):
    """What the cron needs from the outside world.

    A Protocol rather than a concrete client so the harness runs against
    fixtures in tests and against the Info API in production, with no branch
    inside the job itself.
    """

    def addresses(self) -> list[str]: ...
    def book(self, address: str) -> Book: ...
    def spot(self) -> dict[str, float]: ...
    def specs(self) -> dict[str, AssetSpec]: ...


@dataclass(frozen=True, slots=True)
class SweepReport:
    started_at: datetime
    attempted: int
    written: int
    skipped: list[tuple[str, str]] = field(default_factory=list)
    budget_exhausted: bool = False

    def __str__(self) -> str:
        tail = " (budget exhausted, sweep truncated)" if self.budget_exhausted else ""
        return (
            f"shadow sweep {self.started_at.isoformat()}: {self.written}/{self.attempted} "
            f"addresses written, {len(self.skipped)} skipped{tail}"
        )


class ShadowCron:
    def __init__(
        self,
        provider: SnapshotProvider,
        bundle: ModelBundle,
        journal: CalibrationJournal,
        naive: NaiveBaseline,
        n_paths: int = 20_000,
        horizon_hours: int = 24,
        budget: WeightBudget | None = None,
    ) -> None:
        self.provider = provider
        self.bundle = bundle
        self.journal = journal
        self.naive = naive
        self.n_paths = n_paths
        self.horizon_hours = horizon_hours
        self.budget = budget or WeightBudget(reserved_fraction=SHADOW_RESERVED_FRACTION)

    def run_once(self, now: datetime | None = None, seed: int | None = None) -> SweepReport:
        now = now or datetime.now(timezone.utc)
        specs = self.provider.specs()
        spot = self.provider.spot()
        addresses = self.provider.addresses()
        rng = np.random.default_rng(seed if seed is not None else int(now.timestamp()))

        written = 0
        skipped: list[tuple[str, str]] = []
        exhausted = False

        for address in addresses:
            try:
                self.budget.charge(20)
            except RateLimitExceeded:
                exhausted = True
                break
            try:
                book = self.provider.book(address)
                if not book.positions:
                    skipped.append((address, "no open positions"))
                    continue
                if book.equity(spot) <= 0:
                    skipped.append((address, "non-positive equity"))
                    continue
                self._predict_all(address, book, spot, specs, now, int(rng.integers(2**62)))
                written += 1
            except Exception as exc:  # one bad address must not stop the sweep
                skipped.append((address, f"{type(exc).__name__}: {exc}"))

        return SweepReport(
            started_at=now,
            attempted=len(addresses),
            written=written,
            skipped=skipped,
            budget_exhausted=exhausted,
        )

    def _predict_all(
        self,
        address: str,
        book: Book,
        spot: dict[str, float],
        specs: dict[str, AssetSpec],
        now: datetime,
        seed: int,
    ) -> None:
        snapshot = _book_snapshot(book, spot)

        model = MonteCarloEngine(self.bundle, specs).run(
            book, spot, self.horizon_hours, n_paths=self.n_paths, seed=seed, now=now
        )
        self._write(address, VARIANT_MODEL, now, snapshot, model.start_equity,
                    model.p_liq_any.point, (model.p_liq_any.ci_low, model.p_liq_any.ci_high),
                    model.var_95_usd, model.cvar_95_usd.point, model.equity_change,
                    seed, model.n_paths, model.converged)

        b = run_baseline_b(self.bundle, specs, book, spot, self.horizon_hours,
                           n_paths=self.n_paths, seed=seed, now=now)
        self._write(address, VARIANT_BASELINE_B, now, snapshot, b.start_equity,
                    b.p_liq_any.point, (b.p_liq_any.ci_low, b.p_liq_any.ci_high),
                    b.var_95_usd, b.cvar_95_usd.point, b.equity_change,
                    seed, b.n_paths, b.converged)

        a = self.naive.predict(book, spot, specs, n_draws=self.n_paths, seed=seed, now=now)
        self._write(address, VARIANT_BASELINE_A, now, snapshot, a.start_equity,
                    a.p_liq, a.p_liq_ci, a.equity_change.var(0.95), a.equity_change.cvar(0.95),
                    a.equity_change, seed, a.n_draws, True)

    def _write(self, address, variant, now, snapshot, start_equity, p_liq, ci,
               var_95, cvar_95, distribution, seed, n_paths, converged) -> None:
        self.journal.record_prediction(
            address=address,
            variant=variant,
            predicted_at=now,
            horizon_hours=self.horizon_hours,
            model_version=self.bundle.model_version,
            distribution_version=DISTRIBUTION_VERSION,
            seed=seed,
            n_paths=n_paths,
            converged=converged,
            start_equity=start_equity,
            p_liq=p_liq,
            p_liq_ci=ci,
            var_95=var_95,
            cvar_95=cvar_95,
            distribution=distribution,
            book_snapshot=snapshot,
        )


def _book_snapshot(book: Book, spot: dict[str, float]) -> dict:
    """What the book looked like when the prediction was made.

    Stored verbatim so a later reader can tell whether a bad score came from
    the model or from the account being restructured mid-horizon (§3.3, and
    the cohort split in OPEN-QUESTIONS B2). The position fingerprint is what
    the resolver compares against.
    """
    return {
        "cross_collateral": book.cross_collateral,
        "captured_at": book.captured_at.isoformat(),
        "spot": {c: spot[c] for c in book.coins if c in spot},
        "positions": [
            {
                "coin": p.coin,
                "size": p.size,
                "entry_price": p.entry_price,
                "mode": p.mode.value,
                "leverage": p.leverage,
                "isolated_margin": p.isolated_margin,
            }
            for p in book.positions
        ],
        "fingerprint": position_fingerprint(book),
    }


def position_fingerprint(book: Book) -> str:
    """Stable identity of a book's positions, ignoring price movement."""
    parts = sorted(f"{p.coin}:{p.size:.10g}:{p.mode.value}" for p in book.positions)
    return "|".join(parts)


def wilson(successes: int, n: int) -> tuple[float, float]:
    return wilson_interval(successes, n)
