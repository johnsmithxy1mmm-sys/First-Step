"""Resolver: writes the realised outcome next to each prediction (§3.3).

Runs behind the cron. For every prediction whose horizon has elapsed it
records the realised equity, the PIT value, the CRPS and whether the
predicted VaR@95 was breached.

It also records two things §3.3 does not mention but without which the whole
calibration exercise is unsound (OPEN-QUESTIONS B2): the net external cash
flow over the horizon, and whether the set of positions changed. Equity moves
when a user deposits, withdraws, opens or closes -- none of which the model
claimed to predict. Scoring those as model error measures how actively the
sampled traders traded. Both are recorded rather than filtered here, so the
cohort choice is made at analysis time and is visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol

from risk_engine.domain.types import Book
from risk_engine.shadow.cron import position_fingerprint
from risk_engine.shadow.journal import CalibrationJournal


class OutcomeProvider(Protocol):
    def book(self, address: str) -> Book: ...
    def spot(self) -> dict[str, float]: ...

    def external_flow(self, address: str, since: datetime, until: datetime) -> float:
        """Net deposits minus withdrawals over the window, in USD.

        Sourced from `userNonFundingLedgerUpdates`. An implementation that
        cannot supply this must say so rather than return 0.0, because a
        silent zero turns a $50k deposit into a spectacular model failure in
        the calibration score.
        """
        ...


@dataclass(frozen=True, slots=True)
class ResolveReport:
    resolved: int
    failed: list[tuple[int, str]]

    def __str__(self) -> str:
        return f"resolved {self.resolved} predictions, {len(self.failed)} failed"


def resolve_due(
    journal: CalibrationJournal,
    provider: OutcomeProvider,
    now: datetime | None = None,
) -> ResolveReport:
    now = now or datetime.now(timezone.utc)
    spot = provider.spot()
    resolved = 0
    failed: list[tuple[int, str]] = []

    # One address may have several pending rows (model plus both baselines);
    # they share a realised outcome, so it is fetched once per address.
    cache: dict[str, tuple[float, str]] = {}

    for pending in journal.due(now):
        try:
            if pending.address not in cache:
                book = provider.book(pending.address)
                cache[pending.address] = (book.equity(spot), position_fingerprint(book))
            actual_equity, fingerprint = cache[pending.address]

            flow = provider.external_flow(
                pending.address, pending.predicted_at, pending.resolves_at
            )
            # The model predicts the change due to market moves and funding.
            # Removing the external flow is what makes the comparison fair;
            # the raw flow is stored too, so the filtered and unfiltered
            # cohorts can both be scored.
            change = actual_equity - pending.start_equity - flow

            journal.record_outcome(
                prediction_id=pending.id,
                resolved_at=now,
                actual_equity=actual_equity,
                actual_equity_change=change,
                external_flow_usd=flow,
                book_changed=fingerprint != pending.book_snapshot.get("fingerprint"),
                liquidated=actual_equity <= 0.0,
                pit=pending.distribution.pit(change),
                crps=pending.distribution.crps(change),
                var_95_breached=change < -pending.var_95,
                observation_day=pending.predicted_at.astimezone(timezone.utc).date(),
            )
            resolved += 1
        except Exception as exc:  # one bad row must not stop the batch
            failed.append((pending.id, f"{type(exc).__name__}: {exc}"))

    return ResolveReport(resolved=resolved, failed=failed)
