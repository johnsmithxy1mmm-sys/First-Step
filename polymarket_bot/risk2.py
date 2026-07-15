"""Risk 2.0: portfolio stress, parametric VaR, data guard, circuit breaker.

Pure and testable. These sit on top of the hard kill-switch (risk.py), adding
portfolio-level awareness and per-strategy self-defense.
"""

from __future__ import annotations

import math
from collections import defaultdict

from .models import Market, Position
from .portfolio import (correlation, event_exposure_breakdown,
                        event_netted_exposure)


def parametric_var(positions: list[Position], z: float = 1.65) -> float:
    """Coarse 95% VaR: sqrt(w^T C w) over event-netted category exposures.

    Treats each category's worst-case exposure as its loss scale and accounts
    for cross-category correlation. A proxy until a learned covariance (from the
    tick store) replaces the expert matrix.
    """
    exp = event_netted_exposure(positions)
    var = 0.0
    for a, wa in exp.items():
        for b, wb in exp.items():
            var += correlation(a, b) * wa * wb
    return z * math.sqrt(max(var, 0.0))


def portfolio_stress(positions: list[Position]) -> dict:
    """Instant worst-case picture of the open book."""
    rows = event_exposure_breakdown(positions)
    return {
        "gross_usd": sum(r[2] for r in rows),
        "worst_case_usd": sum(r[3] for r in rows),      # sum of per-event worst cases
        "largest_event_usd": max((r[3] for r in rows), default=0.0),
        "var95_usd": parametric_var(positions),
    }


class MarketDataGuard:
    """Detects corrupt market data (bad prices, mass desync) -> pause trading."""

    def problems(self, markets: list[Market]) -> list[str]:
        out: list[str] = []
        bad = 0
        for m in markets:
            for p in m.outcome_prices:
                if p != p or p < -1e-6 or p > 1.0 + 1e-6:   # NaN or out of [0,1]
                    bad += 1
        if bad:
            out.append(f"{bad} out-of-range prices")
        binaries = [m for m in markets
                    if len(m.outcome_prices) == 2 and not m.closed]
        desynced = sum(1 for m in binaries if abs(sum(m.outcome_prices) - 1.0) > 0.15)
        if binaries and desynced > max(5, int(0.3 * len(binaries))):
            out.append(f"{desynced}/{len(binaries)} binary markets not summing to ~1")
        return out

    def severe(self, markets: list[Market]) -> bool:
        return bool(self.problems(markets))


class StrategyCircuitBreaker:
    """Disables a strategy after a losing streak; re-enables once it recovers.

    Fed the cumulative realized PnL per strategy each check; trips a strategy
    when its PnL fell on the last `losing_streak` consecutive checks.
    """

    def __init__(self, losing_streak: int = 3):
        self._streak = losing_streak
        self._history: dict[str, list[float]] = defaultdict(list)
        self.disabled: set[str] = set()

    def update(self, pnl_by_strategy: dict[str, float]) -> None:
        for strategy, pnl in pnl_by_strategy.items():
            h = self._history[strategy]
            h.append(pnl)
            if len(h) > self._streak + 1:
                h.pop(0)
            if len(h) < self._streak + 1:
                continue
            deltas = [h[i + 1] - h[i] for i in range(len(h) - 1)]
            if all(d < 0 for d in deltas[-self._streak:]):
                self.disabled.add(strategy)
            elif all(d >= 0 for d in deltas[-self._streak:]):
                self.disabled.discard(strategy)

    def allows(self, strategy: str) -> bool:
        return strategy not in self.disabled
