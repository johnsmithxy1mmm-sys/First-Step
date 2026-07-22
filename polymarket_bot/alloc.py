"""Sharpe allocator that ACTS: capital tilts toward what is actually earning.

The digest already computes per-strategy Sharpe weights (research.
sharpe_allocation) — but only as advice. This turns the advice into bounded
sizing multipliers:

    target = weight / equal_weight        (1.0 = "as if no information")
    multiplier <- (1-s) * old + s * clamp(target, floor, ceil)

Honesty constraints, by construction:
  * the corridor [floor, ceil] bounds how far the tilt can go — a hot streak
    cannot 10x a strategy, a cold one cannot silently switch it off (the
    circuit breaker owns on/off, with its own recovery semantics);
  * EMA smoothing stops one digest window from whipsawing capital;
  * every hard risk cap (per-market, per-category, total exposure) applies
    AFTER the multiplier — the allocator tilts, it never breaks a limit.

Applied to: MM / sprint quote budgets and fade / longshot Kelly sizing.
"""

from __future__ import annotations

import logging

from .config import BotConfig

log = logging.getLogger(__name__)


class StrategyAllocator:
    def __init__(self, cfg: BotConfig):
        self._cfg = cfg.allocator
        self.multipliers: dict[str, float] = {}

    def factor(self, strategy: str) -> float:
        if not self._cfg.enabled:
            return 1.0
        return self.multipliers.get(strategy, 1.0)

    def update(self, weights: dict[str, float]) -> dict[str, float]:
        """Feed fresh Sharpe weights; returns {strategy: new multiplier} for
        the strategies whose multiplier moved by more than 1pp."""
        if not self._cfg.enabled or not weights:
            return {}
        c = self._cfg
        equal = 1.0 / len(weights)
        changed: dict[str, float] = {}
        for strategy, w in weights.items():
            target = max(c.floor, min(c.ceil, w / equal))
            old = self.multipliers.get(strategy, 1.0)
            new = (1.0 - c.smoothing) * old + c.smoothing * target
            if abs(new - old) > 0.01:
                changed[strategy] = round(new, 3)
            self.multipliers[strategy] = new
        if changed:
            log.info("allocator: multipliers moved: %s",
                     {k: f"{v:.2f}" for k, v in changed.items()})
        return changed

    def summary(self) -> str:
        if not self.multipliers:
            return ""
        return ", ".join(f"{k} x{v:.2f}"
                         for k, v in sorted(self.multipliers.items()))
