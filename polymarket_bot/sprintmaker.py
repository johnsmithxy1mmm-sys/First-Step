"""Short-dated market making: fast capital turnover on soon-resolving markets.

The core MM (marketmaker.py) quotes markets 30+ days out — its capital is
locked for weeks. This variant quotes LIQUID markets that resolve within a
few HOURS to a couple of days, so the same dollars are earning spread + maker
rebate and then freed at settlement to redeploy. That is the only honest way
to turn capital fast on a prediction market: capture the spread, do NOT bet on
direction (the taker fee makes short-horizon direction bets negative-EV).

It reuses the full MarketMaker quoting engine (microprice fair, inventory
skew, volatility-scaled spread, adverse-selection guard, queue-preserving
requote, paper/live fills) and only changes two things:

  * selection — SprintScorer keeps markets inside a short [min, max] HOURS
    window with high liquidity, instead of the core "30+ days, in rewards";
  * risk profile — via config (SprintMakerConfig): harder inventory lean,
    wider base spread, tighter shock guard, and a refusal to quote the final
    settlement window where directional information dominates the flow.

Fills are tagged strategy="sprint_mm" so PnL, the circuit breaker and the
digest attribute them separately from the core MM.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from .config import BotConfig
from .marketmaker import MarketMaker
from .models import Market, simple_estimate
from .monitor import alert
from .scorer import MarketScorer

log = logging.getLogger(__name__)


class SprintScorer(MarketScorer):
    """Selection for short-dated MM: a HOURS window instead of a days floor.

    Mirrors MarketScorer's quality gates (liquidity, stability, spread, clear
    and non-subjective resolution) but swaps the horizon test: the market must
    resolve within max_hours_to_resolution and no sooner than
    min_hours_to_resolution (the settlement window is where direction, not
    spread, sets the price — we stay out of it).
    """

    def __init__(self, cfg: BotConfig):
        self._cfg = cfg.sprint_mm

    def reject_reason(self, m: Market, now: datetime | None = None) -> str | None:
        c = self._cfg
        if m.closed or not m.enable_order_book or len(m.clob_token_ids) < 2:
            return "no book / closed"
        if m.volume_24h_usd < c.min_volume_24h_usd:
            return f"24h volume < ${c.min_volume_24h_usd:,.0f}"
        days = m.days_to_resolution(now or datetime.now(timezone.utc))
        if days is None:
            return "no end date"
        hours = days * 24.0
        if hours > c.max_hours_to_resolution:
            return f"resolution > {c.max_hours_to_resolution:.0f}h away"
        if hours < c.min_hours_to_resolution:
            return f"resolution < {c.min_hours_to_resolution:.0f}h away (settlement window)"
        if c.require_rewards_program and not m.in_rewards_program:
            return "not in rewards program"
        if abs(m.one_day_price_change) > c.max_daily_midpoint_move:
            return "volatile (midpoint move > threshold)"
        if m.best_bid > 0 and m.best_ask > 0 \
                and (m.best_ask - m.best_bid) < 2 * m.tick_size:
            return "spread < 2 ticks"
        if not m.resolution_source and len(m.description.strip()) < 80:
            return "unclear resolution rules"
        if self.uma_risk(m):
            return "UMA risk (subjective resolution)"
        return None


class SprintMaker(MarketMaker):
    """Short-dated market maker. Same engine, short-horizon selection + tags."""

    STRATEGY = "sprint_mm"

    def __init__(self, cfg: BotConfig, ledger, clob, trader, mode: str,
                 top_source=None, feedback=None):
        super().__init__(cfg, ledger, clob, trader, mode,
                         top_source=top_source, feedback=feedback)
        # Rebind the core MM's market_maker config/scorer to the sprint profile.
        self._cfg = cfg.sprint_mm
        self._scorer = SprintScorer(cfg)

    def _record_fill(self, market: Market, outcome_index: int, price: float,
                     size: float, order_id: str | None, status: str) -> None:
        """Same as core MM but tagged strategy/category 'sprint_mm'."""
        self._ledger.record_trade(
            mode=self._mode,
            estimate=simple_estimate(market, outcome_index, price),
            category="sprint_mm", side="BUY", price=price, size=size,
            order_id=order_id, status=status, strategy=self.STRATEGY,
        )
        side = (market.outcomes[outcome_index]
                if outcome_index < len(market.outcomes)
                else ("Yes" if outcome_index == 0 else "No"))
        alert(f"SPRINT fill [{self._mode}] {side} {price:.3f} x {size:,.0f} "
              f"= ${price * size:,.2f} — {market.question[:60]}")
