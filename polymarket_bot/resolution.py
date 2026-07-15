"""Strategy: resolution alpha (near-riskless carry on effectively-decided markets).

When a market's top outcome trades at 0.95-0.985 with fresh volume and is at or
just past its end date, the outcome is effectively decided but not yet paid out
(UMA resolution + on-chain settlement take hours). Buying that side earns
(1 - price) over that window.

This is taker flow (you cross to grab it), so the edge is computed AFTER the
category taker fee AND an honest `dispute_haircut` reserved for the residual
risk that UMA overturns the "obvious" outcome. Detect + alert by default;
execution is opt-in.

The price/volume/date heuristic is a proxy for "an oracle proposal is live".
The clean upgrade is to read the UMA Optimistic Oracle directly — a drop-in
data source behind the same `evaluate` interface.
"""

from __future__ import annotations

import logging

from pydantic import BaseModel

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .fees import FeeModel
from .ledger import Ledger
from .models import Market, simple_estimate
from .monitor import alert
from .portfolio import classify_category

log = logging.getLogger(__name__)


class ResolutionCandidate(BaseModel):
    market: Market
    outcome_index: int
    price: float
    net_edge: float          # (1 - price) - taker_fee - dispute_haircut


class ResolutionAlpha:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str):
        self._cfg = cfg.resolution
        self._fees = FeeModel(cfg.fees)
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode

    def reject_reason(self, m: Market) -> str | None:
        """None = a resolution-alpha candidate; else the reject reason (diagnostics)."""
        c = self._cfg
        if m.closed or not m.outcome_prices or len(m.clob_token_ids) < len(m.outcome_prices):
            return "closed / no book"
        if m.volume_24h_usd < c.min_volume_24h_usd:
            return f"24h volume < ${c.min_volume_24h_usd:,.0f}"
        days = m.days_to_resolution()
        if days is not None and days > c.max_days_to_resolution:
            return "resolution not imminent"
        if m.volume_usd > 0 and m.volume_24h_usd / m.volume_usd < c.volume_spike_ratio:
            return "no fresh volume (stale)"
        price = max(m.outcome_prices)
        if not c.near_min <= price <= c.near_max:
            return f"top outcome outside [{c.near_min}, {c.near_max}]"
        if self._net_edge(m, price) < c.min_net_edge:
            return "net edge below min (fee + dispute haircut)"
        return None

    def _net_edge(self, m: Market, price: float) -> float:
        category = classify_category(m.question, m.category)
        fee = self._fees.taker_fee(category, m.category)
        return (1.0 - price) - fee - self._cfg.dispute_haircut

    def evaluate(self, m: Market) -> ResolutionCandidate | None:
        if self.reject_reason(m) is not None:
            return None
        idx = m.outcome_prices.index(max(m.outcome_prices))
        price = m.outcome_prices[idx]
        return ResolutionCandidate(market=m, outcome_index=idx, price=price,
                                   net_edge=self._net_edge(m, price))

    def execute(self, cand: ResolutionCandidate) -> float:
        """Taker-buy the near-resolved side at its ask. Returns dollars spent."""
        m = cand.market
        token = m.clob_token_ids[cand.outcome_index]
        book = self._clob.order_book(token)
        ask = book.best_ask if book is not None else 0.0
        if ask <= 0 or ask > self._cfg.near_max:
            return 0.0
        price = round_to_tick(ask, m.tick_size)
        size = float(int(self._cfg.max_stake_usd // max(price, 1e-9)))
        if size < m.min_order_size:
            return 0.0
        order_id = None
        if self._trader is not None:
            try:
                resp = self._trader.buy_limit(token, price, size,
                                              neg_risk=m.neg_risk, order_type="FOK")
                order_id = (resp or {}).get("orderID")
            except Exception as exc:
                log.error("resolution execute failed %s: %s", token[:16], exc)
                return 0.0
        self._ledger.record_trade(
            mode=self._mode, estimate=simple_estimate(m, cand.outcome_index, price),
            category="resolution", side="BUY", price=price, size=size,
            order_id=order_id, status="filled" if self._trader else f"{self._mode}-filled",
            strategy="resolution")
        return price * size

    def cycle(self, markets: list[Market]) -> list[ResolutionCandidate]:
        if not self._cfg.enabled:
            return []
        found: list[ResolutionCandidate] = []
        for m in markets:
            cand = self.evaluate(m)
            if cand is None:
                continue
            found.append(cand)
            outcome = m.outcomes[cand.outcome_index] if cand.outcome_index < len(m.outcomes) else "?"
            note = "" if self._cfg.execute else " (execute off)"
            alert(f"RESOLUTION alpha [{self._mode}] [{outcome}] @ {cand.price:.3f} "
                  f"-> net edge +{cand.net_edge * 100:.2f}% after fee+dispute reserve "
                  f"— {m.question[:60]}{note}")
            if self._cfg.execute:
                spent = self.execute(cand)
                if spent > 0:
                    log.info("resolution executed: $%.2f", spent)
        return found[: self._cfg.max_alerts_per_cycle]
