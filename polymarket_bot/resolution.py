"""Strategy: resolution alpha (near-riskless carry on effectively-decided markets).

When a market's top outcome trades at 0.95-0.985 with fresh volume and is at or
just past its end date, the outcome is effectively decided but not yet paid out
(UMA resolution + on-chain settlement take hours). Buying that side earns
(1 - price) over that window.

This is taker flow (you cross to grab it), so the edge is computed AFTER the
category taker fee AND an honest `dispute_haircut` reserved for the residual
risk that UMA overturns the "obvious" outcome. Detect + alert by default;
execution is opt-in.

Candidate admission is oracle-first: when Gamma reports a UMA answer on-chain
(umaResolutionStatus "proposed"/"resolved"), that FACT admits the market and
the price/volume/date heuristics are skipped (they were only ever a proxy for
"an oracle proposal is live"). Actively disputed markets are rejected outright.
Markets with no oracle signal still go through the original heuristics.
Reading the Optimistic Oracle contract directly (instead of Gamma's report of
it) remains a drop-in upgrade behind the same `evaluate` interface.
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

# Gamma's umaResolutionStatus values. An answer EXISTS on-chain:
ORACLE_CONFIRMED_STATUSES = frozenset({"proposed", "resolved"})
# The answer is actively contested — trading "near-certain" here is a bet:
DISPUTED_STATUSES = frozenset({"challenged", "disputed"})


class ResolutionCandidate(BaseModel):
    market: Market
    outcome_index: int
    price: float
    net_edge: float          # (1 - price) - taker_fee - dispute_haircut


class ResolutionAlpha:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str):
        self._cfg = cfg.resolution
        self._alerts = cfg.alerts
        self._fees = FeeModel(cfg.fees)
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode

    def reject_reason(self, m: Market) -> str | None:
        """None = a resolution-alpha candidate; else the reject reason (diagnostics).

        Two admission paths:
          * ORACLE — Gamma reports a UMA answer on-chain ("proposed"/
            "resolved"): a fact, so the imminence/freshness HEURISTICS are
            skipped. The price band still applies — the status does not say
            WHICH outcome was proposed; the market price does, and the band
            also confirms the market agrees with the oracle.
          * HEURISTIC — no oracle signal: the original volume/imminence/
            freshness proxies.
        An actively disputed market is rejected outright on either path.
        """
        c = self._cfg
        if m.closed or not m.outcome_prices or len(m.clob_token_ids) < len(m.outcome_prices):
            return "closed / no book"
        if m.uma_resolution_status in DISPUTED_STATUSES:
            return "UMA dispute active"
        if m.uma_resolution_status not in ORACLE_CONFIRMED_STATUSES:
            if m.volume_24h_usd < c.min_volume_24h_usd:
                return f"24h volume < ${c.min_volume_24h_usd:,.0f}"
            days = m.days_to_resolution()
            # No end date = we cannot verify imminence -> not a candidate.
            if days is None or days > c.max_days_to_resolution:
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
        # Dollars per share AT THIS PRICE. Critical here: this strategy buys at
        # 0.95-0.985, where a flat-fraction fee model overstates the cost 20-67x
        # and made every non-zero-fee category look permanently unprofitable.
        fee = self._fees.taker_fee_per_share(category, price, m.category)
        # "resolved" = the dispute window is over, the outcome is final — only
        # settlement remains, so no dispute reserve. "proposed" can still be
        # challenged: reserve the FULL haircut.
        haircut = 0.0 if m.uma_resolution_status == "resolved" \
            else self._cfg.dispute_haircut
        return (1.0 - price) - fee - haircut

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
        filled = size
        if self._trader is not None:
            try:
                resp = self._trader.buy_limit(token, price, size,
                                              neg_risk=m.neg_risk, order_type="FOK")
                order_id = (resp or {}).get("orderID")
            except Exception as exc:
                log.error("resolution execute failed %s: %s", token[:16], exc)
                return 0.0
            # A killed FOK still returns an order id, so confirm what matched.
            # Recording the request would put shares in the ledger that the
            # account does not hold.
            filled = self._trader.matched_size(order_id, size)
            if filled <= 0:
                log.info("resolution: FOK did not fill for %s — nothing recorded",
                         token[:16])
                return 0.0
        self._ledger.record_trade(
            mode=self._mode, estimate=simple_estimate(m, cand.outcome_index, price),
            category="resolution", side="BUY", price=price, size=filled,
            order_id=order_id, status="filled" if self._trader else f"{self._mode}-filled",
            strategy="resolution")
        return price * filled

    def cycle(self, markets: list[Market],
              allow_execute: bool = True) -> list[ResolutionCandidate]:
        """allow_execute=False (kill-switch / observe-only / breaker): keep
        detecting and alerting — a human can still act — but place no orders."""
        if not self._cfg.enabled:
            return []
        found: list[ResolutionCandidate] = []
        for m in markets:
            if len(found) >= self._cfg.max_alerts_per_cycle:
                break     # the cap bounds alerts AND executions, not just the return
            cand = self.evaluate(m)
            if cand is None:
                continue
            found.append(cand)
            outcome = m.outcomes[cand.outcome_index] if cand.outcome_index < len(m.outcomes) else "?"
            note = "" if self._cfg.execute and allow_execute else " (execute off)"
            alert(f"RESOLUTION alpha [{self._mode}] [{outcome}] @ {cand.price:.3f} "
                  f"-> net edge +{cand.net_edge * 100:.2f}% after fee+dispute reserve "
                  f"— {m.question[:60]}{note}",
                  key=f"resolution:{m.id}",
                  cooldown_sec=self._alerts.opportunity_cooldown_sec,
                  value=cand.net_edge,
                  min_change=self._alerts.opportunity_min_edge_change)
            if self._cfg.execute and allow_execute:
                spent = self.execute(cand)
                if spent > 0:
                    log.info("resolution executed: $%.2f", spent)
        return found
