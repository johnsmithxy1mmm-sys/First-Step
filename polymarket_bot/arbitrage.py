"""Strategy #1: structural arbitrage within neg-risk events.

In a multi-outcome event (elections, nominations) exactly one outcome
resolves YES. Two mirror windows:

  YES basket: sum ask(Yes_i) < 1  -> buy Yes of all outcomes;
              pays $1 per set, profit = 1 - sum(ask).
  NO basket:  sum bid(Yes_i) > 1  <=> sum ask(No_i) < n-1 -> buy No of all
              outcomes; pays $(n-1) per set, profit = (n-1) - sum(ask No).

Two honest corrections to "mathematically guaranteed profit":

  Fees. To take the window, legs are bought aggressively (taker). Edge is
  computed AFTER the per-category taker fee — otherwise a +1.7% "pre-fee" on
  a sports basket (fee ~3%) is actually a loss. The min_profit_pct threshold
  applies to the NET edge.

  Basket completeness. "Risk-free" only holds if ALL mutually exclusive
  outcomes are bought. If the event has an outcome the scanner does not see
  ("the other team"/field), the sum looks understated but is actually a
  directional position, not an arbitrage. An abnormally large gross edge
  (> suspicious_gross_edge) is flagged suspect and NOT executed automatically.

Windows live seconds to minutes and are eaten by fast bots; REST polling is
the "slow hunter".
"""

from __future__ import annotations

import logging

from pydantic import BaseModel, Field

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .fees import FeeModel
from .ledger import Ledger
from .models import Market, OrderBook, simple_estimate
from .portfolio import classify_category

log = logging.getLogger(__name__)


class ArbLeg(BaseModel):
    market: Market
    outcome_index: int          # 0 = Yes, 1 = No
    token_id: str
    ask: float                  # best buy price for this leg
    depth: float                # shares available at ask


class BasketArb(BaseModel):
    event_id: str
    event_title: str
    side: str                   # "YES" | "NO"
    legs: list[ArbLeg] = Field(default_factory=list)
    taker_coef: float = 0.0     # category coefficient theta, NOT a flat fraction
    suspect: bool = False       # abnormal edge — likely an incomplete basket

    @property
    def cost_per_set(self) -> float:
        return sum(leg.ask for leg in self.legs)

    @property
    def payout_per_set(self) -> float:
        # YES basket pays $1; NO basket pays $(n-1).
        return 1.0 if self.side == "YES" else float(len(self.legs) - 1)

    # --- gross (before fees) ---

    @property
    def profit_per_set(self) -> float:
        return self.payout_per_set - self.cost_per_set

    @property
    def profit_pct(self) -> float:
        return self.profit_per_set / self.cost_per_set if self.cost_per_set > 0 else 0.0

    # --- net (after entry taker fees) ---

    @property
    def fee_per_set(self) -> float:
        """Official per-leg fee: sum of theta * p_i * (1 - p_i) over the legs.
        NOT theta * cost -- see fees.FeeModel for why that overstates it."""
        return sum(FeeModel.fee_per_share_from_coef(self.taker_coef, leg.ask)
                   for leg in self.legs)

    @property
    def net_profit_per_set(self) -> float:
        return self.profit_per_set - self.fee_per_set

    @property
    def net_profit_pct(self) -> float:
        return self.net_profit_per_set / self.cost_per_set if self.cost_per_set > 0 else 0.0

    def max_sets_by_depth(self) -> int:
        return int(min((leg.depth for leg in self.legs), default=0))


class ArbitrageScanner:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str):
        self._live_order_ids: set[str] = set()
        self._cfg = cfg.arbitrage
        self._fees = FeeModel(cfg.fees)
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode

    # --- selecting events worth hitting the books for ---

    def prefilter_events(self, markets: list[Market]) -> list[list[Market]]:
        """Groups of neg-risk markets where the sum of Gamma Yes prices hints at a window."""
        by_event: dict[str, list[Market]] = {}
        for m in markets:
            if (m.event_neg_risk and m.event_id and not m.closed
                    and m.enable_order_book and m.outcome_prices
                    and len(m.clob_token_ids) >= 2):
                by_event.setdefault(m.event_id, []).append(m)

        suspicious = []
        for group in by_event.values():
            if not 2 <= len(group) <= self._cfg.max_legs:
                continue
            if min(mm.volume_24h_usd for mm in group) < self._cfg.min_leg_volume_24h_usd:
                continue
            total = sum(mm.outcome_prices[0] for mm in group)
            # Threshold looser than real edge: Gamma prices lag; the book gives precision.
            if total < 1.0 - self._cfg.prefilter_tolerance \
                    or total > 1.0 + self._cfg.prefilter_tolerance:
                suspicious.append(group)
        suspicious.sort(key=lambda g: abs(sum(m.outcome_prices[0] for m in g) - 1.0),
                        reverse=True)
        return suspicious[: self._cfg.max_events_per_cycle]

    # --- verification against real order books ---

    def verify(self, group: list[Market]) -> BasketArb | None:
        yes_books: list[tuple[Market, OrderBook]] = []
        no_books: list[tuple[Market, OrderBook]] = []
        for m in group:
            yb = self._clob.order_book(m.clob_token_ids[0])
            nb = self._clob.order_book(m.clob_token_ids[1])
            if yb is None or nb is None or yb.best_ask <= 0 or nb.best_ask <= 0:
                return None  # no arbitrage without the full set of legs
            yes_books.append((m, yb))
            no_books.append((m, nb))

        candidates = []
        yes_arb = self._build("YES", [(m, b, 0) for m, b in yes_books])
        no_arb = self._build("NO", [(m, b, 1) for m, b in no_books])
        for arb in (yes_arb, no_arb):
            # Threshold is on the NET edge (after fees).
            if arb is not None and arb.net_profit_pct >= self._cfg.min_profit_pct \
                    and arb.max_sets_by_depth() >= self._cfg.min_sets:
                candidates.append(arb)
        if not candidates:
            return None
        best = max(candidates, key=lambda a: a.net_profit_pct)
        # Abnormally large gross edge = the basket is probably incomplete.
        best.suspect = best.profit_pct > self._cfg.suspicious_gross_edge
        return best

    def _build(self, side: str, legs_raw) -> BasketArb | None:
        legs = []
        for market, book, idx in legs_raw:
            ask = book.best_ask
            depth = next((l.size for l in sorted(book.asks, key=lambda x: x.price)
                          if l.size > 0), 0.0)
            legs.append(ArbLeg(market=market, outcome_index=idx,
                               token_id=market.clob_token_ids[idx],
                               ask=ask, depth=depth))
        market0 = legs_raw[0][0]
        category = classify_category(market0.question, market0.category)
        taker_coef = self._fees.taker_coef(category, market0.category)
        arb = BasketArb(event_id=market0.event_id, event_title=market0.event_title,
                        side=side, legs=legs, taker_coef=taker_coef)
        return arb if arb.profit_per_set > 0 else None

    def local_order_ids(self) -> set[str]:
        """Legs currently in flight — see KillSwitch.reconcile."""
        return set(self._live_order_ids)

    # --- execution ---

    def _legs_look_painted(self, legs) -> bool:
        """Re-fetch each leg's book and refuse if the ask side looks spoofed."""
        from .spoofguard import screen_ask
        for leg in legs:
            book = self._clob.order_book(leg.token_id)
            if book is None:
                log.warning("arbitrage: leg %s book vanished before execute",
                            leg.token_id[:16])
                return True
            v = screen_ask(book)
            if v.suspicious:
                log.warning("arbitrage: leg %s book looks painted (%s) - skipping",
                            leg.token_id[:16], "; ".join(v.reasons))
                return True
        return False

    def execute(self, arb: BasketArb) -> float:
        """Buys sets. Returns dollars spent (0 = not executed).

        Leg risk: some limit orders may not fill if the book moved — leaving a
        directional position instead of an arbitrage. Suspect (possibly
        incomplete) baskets are not executed at all.
        """
        if arb.suspect:
            log.warning("arbitrage %s flagged suspect - not executing", arb.event_title[:50])
            return 0.0
        if self._cfg.spoof_screen and self._legs_look_painted(arb.legs):
            return 0.0
        sets = min(
            arb.max_sets_by_depth(),
            int(self._cfg.max_stake_usd // max(arb.cost_per_set, 1e-9)),
        )
        min_size = max(int(leg.market.min_order_size) for leg in arb.legs)
        if sets < max(self._cfg.min_sets, min_size):
            return 0.0

        spent = 0.0
        for leg in arb.legs:
            price = round_to_tick(leg.ask, leg.market.tick_size)
            order_id = None
            filled = float(sets)
            if self._trader is not None:
                try:
                    # FOK, like chainarb: a basket leg that RESTS is not a basket
                    # leg. A GTC leg could sit unfilled while the ledger booked
                    # the structure complete, turning a "riskless" set into a
                    # directional bet nobody could see.
                    resp = self._trader.buy_limit(leg.token_id, price, float(sets),
                                                  neg_risk=True, order_type="FOK")
                    order_id = (resp or {}).get("orderID")
                except Exception as exc:
                    log.error("arb leg failed %s: %s - other legs will not overpay",
                              leg.token_id[:16], exc)
                    continue
                # Visible to reconcile while its fate is unknown, so a leg in
                # flight is never mistaken for an unknown exchange order.
                if order_id:
                    self._live_order_ids.add(order_id)
                # Confirm, never assume: an id means accepted, not filled.
                filled = self._trader.matched_size(order_id, sets)
                self._live_order_ids.discard(order_id)
                if filled <= 0:
                    log.warning("arb leg %s did not fill — not recorded",
                                leg.token_id[:16])
                    continue
            self._ledger.record_trade(
                mode=self._mode,
                estimate=simple_estimate(leg.market, leg.outcome_index, price),
                category="arb", side="BUY", price=price, size=filled,
                order_id=order_id,
                status="filled" if self._trader else "sim-filled",
                strategy="arb",
            )
            spent += price * filled
        return spent

    # --- cycle ---

    def check_group(self, group: list[Market],
                    allow_execute: bool = True) -> BasketArb | None:
        """Verify ONE event group against live books; execute when allowed.

        Shared by the polling cycle and the WS fastlane (which re-checks just
        the group whose token ticked, instead of waiting for the next poll).
        """
        arb = self.verify(group)
        if arb is None:
            return None
        warn = ("  ⚠️ SUSPECT: likely an incomplete basket, check by hand"
                if arb.suspect else "")
        log.info("ARBITRAGE %s %s: %d legs, set $%.4f, gross +%.2f%% -> "
                 "NET after fees +%.2f%% (fee $%.4f/set = %.2f%% of cost), "
                 "depth %d sets%s",
                 arb.side, arb.event_title[:50], len(arb.legs), arb.cost_per_set,
                 arb.profit_pct * 100, arb.net_profit_pct * 100,
                 arb.fee_per_set,
                 (arb.fee_per_set / arb.cost_per_set * 100) if arb.cost_per_set else 0.0,
                 arb.max_sets_by_depth(), warn)
        if self._cfg.execute and allow_execute and not arb.suspect:
            spent = self.execute(arb)
            if spent > 0:
                log.info("arbitrage executed: $%.2f", spent)
        return arb

    def cycle(self, markets: list[Market],
              allow_execute: bool = True) -> list[BasketArb]:
        """allow_execute=False (kill-switch / observe-only / breaker): keep
        detecting and alerting — a human can still act — but place no orders."""
        if not self._cfg.enabled:
            return []
        found: list[BasketArb] = []
        for group in self.prefilter_events(markets):
            arb = self.check_group(group, allow_execute)
            if arb is not None:
                found.append(arb)
        return found
