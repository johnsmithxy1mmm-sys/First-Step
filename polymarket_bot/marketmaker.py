"""Core (80% of capital): market making + liquidity rewards farming.

Three income streams on the same orders: bid-ask spread, maker rebate, daily
rewards pool. The edge is structural — independent of speed and of predicting
outcomes.

Mechanics (per the master prompt):
1. Fair value = microprice (midpoint weighted by bid/ask sizes).
2. Quotes symmetric around fair within the rewards program's max_spread
   (otherwise they do not count in quadratic scoring); size >= rewards_min_size.
3. Inventory skew — the main risk mechanism: fair shifts against inventory
   proportional to skew_k * inventory / max_position.
4. Requote with hysteresis: reprice only if fair moved
   >= requote_threshold_ticks OR the quote is older than requote_timer_sec.
   Every extra cancel eats rate limit and interrupts rewards sampling.
5. Adverse selection guard: midpoint jump / volume spike -> pull quotes,
   cooldown.
6. At midpoint <0.10 or >0.90 a two-sided quote is required for rewards; if
   inventory allows only one side — leave the market.

Quoting with two BUYS (Yes bid + No bid): both sides need only pUSD; filling
both gives Yes+No = $1 at redemption, profit = spread + rebate.

Modes: dry-run — intentions are logged; paper — virtual fills on the real
flow (a bid "fills" when the market trades through its price); live — real
orders.
"""

from __future__ import annotations

import logging
import math
import threading
import time

from pydantic import BaseModel

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .fees import FeeModel
from .ledger import Ledger
from .microstructure import RealizedVol, fill_probability
from .models import Market, simple_estimate
from .monitor import alert
from .portfolio import classify_category
from .scorer import MarketScorer
from .ws_feed import TopOfBook

log = logging.getLogger(__name__)


class Quote(BaseModel):
    market: Market
    fair: float
    yes_bid: float
    no_bid: float               # bid on the No token; in Yes terms this is ask = 1 - no_bid
    size: float
    ts: float = 0.0
    p_fill_pred: float = -1.0   # fill probability predicted at placement (<0 = none)

    @property
    def implied_yes_ask(self) -> float:
        return 1.0 - self.no_bid

    @property
    def captured_spread(self) -> float:
        return self.implied_yes_ask - self.yes_bid


class TrackedOrder(BaseModel):
    order_id: str
    market: Market
    outcome_index: int
    price: float
    size: float
    matched_recorded: float = 0.0


class MarketMaker:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str,
                 top_source=None, feedback=None):
        """top_source: callable(token_id) -> TopOfBook | None (WS feed);
        without it the top of book is taken from REST.
        feedback: MarkoutFeedback | None — per-market spread multiplier."""
        self._cfg = cfg.market_maker
        self._risk = cfg.risk
        self._fees = FeeModel(cfg.fees)
        self._scorer = MarketScorer(cfg)
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode
        self._top_source = top_source
        self._feedback = feedback
        self._vol = RealizedVol()
        self._size_map: dict[str, float] = {}        # rewards-weighted quote sizes
        self._last_mid: dict[str, float] = {}       # cycle-anchored mid per market
        self._cooldown_until: dict[str, float] = {}  # guard cooldown deadline (epoch s)
        self._quotes: dict[str, Quote] = {}          # active quotes (all modes)
        self._orders: dict[str, list[TrackedOrder]] = {}  # live orders per market
        self._quoted: dict[str, Market] = {}         # token_id -> quoted market (WS reprice)
        self._lock = threading.RLock()               # cycle vs WS-tick react
        # Optional paper-fill sizer: callable(quote, outcome_index, top) -> size.
        # None = classic optimistic full fill; the replay installs a queue-aware one.
        self.fill_model = None
        # Learning hooks (set by the bot; both default to no-ops):
        # fill_calibrator maps predicted -> realized fill probability;
        # size_factor is the Sharpe allocator's capital multiplier (caps still
        # apply after it — it tilts, never breaks a limit).
        self.fill_calibrator = None
        self.size_factor: float = 1.0

    def _spread_mult(self, market_id: str) -> float:
        return self._feedback.multiplier(market_id) if self._feedback is not None else 1.0

    def refresh_feedback(self, horizon_sec: int = 60) -> None:
        """Refit the markout feedback from the ledger (called by a job)."""
        if self._feedback is not None:
            self._feedback.fit(self._ledger.markout_by_market(self._mode, horizon_sec))

    # --- data sources ---

    def _top(self, token: str) -> TopOfBook | None:
        if self._top_source is not None:
            top = self._top_source(token)
            if top is not None:
                return top
        book = self._clob.order_book(token)
        if book is None:
            return None
        bid, ask = book.best_bid, book.best_ask
        bid_size = next((l.size for l in book.bids if l.price == bid), 0.0)
        ask_size = next((l.size for l in book.asks if l.price == ask), 0.0)
        return TopOfBook(bid=bid, bid_size=bid_size, ask=ask, ask_size=ask_size,
                         ts=time.time())

    # --- quotes ---

    def compute_quote(self, market: Market, top: TopOfBook) -> Quote | None:
        c = self._cfg
        tick = market.tick_size
        if top.bid <= 0 or top.ask <= 0:
            return None

        fair = top.microprice
        sigma = self._vol.update(market.id, top.mid)   # A-S realized volatility
        # Inventory skew: long Yes -> fair down (bid lower, ask more aggressive).
        # Scaled by (1 + vol) so we lean harder against inventory in fast markets.
        skew = c.inventory_skew_k * self._inventory_frac(market)
        fair -= skew * max(c.half_spread, tick) * (1.0 + c.vol_spread_k * sigma)

        # Half-spread: inside the rewards band, but not below fee break-even.
        # Widened by realized markout (adverse selection) and realized volatility.
        category = classify_category(market.question, market.category)
        # Rebate is priced at `fair`: near $1 the rebate is almost nothing, so
        # the break-even floor correctly demands the spread carry the whole edge.
        min_half = self._fees.mm_min_half_spread(
            category, fair, market.category, self._risk.min_edge_after_fees)
        base_half = c.half_spread * self._spread_mult(market.id) * (1.0 + c.vol_spread_k * sigma)
        half = max(base_half, min_half, tick)
        if market.in_rewards_program:
            half = min(half, market.rewards_max_spread * 0.9)
            if half < max(min_half, tick):
                return None  # reward band narrower than fee break-even — do not quote

        yes_bid = round_to_tick(fair - half, tick)
        yes_ask = round_to_tick(fair + half, tick)
        # Do not cross the book.
        yes_bid = min(yes_bid, round_to_tick(top.ask - tick, tick))
        yes_ask = max(yes_ask, round_to_tick(top.bid + tick, tick))
        if not tick <= yes_bid < yes_ask <= 1 - tick:
            return None

        quote_usd = self._size_map.get(market.id, c.quote_size_usd) * self.size_factor
        size = float(math.floor(quote_usd / max(yes_bid, tick)))
        size = max(size, market.rewards_min_size)  # otherwise it will not count for rewards
        if size < market.min_order_size:
            return None
        # Per-market position cap.
        if abs(self._inventory_usd(market)) + size * yes_bid \
                > self._risk.max_position_per_market_usd * 2:
            return None
        # Predicted fill probability at placement — checked later against what
        # actually happens to this quote (FillCalibrator's training data).
        queue_ahead = top.bid * top.bid_size if yes_bid <= top.bid else 0.0
        flow = market.volume_24h_usd * (max(c.requote_timer_sec, 1.0) / 86_400.0)
        p_pred = fill_probability(queue_ahead, size * yes_bid, flow)
        return Quote(market=market, fair=fair, yes_bid=yes_bid,
                     no_bid=round_to_tick(1.0 - yes_ask, tick),
                     size=size, ts=time.time(), p_fill_pred=p_pred)

    def needs_requote(self, market: Market, top: TopOfBook) -> bool:
        """Hysteresis: do not churn orders without need."""
        current = self._quotes.get(market.id)
        if current is None:
            return True
        moved = abs(top.microprice - current.fair)
        if moved >= self._cfg.requote_threshold_ticks * market.tick_size:
            return True
        return (time.time() - current.ts) >= self._cfg.requote_timer_sec

    # --- inventory ---

    def _inventory_usd(self, market: Market) -> float:
        yes_token, no_token = market.clob_token_ids[0], market.clob_token_ids[1]
        skew = 0.0
        for p in self._ledger.open_positions(self._mode):
            if p.token_id == yes_token:
                skew += p.cost_usd
            elif p.token_id == no_token:
                skew -= p.cost_usd
        return skew

    def _inventory_frac(self, market: Market) -> float:
        cap = max(self._risk.max_position_per_market_usd, 1e-9)
        return max(-1.0, min(1.0, self._inventory_usd(market) / cap))

    def sides_allowed(self, market: Market) -> tuple[bool, bool]:
        inv = self._inventory_usd(market)
        cap = self._risk.max_position_per_market_usd
        return inv < cap, inv > -cap

    # --- adverse-selection guard ---

    def guard_blocks(self, market: Market, top: TopOfBook, anchor: bool = True) -> bool:
        """Adverse-selection guard.

        The reference mid is CYCLE-anchored (updated only when anchor=True, i.e.
        from the scheduler cycle) so per-tick calls from the WS fastlane compare
        against the window start and see cumulative moves — many small ticks
        adding up to a shock still trip the guard. The cooldown is wall-clock
        (cycles * interval_sec), so tick frequency cannot burn it.
        """
        c = self._cfg
        mid = top.mid
        prev = self._last_mid.get(market.id)
        if anchor:
            self._last_mid[market.id] = mid
        if time.time() < self._cooldown_until.get(market.id, 0.0):
            return True
        if prev is not None and abs(mid - prev) >= c.guard_price_move:
            log.info("MM guard: %s mid %.3f -> %.3f — pulling quotes, cooldown",
                     market.question[:40], prev, mid)
            self._cooldown_until[market.id] = (
                time.time() + c.guard_cooldown_cycles * c.interval_sec)
            return True
        if market.volume_usd > 0 and \
                market.volume_24h_usd / market.volume_usd >= c.guard_volume_ratio:
            return True
        return False

    # --- execution ---

    def _cancel_market(self, market_id: str) -> None:
        quote = self._quotes.pop(market_id, None)
        if quote is not None and quote.p_fill_pred >= 0:
            # The quote died unfilled — a labeled outcome for the calibrator.
            self._ledger.record_quote_outcome(self._mode, market_id,
                                              quote.p_fill_pred, filled=False)
        for order in self._orders.pop(market_id, []):
            if self._trader is not None:
                try:
                    self._trader.cancel(order.order_id)
                except Exception:
                    pass

    def _record_fill(self, market: Market, outcome_index: int, price: float,
                     size: float, order_id: str | None, status: str) -> None:
        self._ledger.record_trade(
            mode=self._mode,
            estimate=simple_estimate(market, outcome_index, price),
            category="mm", side="BUY", price=price, size=size,
            order_id=order_id, status=status, strategy="mm",
        )
        side = (market.outcomes[outcome_index]
                if outcome_index < len(market.outcomes)
                else ("Yes" if outcome_index == 0 else "No"))
        alert(f"MM fill [{self._mode}] {side} {price:.3f} x {size:,.0f} "
              f"= ${price * size:,.2f} — {market.question[:60]}")

    def _record_quote_filled(self, quote: Quote) -> None:
        if quote.p_fill_pred >= 0:
            self._ledger.record_quote_outcome(self._mode, quote.market.id,
                                              quote.p_fill_pred, filled=True)
            quote.p_fill_pred = -1.0    # one outcome per quote

    def _sync_live_fills(self) -> None:
        if self._trader is None:
            return
        for orders in self._orders.values():
            for order in orders:
                try:
                    status = self._trader.order_status(order.order_id)
                except Exception:
                    continue
                new_fill = status.get("size_matched", 0.0) - order.matched_recorded
                if new_fill > 0:
                    self._record_fill(order.market, order.outcome_index, order.price,
                                      new_fill, order.order_id, "filled")
                    order.matched_recorded += new_fill
                    quote = self._quotes.get(order.market.id)
                    if quote is not None:
                        self._record_quote_filled(quote)

    def _paper_fills(self) -> None:
        """Paper mode: a bid fills if the market traded through it.

        fill_model (callable(quote, outcome_index, top) -> size), when set,
        decides the filled size (e.g. the replay's queue-aware simulator);
        otherwise the classic optimistic full-size fill applies.
        """
        for quote in list(self._quotes.values()):
            m = quote.market
            yes_top = self._top(m.clob_token_ids[0])
            no_top = self._top(m.clob_token_ids[1])
            if yes_top is not None and 0 < yes_top.ask <= quote.yes_bid:
                size = (self.fill_model(quote, 0, yes_top)
                        if self.fill_model is not None else quote.size)
                if size > 0:
                    self._record_fill(m, 0, quote.yes_bid, size, None, "paper-filled")
                    self._record_quote_filled(quote)
                    self._quotes.pop(m.id, None)
                    log.info("MM paper fill: Yes %.3f x %.0f (%s)",
                             quote.yes_bid, size, m.question[:40])
            elif no_top is not None and 0 < no_top.ask <= quote.no_bid:
                size = (self.fill_model(quote, 1, no_top)
                        if self.fill_model is not None else quote.size)
                if size > 0:
                    self._record_fill(m, 1, quote.no_bid, size, None, "paper-filled")
                    self._record_quote_filled(quote)
                    self._quotes.pop(m.id, None)
                    log.info("MM paper fill: No %.3f x %.0f (%s)",
                             quote.no_bid, size, m.question[:40])

    def _place(self, quote: Quote, quote_yes: bool, quote_no: bool) -> None:
        m = quote.market
        self._quotes[m.id] = quote
        if self._mode != "live":
            log.info("MM [%s] quote %s: bid %.3f / ask %.3f (fair %.4f) x %.0f",
                     self._mode, m.question[:40], quote.yes_bid,
                     quote.implied_yes_ask, quote.fair, quote.size)
            return
        placed: list[TrackedOrder] = []
        legs = []
        if quote_yes:
            legs.append((0, m.clob_token_ids[0], quote.yes_bid))
        if quote_no:
            legs.append((1, m.clob_token_ids[1], quote.no_bid))
        for idx, token, price in legs:
            try:
                resp = self._trader.buy_limit(token, price, quote.size,
                                              neg_risk=m.neg_risk)
            except Exception as exc:
                log.error("MM order failed %s: %s", token[:16], exc)
                continue
            order_id = (resp or {}).get("orderID")
            if order_id:
                placed.append(TrackedOrder(order_id=order_id, market=m,
                                           outcome_index=idx, price=price,
                                           size=quote.size))
        if placed:
            self._orders[m.id] = placed

    def _rewards_weighted_sizes(self, selected, books) -> dict[str, float]:
        """Allocate the quote budget across markets by rewards score, not evenly."""
        c = self._cfg
        if not c.rewards_weighting or not selected:
            return {}
        scores = {m.id: max(self._scorer.score(m, books.get(m.clob_token_ids[0])), 0.0)
                  for m in selected}
        total = sum(scores.values())
        if total <= 0:
            return {}
        budget = c.quote_size_usd * len(selected)     # same total, redistributed
        lo, hi = c.quote_size_usd * 0.3, c.quote_size_usd * 3.0
        return {mid: min(max(budget * s / total, lo), hi) for mid, s in scores.items()}

    # --- cycle ---

    def cycle(self, markets: list[Market]) -> list[Quote]:
        if not self._cfg.enabled:
            return []
        # Book fetches are slow network READS that touch no shared state, so do
        # them BEFORE taking the lock. Held inside, N sequential fetches at the
        # request timeout could pin the lock for minutes, and react_to_tick on
        # the WS thread would be starved behind it.
        books = {m.clob_token_ids[0]: self._clob.order_book(m.clob_token_ids[0])
                 for m in markets if self._scorer.eligible(m)}
        with self._lock:
            self._sync_live_fills()
            if self._mode == "paper":
                self._paper_fills()

            selected = self._scorer.top_markets(markets, books)
            self._size_map = self._rewards_weighted_sizes(selected, books)
            selected_ids = {m.id for m in selected}
            for market_id in list(self._quotes) + list(self._orders):
                if market_id not in selected_ids:
                    self._cancel_market(market_id)

            active: list[Quote] = []
            self._quoted = {}
            for market in selected:
                top = self._top(market.clob_token_ids[0])
                if top is None:
                    continue
                quote = self._requote_market(market, top)
                if quote is not None:
                    active.append(quote)
                    for tok in market.clob_token_ids[:2]:   # for WS-tick reprice
                        self._quoted[tok] = market
            return active

    def _requote_market(self, market: Market, top: TopOfBook,
                        anchor: bool = True) -> Quote | None:
        """One market's quote decision (shared by the cycle and the WS fastlane)."""
        if self.guard_blocks(market, top, anchor=anchor):
            self._cancel_market(market.id)
            return None
        if not self.needs_requote(market, top):
            return self._quotes.get(market.id)

        quote = self.compute_quote(market, top)
        quote_yes, quote_no = self.sides_allowed(market)
        mid = top.mid
        # Extreme midpoint: a two-sided quote is required for rewards; skip one-sided.
        if (mid < 0.10 or mid > 0.90) and not (quote_yes and quote_no):
            self._cancel_market(market.id)
            return None
        if quote is None or not (quote_yes or quote_no):
            self._cancel_market(market.id)
            return None
        # Queue preservation: a cancel-replace for a <=1-tick improvement loses
        # our place in line. If the resting order is likely to fill where it
        # is, keep it — the queue position is worth more than the tick.
        current = self._quotes.get(market.id)
        if (current is not None and self._cfg.requote_min_fill_prob > 0
                and abs(quote.yes_bid - current.yes_bid) <= market.tick_size + 1e-12):
            queue_ahead = top.bid * top.bid_size          # size at the touch (proxy)
            recent_flow = market.volume_24h_usd * (self._cfg.interval_sec / 86_400.0)
            p_fill = fill_probability(queue_ahead, current.size * current.yes_bid,
                                      recent_flow)
            if self.fill_calibrator is not None:
                # Learned map: what this predicted probability ACTUALLY
                # converts to on our own quotes.
                p_fill = self.fill_calibrator.calibrate(p_fill)
            if p_fill >= self._cfg.requote_min_fill_prob:
                return current
        self._cancel_market(market.id)
        self._place(quote, quote_yes, quote_no)
        return quote

    def react_to_tick(self, token: str, top: TopOfBook) -> bool:
        """WS fastlane: instantly reprice the quoted market this token belongs to.

        NEVER blocks: this runs on the websocket recv thread (main.on_tick), and
        a stalled recv loop stops advancing the last-message timestamp, which
        trips risk.ws_staleness_kill_sec and pulls every quote. cycle() can hold
        this lock across order placement, so take it non-blocking and skip the
        reprice when busy — the cycle we are contending with is itself requoting,
        and the next tick will catch anything it missed. A skipped reprice is a
        non-event; a stalled stream is a self-inflicted kill-switch.
        """
        if not self._cfg.enabled or top.bid <= 0 or top.ask <= 0:
            return False
        if not self._lock.acquire(blocking=False):
            log.debug("mm: skipped WS reprice of %s (cycle holds the lock)",
                      token[:16])
            return False
        try:
            market = self._quoted.get(token)
            if market is None:
                return False
            # WS thread: use the tick or the WS store only — NEVER fall back to
            # REST here (a blocking HTTP call would stall the recv loop and can
            # trip the staleness kill-switch).
            if token == market.clob_token_ids[0]:
                yes_top = top
            elif self._top_source is not None:
                yes_top = self._top_source(market.clob_token_ids[0])
            else:
                return False
            if yes_top is None:
                return False
            return self._requote_market(market, yes_top, anchor=False) is not None
        finally:
            self._lock.release()

    def local_order_ids(self) -> set[str]:
        with self._lock:
            return {o.order_id for orders in self._orders.values() for o in orders}

    def shutdown(self) -> None:
        with self._lock:
            for market_id in list(self._quotes) + list(self._orders):
                self._cancel_market(market_id)
