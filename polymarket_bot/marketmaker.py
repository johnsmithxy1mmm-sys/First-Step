"""Ядро (80% капитала): маркет-мейкинг + liquidity rewards farming.

Три потока дохода на одних ордерах: bid-ask спред, maker rebate, дневной
rewards-пул. Edge структурный — не зависит от скорости и предсказания
исходов.

Механика (по мастер-промпту):
1. Fair value = microprice (midpoint, взвешенный объёмами bid/ask).
2. Котировки симметрично вокруг fair внутри max_spread rewards-программы
   (иначе не засчитываются в quadratic scoring); размер >= rewards_min_size.
3. Inventory skew — главный механизм риска: fair сдвигается против
   инвентаря пропорционально skew_k * inventory / max_position.
4. Requote с гистерезисом: переставляем ордера только если fair ушёл на
   >= requote_threshold_ticks ИЛИ котировка старше requote_timer_sec.
   Каждая лишняя отмена ест rate limit и прерывает rewards-сэмплинг.
5. Adverse selection guard: скачок midpoint / всплеск объёма → снять
   котировки, cooldown.
6. При midpoint <0.10 или >0.90 двусторонняя котировка обязательна для
   rewards; если инвентарь позволяет только одну сторону — покидаем рынок.

Котирование двумя ПОКУПКАМИ (Yes-бид + No-бид): обе стороны требуют только
pUSD; исполнение обеих даёт Yes+No = $1 к выкупу, прибыль = спред + rebate.

Режимы: dry-run — логируются намерения; paper — виртуальные исполнения по
реальному потоку (бид «филлится», когда рынок проторговывается сквозь его
цену); live — реальные ордера.
"""

from __future__ import annotations

import logging
import math
import time

from pydantic import BaseModel

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .fees import FeeModel
from .ledger import Ledger
from .models import Market, simple_estimate
from .portfolio import classify_category
from .scorer import MarketScorer
from .ws_feed import TopOfBook

log = logging.getLogger(__name__)


class Quote(BaseModel):
    market: Market
    fair: float
    yes_bid: float
    no_bid: float               # бид на No-токен; в Yes-терминах это ask = 1 - no_bid
    size: float
    ts: float = 0.0

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
                 top_source=None):
        """top_source: callable(token_id) -> TopOfBook | None (WS-фид);
        без него топ книги берётся из REST."""
        self._cfg = cfg.market_maker
        self._risk = cfg.risk
        self._fees = FeeModel(cfg.fees)
        self._scorer = MarketScorer(cfg)
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode
        self._top_source = top_source
        self._last_mid: dict[str, float] = {}
        self._cooldown: dict[str, int] = {}
        self._quotes: dict[str, Quote] = {}          # активные котировки (все режимы)
        self._orders: dict[str, list[TrackedOrder]] = {}  # live-ордера по рынку

    # --- источники данных ---

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

    # --- котировки ---

    def compute_quote(self, market: Market, top: TopOfBook) -> Quote | None:
        c = self._cfg
        tick = market.tick_size
        if top.bid <= 0 or top.ask <= 0:
            return None

        fair = top.microprice
        # Inventory skew: длинный Yes → fair вниз (bid ниже, ask агрессивнее).
        skew = c.inventory_skew_k * self._inventory_frac(market)
        fair -= skew * max(c.half_spread, tick)

        # Полуспред: внутри rewards-диапазона, но не ниже fee-безубыточности.
        category = classify_category(market.question, market.category)
        min_half = self._fees.mm_min_half_spread(
            category, market.category, self._risk.min_edge_after_fees)
        half = max(c.half_spread, min_half, tick)
        if market.in_rewards_program:
            half = min(half, market.rewards_max_spread * 0.9)
            if half < max(min_half, tick):
                return None  # reward-диапазон уже fee-безубыточности — не котируем

        yes_bid = round_to_tick(fair - half, tick)
        yes_ask = round_to_tick(fair + half, tick)
        # Не пересекать книгу.
        yes_bid = min(yes_bid, round_to_tick(top.ask - tick, tick))
        yes_ask = max(yes_ask, round_to_tick(top.bid + tick, tick))
        if not tick <= yes_bid < yes_ask <= 1 - tick:
            return None

        size = float(math.floor(c.quote_size_usd / max(yes_bid, tick)))
        size = max(size, market.rewards_min_size)  # иначе не засчитается в rewards
        if size < market.min_order_size:
            return None
        # Кэп позиции на рынок.
        if abs(self._inventory_usd(market)) + size * yes_bid \
                > self._risk.max_position_per_market_usd * 2:
            return None
        return Quote(market=market, fair=fair, yes_bid=yes_bid,
                     no_bid=round_to_tick(1.0 - yes_ask, tick),
                     size=size, ts=time.time())

    def needs_requote(self, market: Market, top: TopOfBook) -> bool:
        """Гистерезис: не дёргать ордера без необходимости."""
        current = self._quotes.get(market.id)
        if current is None:
            return True
        moved = abs(top.microprice - current.fair)
        if moved >= self._cfg.requote_threshold_ticks * market.tick_size:
            return True
        return (time.time() - current.ts) >= self._cfg.requote_timer_sec

    # --- инвентарь ---

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

    # --- guard от adverse selection ---

    def guard_blocks(self, market: Market, top: TopOfBook) -> bool:
        c = self._cfg
        mid = top.mid
        prev = self._last_mid.get(market.id)
        self._last_mid[market.id] = mid
        if self._cooldown.get(market.id, 0) > 0:
            self._cooldown[market.id] -= 1
            return True
        if prev is not None and abs(mid - prev) >= c.guard_price_move:
            log.info("MM guard: %s mid %.3f -> %.3f — снимаем котировки, cooldown",
                     market.question[:40], prev, mid)
            self._cooldown[market.id] = c.guard_cooldown_cycles
            return True
        if market.volume_usd > 0 and \
                market.volume_24h_usd / market.volume_usd >= c.guard_volume_ratio:
            return True
        return False

    # --- исполнение ---

    def _cancel_market(self, market_id: str) -> None:
        self._quotes.pop(market_id, None)
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

    def _paper_fills(self) -> None:
        """Paper-режим: бид исполняется, если рынок проторговался сквозь него."""
        for quote in list(self._quotes.values()):
            m = quote.market
            yes_top = self._top(m.clob_token_ids[0])
            no_top = self._top(m.clob_token_ids[1])
            if yes_top is not None and 0 < yes_top.ask <= quote.yes_bid:
                self._record_fill(m, 0, quote.yes_bid, quote.size, None, "paper-filled")
                self._quotes.pop(m.id, None)
                log.info("MM paper fill: Yes %.3f x %.0f (%s)",
                         quote.yes_bid, quote.size, m.question[:40])
            elif no_top is not None and 0 < no_top.ask <= quote.no_bid:
                self._record_fill(m, 1, quote.no_bid, quote.size, None, "paper-filled")
                self._quotes.pop(m.id, None)
                log.info("MM paper fill: No %.3f x %.0f (%s)",
                         quote.no_bid, quote.size, m.question[:40])

    def _place(self, quote: Quote, quote_yes: bool, quote_no: bool) -> None:
        m = quote.market
        self._quotes[m.id] = quote
        if self._mode != "live":
            log.info("MM [%s] котировка %s: bid %.3f / ask %.3f (fair %.4f) x %.0f",
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

    # --- цикл ---

    def cycle(self, markets: list[Market]) -> list[Quote]:
        if not self._cfg.enabled:
            return []
        self._sync_live_fills()
        if self._mode == "paper":
            self._paper_fills()

        books = {m.clob_token_ids[0]: self._clob.order_book(m.clob_token_ids[0])
                 for m in markets if self._scorer.eligible(m)}
        selected = self._scorer.top_markets(markets, books)
        selected_ids = {m.id for m in selected}
        for market_id in list(self._quotes) + list(self._orders):
            if market_id not in selected_ids:
                self._cancel_market(market_id)

        active: list[Quote] = []
        for market in selected:
            top = self._top(market.clob_token_ids[0])
            if top is None:
                continue
            if self.guard_blocks(market, top):
                self._cancel_market(market.id)
                continue
            if not self.needs_requote(market, top):
                current = self._quotes.get(market.id)
                if current is not None:
                    active.append(current)
                continue

            quote = self.compute_quote(market, top)
            quote_yes, quote_no = self.sides_allowed(market)
            mid = top.mid
            # Экстремальный midpoint: двусторонняя котировка обязательна для
            # rewards; одностороннюю не ставим — покидаем рынок.
            if (mid < 0.10 or mid > 0.90) and not (quote_yes and quote_no):
                self._cancel_market(market.id)
                continue
            if quote is None or not (quote_yes or quote_no):
                self._cancel_market(market.id)
                continue
            self._cancel_market(market.id)
            self._place(quote, quote_yes, quote_no)
            active.append(quote)
        return active

    def local_order_ids(self) -> set[str]:
        return {o.order_id for orders in self._orders.values() for o in orders}

    def shutdown(self) -> None:
        for market_id in list(self._quotes) + list(self._orders):
            self._cancel_market(market_id)
