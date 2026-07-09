"""Стратегия №3: маркет-мейкинг + liquidity rewards — база денежного потока.

Котируем двусторонне ликвидные средние рынки (не хвосты). На Polymarket
двусторонняя котировка делается ДВУМЯ ПОКУПКАМИ: бид на Yes-токен по
(mid - s/2) и бид на No-токен по (1 - (mid + s/2)). Если исполняются оба,
мы держим Yes+No = $1 к выкупу, заплатив 1 - spread: прибыль = спред.
Котирование внутри reward-диапазона дополнительно собирает liquidity
rewards программы Polymarket.

Главный риск — adverse selection перед новостями: нас переезжают
информированные. Лечение (автоматическое):
  - guard по движению цены: mid сдвинулся сильнее порога с прошлого
    цикла → снять котировки и остыть cooldown циклов;
  - guard по всплеску объёма: суточный оборот аномален к среднему → не котировать;
  - лимит инвентаря: перекос Yes/No выше кэпа → котировать только
    сокращающую перекос сторону.
"""

from __future__ import annotations

import logging
import math

from pydantic import BaseModel

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .ledger import Ledger
from .models import Market, OrderBook, simple_estimate

log = logging.getLogger(__name__)


class Quote(BaseModel):
    market: Market
    yes_bid: float      # бид на Yes-токен
    no_bid: float       # бид на No-токен
    size: float         # акций на каждую сторону

    @property
    def captured_spread(self) -> float:
        """Прибыль на пару, если исполнятся обе стороны."""
        return 1.0 - self.yes_bid - self.no_bid


class TrackedOrder(BaseModel):
    order_id: str
    market: Market
    outcome_index: int
    price: float
    size: float
    matched_recorded: float = 0.0


class MarketMaker:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str):
        self._cfg = cfg.market_maker
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode
        self._last_mid: dict[str, float] = {}
        self._cooldown: dict[str, int] = {}
        self._orders: dict[str, list[TrackedOrder]] = {}  # market_id -> активные котировки

    # --- отбор рынков ---

    def select_markets(self, markets: list[Market]) -> list[Market]:
        c = self._cfg
        eligible = [
            m for m in markets
            if not m.closed and m.enable_order_book
            and len(m.clob_token_ids) >= 2 and m.outcome_prices
            and c.price_lo <= m.outcome_prices[0] <= c.price_hi   # средние, не хвосты
            and m.volume_24h_usd >= c.min_volume_24h_usd
            and (m.days_to_resolution() or 0) >= c.min_days_to_resolution
        ]
        eligible.sort(key=lambda m: m.volume_24h_usd, reverse=True)
        return eligible[: c.max_markets]

    # --- котировки ---

    def compute_quote(self, market: Market, book: OrderBook) -> Quote | None:
        c = self._cfg
        tick = market.tick_size
        mid = book.mid
        if mid <= 0 or book.best_bid <= 0 or book.best_ask <= 0:
            return None
        # Не лезем в рынки, где спред уже уже нашего: там нечего зарабатывать.
        if book.best_ask - book.best_bid < c.half_spread:
            return None

        half = max(c.half_spread, tick)
        yes_bid = round_to_tick(mid - half, tick)
        yes_ask = round_to_tick(mid + half, tick)
        # Не пересекать книгу: бид ниже best ask, «ask» (бид на No) выше best bid.
        yes_bid = min(yes_bid, round_to_tick(book.best_ask - tick, tick))
        yes_ask = max(yes_ask, round_to_tick(book.best_bid + tick, tick))
        if not tick <= yes_bid < yes_ask <= 1 - tick:
            return None

        no_bid = round_to_tick(1.0 - yes_ask, tick)
        size = float(math.floor(c.quote_size_usd / max(yes_bid, tick)))
        if size < market.min_order_size:
            return None
        return Quote(market=market, yes_bid=yes_bid, no_bid=no_bid, size=size)

    # --- защита от adverse selection ---

    def guard_blocks(self, market: Market, book: OrderBook) -> bool:
        c = self._cfg
        mid = book.mid
        prev = self._last_mid.get(market.id)
        self._last_mid[market.id] = mid

        if self._cooldown.get(market.id, 0) > 0:
            self._cooldown[market.id] -= 1
            return True
        if prev is not None and abs(mid - prev) >= c.guard_price_move:
            log.info("MM guard: %s mid сдвинулся %.3f -> %.3f — снимаем котировки",
                     market.question[:40], prev, mid)
            self._cooldown[market.id] = c.guard_cooldown_cycles
            return True
        if market.volume_usd > 0 and \
                market.volume_24h_usd / market.volume_usd >= c.guard_volume_ratio:
            log.info("MM guard: %s всплеск объёма (24h/total=%.2f) — пропуск",
                     market.question[:40],
                     market.volume_24h_usd / market.volume_usd)
            return True
        return False

    # --- инвентарь ---

    def inventory_skew_usd(self, market: Market) -> float:
        """Перекос: +$ = лишние Yes, -$ = лишние No."""
        yes_token, no_token = market.clob_token_ids[0], market.clob_token_ids[1]
        skew = 0.0
        for p in self._ledger.open_positions(self._mode):
            if p.token_id == yes_token:
                skew += p.cost_usd
            elif p.token_id == no_token:
                skew -= p.cost_usd
        return skew

    def sides_allowed(self, market: Market) -> tuple[bool, bool]:
        """(котировать Yes-бид, котировать No-бид) с учётом кэпа инвентаря."""
        skew = self.inventory_skew_usd(market)
        cap = self._cfg.inventory_cap_usd
        return skew < cap, skew > -cap

    # --- исполнение ---

    def _cancel_market_orders(self, market_id: str) -> None:
        for order in self._orders.pop(market_id, []):
            if self._trader is not None:
                try:
                    self._trader.cancel(order.order_id)
                except Exception:
                    pass

    def _sync_fills(self) -> None:
        """Фиксирует исполнившиеся куски котировок в леджере (live)."""
        if self._trader is None:
            return
        for market_id, orders in self._orders.items():
            for order in orders:
                try:
                    status = self._trader.order_status(order.order_id)
                except Exception:
                    continue
                new_fill = status.get("size_matched", 0.0) - order.matched_recorded
                if new_fill > 0:
                    self._ledger.record_trade(
                        mode=self._mode,
                        estimate=simple_estimate(order.market, order.outcome_index,
                                                 order.price),
                        category="mm", side="BUY", price=order.price, size=new_fill,
                        order_id=order.order_id, status="filled", strategy="mm",
                    )
                    order.matched_recorded += new_fill
                    log.info("MM fill: %s %.3f x %.0f (%s)",
                             "Yes" if order.outcome_index == 0 else "No",
                             order.price, new_fill, order.market.question[:40])

    def _place(self, quote: Quote, quote_yes: bool, quote_no: bool) -> None:
        m = quote.market
        placed: list[TrackedOrder] = []
        legs = []
        if quote_yes:
            legs.append((0, m.clob_token_ids[0], quote.yes_bid))
        if quote_no:
            legs.append((1, m.clob_token_ids[1], quote.no_bid))
        for idx, token, price in legs:
            if self._trader is None:
                log.info("MM [dry-run] котировка %s bid %.3f x %.0f (%s)",
                         "Yes" if idx == 0 else "No", price, quote.size,
                         m.question[:40])
                continue
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
        self._sync_fills()
        quotes: list[Quote] = []
        for market in self.select_markets(markets):
            self._cancel_market_orders(market.id)  # cancel-replace
            book = self._clob.order_book(market.clob_token_ids[0])
            if book is None or self.guard_blocks(market, book):
                continue
            quote = self.compute_quote(market, book)
            if quote is None:
                continue
            quote_yes, quote_no = self.sides_allowed(market)
            if not quote_yes and not quote_no:
                continue
            self._place(quote, quote_yes, quote_no)
            quotes.append(quote)
        log.info("MM: котируем %d рынков", len(quotes))
        return quotes

    def shutdown(self) -> None:
        """Снять все котировки (вызывается при остановке бота)."""
        for market_id in list(self._orders):
            self._cancel_market_orders(market_id)
