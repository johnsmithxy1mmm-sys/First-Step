"""Стратегия №1: структурный арбитраж внутри neg-risk событий.

В мульти-исходном событии (выборы, номинации) ровно один исход резолвится
YES. Два зеркальных окна:

  YES-корзина: Σ ask(Yes_i) < 1  → купить Yes всех исходов;
               выплата $1 за комплект, прибыль = 1 - Σask.
  NO-корзина:  Σ bid(Yes_i) > 1  ⟺ Σ ask(No_i) < n-1 → купить No всех
               исходов; выплата $(n-1) за комплект, прибыль = (n-1) - Σask(No).

Это математически гарантированная прибыль независимо от исхода, но окна
живут секунды-минуты и выедаются скоростными ботами. REST-поллинг этого
модуля — «медленный охотник»: он поймает только то, что осталось после
websocket-ботов. Capacity жёстко ограничен глубиной книг — сайзинг берёт
минимум по ногам.
"""

from __future__ import annotations

import logging
import math

from pydantic import BaseModel, Field

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .ledger import Ledger
from .models import Market, OrderBook, simple_estimate

log = logging.getLogger(__name__)


class ArbLeg(BaseModel):
    market: Market
    outcome_index: int          # 0 = Yes, 1 = No
    token_id: str
    ask: float                  # лучшая цена покупки этой ноги
    depth: float                # сколько акций доступно по ask


class BasketArb(BaseModel):
    event_id: str
    event_title: str
    side: str                   # "YES" | "NO"
    legs: list[ArbLeg] = Field(default_factory=list)

    @property
    def cost_per_set(self) -> float:
        return sum(leg.ask for leg in self.legs)

    @property
    def payout_per_set(self) -> float:
        # YES-корзина платит $1; NO-корзина платит $(n-1).
        return 1.0 if self.side == "YES" else float(len(self.legs) - 1)

    @property
    def profit_per_set(self) -> float:
        return self.payout_per_set - self.cost_per_set

    @property
    def profit_pct(self) -> float:
        return self.profit_per_set / self.cost_per_set if self.cost_per_set > 0 else 0.0

    def max_sets_by_depth(self) -> int:
        return int(min((leg.depth for leg in self.legs), default=0))


class ArbitrageScanner:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str):
        self._cfg = cfg.arbitrage
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode

    # --- отбор событий, стоящих похода в стаканы ---

    def prefilter_events(self, markets: list[Market]) -> list[list[Market]]:
        """Группы neg-risk рынков, где сумма Yes-цен Gamma намекает на окно."""
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
            # Порог мягче реального edge: Gamma-цены запаздывают, точность даст стакан.
            if total < 1.0 - self._cfg.prefilter_tolerance \
                    or total > 1.0 + self._cfg.prefilter_tolerance:
                suspicious.append(group)
        suspicious.sort(key=lambda g: abs(sum(m.outcome_prices[0] for m in g) - 1.0),
                        reverse=True)
        return suspicious[: self._cfg.max_events_per_cycle]

    # --- проверка по реальным стаканам ---

    def verify(self, group: list[Market]) -> BasketArb | None:
        yes_books: list[tuple[Market, OrderBook]] = []
        no_books: list[tuple[Market, OrderBook]] = []
        for m in group:
            yb = self._clob.order_book(m.clob_token_ids[0])
            nb = self._clob.order_book(m.clob_token_ids[1])
            if yb is None or nb is None or yb.best_ask <= 0 or nb.best_ask <= 0:
                return None  # без полного комплекта ног арбитража нет
            yes_books.append((m, yb))
            no_books.append((m, nb))

        candidates = []
        yes_arb = self._build("YES", [(m, b, 0) for m, b in yes_books])
        no_arb = self._build("NO", [(m, b, 1) for m, b in no_books])
        for arb in (yes_arb, no_arb):
            if arb is not None and arb.profit_pct >= self._cfg.min_profit_pct \
                    and arb.max_sets_by_depth() >= self._cfg.min_sets:
                candidates.append(arb)
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.profit_pct)

    def _build(self, side: str, legs_raw) -> BasketArb | None:
        legs = []
        for market, book, idx in legs_raw:
            ask = book.best_ask
            depth = next((l.size for l in sorted(book.asks, key=lambda x: x.price)
                          if l.size > 0), 0.0)
            legs.append(ArbLeg(market=market, outcome_index=idx,
                               token_id=market.clob_token_ids[idx],
                               ask=ask, depth=depth))
        arb = BasketArb(event_id=legs_raw[0][0].event_id,
                        event_title=legs_raw[0][0].event_title,
                        side=side, legs=legs)
        return arb if arb.profit_per_set > 0 else None

    # --- исполнение ---

    def execute(self, arb: BasketArb) -> float:
        """Покупает комплекты. Возвращает потраченные доллары (0 = не исполнено).

        Риск ноги: часть лимиток может не исполниться, если стакан сдвинулся, —
        тогда остаётся направленная позиция вместо арбитража. Лимитки по ask
        не переплатят, но гарантию комплекта не дают; поэтому min_profit_pct
        должен покрывать этот риск.
        """
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
            if self._trader is not None:
                try:
                    resp = self._trader.buy_limit(leg.token_id, price, float(sets),
                                                  neg_risk=True)
                    order_id = (resp or {}).get("orderID")
                except Exception as exc:
                    log.error("arb leg failed %s: %s — остальные ноги не переплатят",
                              leg.token_id[:16], exc)
                    continue
            self._ledger.record_trade(
                mode=self._mode,
                estimate=simple_estimate(leg.market, leg.outcome_index, price),
                category="arb", side="BUY", price=price, size=float(sets),
                order_id=order_id,
                status="filled" if self._trader else "sim-filled",
                strategy="arb",
            )
            spent += price * sets
        return spent

    # --- цикл ---

    def cycle(self, markets: list[Market]) -> list[BasketArb]:
        if not self._cfg.enabled:
            return []
        found: list[BasketArb] = []
        for group in self.prefilter_events(markets):
            arb = self.verify(group)
            if arb is None:
                continue
            found.append(arb)
            log.info("АРБИТРАЖ %s-корзина «%s»: %d ног, комплект $%.4f -> $%.2f "
                     "(+%.1f%%), глубина %d комплектов",
                     arb.side, arb.event_title[:50], len(arb.legs),
                     arb.cost_per_set, arb.payout_per_set,
                     arb.profit_pct * 100, arb.max_sets_by_depth())
            if self._cfg.execute:
                spent = self.execute(arb)
                if spent > 0:
                    log.info("арбитраж исполнен: $%.2f", spent)
        return found
