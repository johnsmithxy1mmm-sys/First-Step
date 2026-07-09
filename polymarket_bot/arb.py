"""Арбитраж neg-risk событий: сумма цен всех исходов < $1.

В neg-risk событии (например, выборы с несколькими кандидатами) побеждает ровно
один исход. Если купить Yes каждого кандидата, выплата гарантированно $1 за
комплект. Когда суммарная цена комплекта < $1 — это безрисковая прибыль,
единственный «бесплатный сыр» на предикшн-рынках. Такие окна возникают при
резких новостях, когда маркетмейкеры не успевают переставить котировки.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import BotConfig
from .gamma import market_float, parse_json_list


@dataclass
class ArbLeg:
    question: str
    token_id: str
    ask: float
    tick_size: float
    min_order_size: float


@dataclass
class ArbOpportunity:
    event_id: str
    title: str
    sum_asks: float
    legs: list[ArbLeg]

    @property
    def edge(self) -> float:
        """Гарантированная маржа с одного комплекта: $1 выплата минус цена входа."""
        return 1.0 - self.sum_asks

    def sets_for_stake(self, stake_usd: float) -> int:
        """Сколько полных комплектов купить на заданную сумму (одинаковый размер каждой ноги)."""
        if self.sum_asks <= 0:
            return 0
        sets = math.floor(stake_usd / self.sum_asks)
        min_size = max((leg.min_order_size for leg in self.legs), default=5.0)
        return sets if sets >= min_size else 0


def _leg_ask(market: dict) -> float | None:
    """Цена покупки Yes: bestAsk из Gamma, запасной вариант — outcomePrices[0]."""
    ask = market_float(market, "bestAsk")
    if ask > 0:
        return ask
    prices = parse_json_list(market.get("outcomePrices"))
    if prices:
        try:
            price = float(prices[0])
            return price if price > 0 else None
        except (TypeError, ValueError):
            return None
    return None


def find_arbs(events, cfg: BotConfig) -> list[ArbOpportunity]:
    """Ищет neg-risk события, где комплект всех Yes стоит меньше $1 - edge."""
    out: list[ArbOpportunity] = []
    for event in events:
        if not event.get("negRisk"):
            continue
        markets = event.get("markets") or []
        if len(markets) < 2:
            continue

        legs: list[ArbLeg] = []
        ok = True
        for m in markets:
            if m.get("closed") or not m.get("enableOrderBook", True):
                ok = False
                break
            ask = _leg_ask(m)
            token_ids = parse_json_list(m.get("clobTokenIds"))
            if ask is None or not token_ids:
                ok = False
                break
            legs.append(ArbLeg(
                question=m.get("question") or m.get("groupItemTitle") or "",
                token_id=str(token_ids[0]),
                ask=ask,
                tick_size=market_float(m, "orderPriceMinTickSize") or 0.001,
                min_order_size=market_float(m, "orderMinSize") or 5.0,
            ))
        if not ok or not legs:
            continue

        sum_asks = sum(leg.ask for leg in legs)
        if sum_asks < 1.0 - cfg.arb_min_edge:
            out.append(ArbOpportunity(
                event_id=str(event.get("id", "")),
                title=event.get("title") or "",
                sum_asks=round(sum_asks, 4),
                legs=legs,
            ))
    out.sort(key=lambda a: a.edge, reverse=True)
    return out


def verify_arb(arb: ArbOpportunity, cfg: BotConfig, session) -> ArbOpportunity | None:
    """Пересчитывает арбитраж по реальным стаканам CLOB (Gamma может отставать)."""
    from . import clob

    fresh_legs: list[ArbLeg] = []
    for leg in arb.legs:
        quote = clob.get_quote(cfg, leg.token_id, session)
        if quote is None or quote.best_ask <= 0:
            return None
        fresh_legs.append(ArbLeg(
            question=leg.question,
            token_id=leg.token_id,
            ask=quote.best_ask,
            tick_size=leg.tick_size,
            min_order_size=leg.min_order_size,
        ))
    sum_asks = sum(leg.ask for leg in fresh_legs)
    if sum_asks >= 1.0 - cfg.arb_min_edge:
        return None
    return ArbOpportunity(event_id=arb.event_id, title=arb.title,
                          sum_asks=round(sum_asks, 4), legs=fresh_legs)


def execute_arb(arb: ArbOpportunity, trader, log, cfg: BotConfig) -> tuple[int, float]:
    """Покупает комплект: лимитные ордера по ask каждой ноги.

    Риск: часть ног может исполниться, часть — зависнуть, если стакан сдвинулся
    (лимитка останется стоять по своей цене — переплаты не будет, но и
    гарантии комплекта тоже). Поэтому по умолчанию arb_execute=False.
    """
    sets = arb.sets_for_stake(cfg.arb_stake_usd)
    if sets <= 0:
        return 0, 0.0
    spent = 0.0
    placed = 0
    for leg in arb.legs:
        from .clob import round_to_tick
        price = round_to_tick(leg.ask, leg.tick_size)
        try:
            resp = trader.buy_limit(leg.token_id, price, float(sets), neg_risk=True)
        except Exception:
            continue  # нога не встала — остальные лимитки не переплатят
        log.record(
            market_id=arb.event_id, question=f"[ARB {arb.title}] {leg.question}",
            slug="", outcome="Yes", token_id=leg.token_id,
            price=price, size=float(sets), live=True, side="BUY",
            order_id=(resp or {}).get("orderID"),
            status=(resp or {}).get("status", "unknown"),
        )
        placed += 1
        spent += price * sets
    return placed, spent
