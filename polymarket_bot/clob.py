"""Работа с CLOB Polymarket: стакан, расчёт цены входа, размещение и отмена ордеров."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import requests

from .config import BotConfig
from .strategy import Candidate


@dataclass
class BookQuote:
    best_ask: float   # лучшая цена продажи (0 = ask-ов нет)
    ask_depth: float  # сколько акций доступно по best_ask
    best_bid: float   # лучшая цена покупки (0 = бидов нет)
    bid_depth: float


def get_quote(cfg: BotConfig, token_id: str, session: requests.Session | None = None) -> BookQuote | None:
    """Лучшие bid/ask из стакана. None — если стакан недоступен или пуст."""
    session = session or requests.Session()
    try:
        resp = session.get(f"{cfg.clob_host}/book", params={"token_id": token_id}, timeout=30)
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    book = resp.json()

    def best(levels, want_min: bool) -> tuple[float, float]:
        chosen = (0.0, 0.0)
        for level in levels or []:
            try:
                price, size = float(level["price"]), float(level["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if size <= 0:
                continue
            if chosen[0] == 0 or (price < chosen[0]) == want_min:
                chosen = (price, size)
        return chosen

    ask, ask_depth = best(book.get("asks"), want_min=True)
    bid, bid_depth = best(book.get("bids"), want_min=False)
    if ask == 0 and bid == 0:
        return None
    return BookQuote(best_ask=ask, ask_depth=ask_depth, best_bid=bid, bid_depth=bid_depth)


def round_to_tick(price: float, tick: float) -> float:
    """Цена ордера должна быть кратна тику рынка (обычно 0.001 или 0.01)."""
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 6)


def compute_entry_price(quote: BookQuote, c: Candidate, cfg: BotConfig) -> float | None:
    """Цена лимитного ордера на покупку.

    taker — по лучшему ask: исполняется сразу, но дороже.
    maker — свой ордер у бида (best_bid + тик, но не выше ask - тик): дешевле,
    зато может не исполниться. На ценах 0.005–0.01 разница в один тик — это
    десятки процентов будущей выплаты, поэтому maker — режим по умолчанию.
    """
    tick = c.tick_size or 0.001
    if quote.best_ask > 0 and quote.best_ask > cfg.max_price * 3:
        return None  # цена в Gamma устарела: реальный стакан ушёл далеко вверх

    if cfg.entry_mode == "taker":
        # Смысл taker — мгновенное исполнение: если ask дороже порога, не берём
        # (зажатая до max_price лимитка просто повиснет, это уже не taker).
        if quote.best_ask <= 0 or quote.best_ask > cfg.max_price:
            return None
        price = quote.best_ask
    else:
        if quote.best_bid > 0 and quote.best_ask > 0:
            price = min(quote.best_ask - tick, quote.best_bid + tick)
        elif quote.best_ask > 0:
            price = quote.best_ask - tick  # бидов нет: встаём первым под ask
        else:
            price = quote.best_bid + tick  # ask-ов нет: чуть выше лучшего бида
        price = max(price, tick)

    price = round_to_tick(min(price, cfg.max_price), tick)
    if not cfg.min_price <= price <= cfg.max_price:
        return None
    return price


def shares_for_stake(stake_usd: float, price: float, min_order_size: float) -> float:
    """Сколько акций покупаем на ставку. Целое число вниз, но не меньше минимума биржи."""
    if price <= 0:
        return 0.0
    return max(float(math.floor(stake_usd / price)), min_order_size)


class Trader:
    """Обёртка над py-clob-client. Создаётся только для реальной торговли (--live)."""

    def __init__(self, cfg: BotConfig):
        try:
            from py_clob_client.client import ClobClient
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "Для реальной торговли установите py-clob-client: pip install py-clob-client"
            ) from exc

        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            raise SystemExit(
                "Не задан POLYMARKET_PRIVATE_KEY. Экспортируйте приватный ключ кошелька:\n"
                "  export POLYMARKET_PRIVATE_KEY=0x..."
            )
        funder = os.environ.get("POLYMARKET_FUNDER")  # адрес прокси-кошелька Polymarket
        signature_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", cfg.signature_type))
        if signature_type in (1, 2) and not funder:
            raise SystemExit(
                "Для аккаунта через email/браузерный кошелёк задайте POLYMARKET_FUNDER "
                "(адрес депозита из настроек Polymarket)."
            )

        kwargs = dict(key=private_key, chain_id=cfg.chain_id, signature_type=signature_type)
        if funder:
            kwargs["funder"] = funder
        self._client = ClobClient(cfg.clob_host, **kwargs)
        self._client.set_api_creds(self._client.create_or_derive_api_creds())

    def _limit_order(self, side, token_id: str, price: float, size: float, neg_risk: bool) -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

        args = OrderArgs(price=price, size=size, side=side, token_id=token_id)
        options = PartialCreateOrderOptions(neg_risk=True) if neg_risk else None
        signed = self._client.create_order(args, options)
        return self._client.post_order(signed, OrderType.GTC)

    def buy_limit(self, token_id: str, price: float, size: float, neg_risk: bool = False) -> dict:
        """Лимитный ордер на покупку. GTC: висит в стакане, пока не исполнится."""
        from py_clob_client.order_builder.constants import BUY
        return self._limit_order(BUY, token_id, price, size, neg_risk)

    def sell_limit(self, token_id: str, price: float, size: float, neg_risk: bool = False) -> dict:
        """Лимитный ордер на продажу (фиксация прибыли до резолюции)."""
        from py_clob_client.order_builder.constants import SELL
        return self._limit_order(SELL, token_id, price, size, neg_risk)

    def cancel(self, order_id: str) -> None:
        """Снимает неисполненный ордер, освобождая замороженный бюджет."""
        self._client.cancel(order_id)
