"""Работа с CLOB Polymarket: проверка стакана и размещение ордеров."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import requests

from .config import BotConfig
from .strategy import Candidate


@dataclass
class BookQuote:
    best_ask: float   # лучшая цена продажи (по ней покупаем)
    ask_depth: float  # сколько акций доступно по этой цене


def get_best_ask(cfg: BotConfig, token_id: str, session: requests.Session | None = None) -> BookQuote | None:
    """Лучший ask из стакана. None — если стакан пустой или недоступен."""
    session = session or requests.Session()
    resp = session.get(f"{cfg.clob_host}/book", params={"token_id": token_id}, timeout=30)
    if resp.status_code != 200:
        return None
    asks = resp.json().get("asks") or []
    best = None
    for level in asks:
        try:
            price, size = float(level["price"]), float(level["size"])
        except (KeyError, TypeError, ValueError):
            continue
        if size <= 0:
            continue
        if best is None or price < best.best_ask:
            best = BookQuote(best_ask=price, ask_depth=size)
    return best


def round_to_tick(price: float, tick: float) -> float:
    """Цена ордера должна быть кратна тику рынка (обычно 0.001 или 0.01)."""
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 6)


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

    def buy_limit(self, candidate: Candidate, price: float, size: float) -> dict:
        """Ставит лимитный ордер на покупку. GTC: висит в стакане, пока не исполнится."""
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        from py_clob_client.order_builder.constants import BUY

        args = OrderArgs(price=price, size=size, side=BUY, token_id=candidate.token_id)
        options = PartialCreateOrderOptions(neg_risk=True) if candidate.neg_risk else None
        signed = self._client.create_order(args, options)
        return self._client.post_order(signed, OrderType.GTC)
