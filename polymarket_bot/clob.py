"""CLOB Polymarket: чтение стакана, история цен, торговый клиент."""

from __future__ import annotations

import logging
import os

import httpx

from .config import BotConfig
from .http_client import get_with_backoff, make_client
from .models import BookLevel, OrderBook

log = logging.getLogger(__name__)


class ClobReader:
    """Read-only доступ к CLOB: стакан, статусы ордеров не требуют подписи."""

    def __init__(self, cfg: BotConfig, client: httpx.Client | None = None):
        self._cfg = cfg
        self._client = client or make_client(cfg.runtime.request_timeout_sec)

    def order_book(self, token_id: str) -> OrderBook | None:
        try:
            resp = get_with_backoff(
                self._client,
                f"{self._cfg.runtime.clob_host}/book",
                params={"token_id": token_id},
                max_retries=1,
            )
        except httpx.HTTPError as exc:
            log.warning("book %s: %s", token_id[:16], exc)
            return None
        data = resp.json()

        def levels(raw) -> list[BookLevel]:
            out = []
            for level in raw or []:
                try:
                    out.append(BookLevel(price=float(level["price"]), size=float(level["size"])))
                except (KeyError, TypeError, ValueError):
                    continue
            return out

        book = OrderBook(bids=levels(data.get("bids")), asks=levels(data.get("asks")))
        return book if (book.bids or book.asks) else None

    def price_history(self, token_id: str, start_ts: int, end_ts: int) -> list[tuple[int, float]]:
        """История цен токена (для бэктеста): [(unix_ts, price), ...]."""
        try:
            resp = get_with_backoff(
                self._client,
                f"{self._cfg.runtime.clob_host}/prices-history",
                params={"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": 720},
                max_retries=1,
            )
        except httpx.HTTPError:
            return []
        points = resp.json().get("history") or []
        out = []
        for p in points:
            try:
                out.append((int(p["t"]), float(p["p"])))
            except (KeyError, TypeError, ValueError):
                continue
        return out


def round_to_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return round(round(price / tick) * tick, 6)


class Trader:
    """Подписанные операции через официальный py-clob-client. Только для live."""

    def __init__(self, cfg: BotConfig):
        try:
            from py_clob_client.client import ClobClient
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install py-clob-client для live-режима") from exc

        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            raise SystemExit("POLYMARKET_PRIVATE_KEY не задан (см. .env.example)")
        funder = os.environ.get("POLYMARKET_FUNDER")
        signature_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "0"))
        if signature_type in (1, 2) and not funder:
            raise SystemExit("Для signature_type 1/2 требуется POLYMARKET_FUNDER")

        kwargs: dict = dict(key=private_key, chain_id=cfg.runtime.chain_id,
                            signature_type=signature_type)
        if funder:
            kwargs["funder"] = funder
        self._client = ClobClient(cfg.runtime.clob_host, **kwargs)
        self._client.set_api_creds(self._client.create_or_derive_api_creds())
        self._data_api = cfg.runtime.data_api_host
        self._funder = funder

    def _limit_order(self, side, token_id: str, price: float, size: float,
                     neg_risk: bool) -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

        args = OrderArgs(price=price, size=size, side=side, token_id=token_id)
        options = PartialCreateOrderOptions(neg_risk=True) if neg_risk else None
        signed = self._client.create_order(args, options)
        return self._client.post_order(signed, OrderType.GTC) or {}

    def buy_limit(self, token_id: str, price: float, size: float, neg_risk: bool = False) -> dict:
        from py_clob_client.order_builder.constants import BUY
        return self._limit_order(BUY, token_id, price, size, neg_risk)

    def sell_limit(self, token_id: str, price: float, size: float, neg_risk: bool = False) -> dict:
        from py_clob_client.order_builder.constants import SELL
        return self._limit_order(SELL, token_id, price, size, neg_risk)

    def cancel(self, order_id: str) -> None:
        self._client.cancel(order_id)

    def order_status(self, order_id: str) -> dict:
        """{'status': 'LIVE'|'MATCHED'|'CANCELED'..., 'size_matched': float}."""
        raw = self._client.get_order(order_id) or {}
        return {
            "status": str(raw.get("status", "unknown")).lower(),
            "size_matched": float(raw.get("size_matched") or 0),
        }

    def open_orders(self) -> list[dict]:
        return self._client.get_orders() or []

    def api_positions(self) -> list[dict]:
        """Фактические позиции кошелька из data-api (для сверки идемпотентности)."""
        if not self._funder:
            return []
        client = make_client(15.0)
        try:
            resp = get_with_backoff(
                client, f"{self._data_api}/positions",
                params={"user": self._funder, "limit": 500}, max_retries=2,
            )
            return resp.json() or []
        except httpx.HTTPError:
            return []
        finally:
            client.close()
