"""Polymarket CLOB: order-book reads, price history, trading client."""

from __future__ import annotations

import logging
import os

import httpx

from .config import BotConfig
from .http_client import get_with_backoff, make_client
from .models import BookLevel, OrderBook

log = logging.getLogger(__name__)


class ClobReader:
    """Read-only CLOB access: book and order status need no signature."""

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
        except httpx.HTTPStatusError as exc:
            # 404 = the CLOB has no book for this token (inactive/untradable
            # market that Gamma still lists). Expected and benign — the callers
            # already skip a None book, so do not shout about it.
            if exc.response.status_code == 404:
                log.debug("book %s: no CLOB book (404)", token_id[:16])
            else:
                log.warning("book %s: %s", token_id[:16], exc)
            return None
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
        """Token price history (for backtests): [(unix_ts, price), ...]."""
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
    """Signed operations via the official py-clob-client. Live only."""

    def __init__(self, cfg: BotConfig):
        try:
            from py_clob_client.client import ClobClient
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("pip install py-clob-client for live mode") from exc

        private_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            raise SystemExit("POLYMARKET_PRIVATE_KEY not set (see .env.example)")
        funder = os.environ.get("POLYMARKET_FUNDER")
        signature_type = int(os.environ.get("POLYMARKET_SIGNATURE_TYPE", "0"))
        if signature_type in (1, 2, 3) and not funder:
            raise SystemExit("signature_type 1/2/3 requires POLYMARKET_FUNDER")

        def build(sig_type: int):
            kwargs: dict = dict(key=private_key, chain_id=cfg.runtime.chain_id,
                                signature_type=sig_type)
            if funder:
                kwargs["funder"] = funder
            client = ClobClient(cfg.runtime.clob_host, **kwargs)
            client.set_api_creds(client.create_or_derive_api_creds())
            return client

        # Known bug: sigtype 3 (deposit wallets / POLY_1271) may misbehave in
        # the SDK — on failure we fall back to sigtype 2.
        try:
            self._client = build(signature_type)
        except Exception as exc:
            if signature_type == 3:
                log.warning("signature_type=3 failed (%s) — falling back to 2 (proxy)", exc)
                signature_type = 2
                self._client = build(signature_type)
            else:
                raise
        self.signature_type = signature_type
        log.info("Trader: signature mode signature_type=%d, funder=%s",
                 signature_type, (funder or "-")[:12])
        self._data_api = cfg.runtime.data_api_host
        self._funder = funder

    def _limit_order(self, side, token_id: str, price: float, size: float,
                     neg_risk: bool, order_type: str = "GTC") -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

        args = OrderArgs(price=price, size=size, side=side, token_id=token_id)
        options = PartialCreateOrderOptions(neg_risk=True) if neg_risk else None
        signed = self._client.create_order(args, options)
        # No market orders on the platform: aggressive legs are FOK/IOC limits.
        ot = getattr(OrderType, order_type, OrderType.GTC)
        return self._client.post_order(signed, ot) or {}

    def buy_limit(self, token_id: str, price: float, size: float,
                  neg_risk: bool = False, order_type: str = "GTC") -> dict:
        from py_clob_client.order_builder.constants import BUY
        return self._limit_order(BUY, token_id, price, size, neg_risk, order_type)

    def sell_limit(self, token_id: str, price: float, size: float,
                   neg_risk: bool = False, order_type: str = "GTC") -> dict:
        from py_clob_client.order_builder.constants import SELL
        return self._limit_order(SELL, token_id, price, size, neg_risk, order_type)

    def cancel(self, order_id: str) -> None:
        self._client.cancel(order_id)

    def cancel_all(self) -> None:
        """Bulk-cancel all orders (emergency kill-switch action)."""
        self._client.cancel_all()

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
        """Actual wallet positions from data-api (for idempotency reconcile)."""
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
