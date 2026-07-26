"""Polymarket CLOB: order-book reads, price history, trading client."""

from __future__ import annotations

import logging
import os

import httpx

from .config import BotConfig
from .http_client import get_with_backoff, make_client
from .models import BookLevel, OrderBook
from .ratelimit import RateLimited, TokenBucket

log = logging.getLogger(__name__)


class ClobReader:
    """Read-only CLOB access: book and order status need no signature."""

    def __init__(self, cfg: BotConfig, client: httpx.Client | None = None):
        self._cfg = cfg
        self._client = client or make_client(cfg.runtime.request_timeout_sec)
        # Self-throttle reads: a Cloudflare queue on overshoot is worse than
        # waiting locally, and a throttled MM cannot pull its quotes.
        self._reads = TokenBucket(cfg.ratelimit.reads_per_sec,
                                  cfg.ratelimit.reads_burst)

    def _throttle(self, what: str) -> bool:
        if self._reads.acquire(1.0, timeout=self._cfg.runtime.request_timeout_sec):
            return True
        log.warning("rate limit: dropped read %s (local bucket exhausted)", what)
        return False

    def order_book(self, token_id: str) -> OrderBook | None:
        if not self._throttle(f"book {token_id[:16]}"):
            return None
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
        if not self._throttle(f"history {token_id[:16]}"):
            return []
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
    """Snap an ORDER price to the market's tick, clamped to a tradable price.

    Every caller uses this for a price it is about to send to the exchange, and
    0 or 1 is never a valid order price here. Plain rounding produces them from
    ordinary inputs: 0.0004 at a 0.001 tick rounds to 0.0, and a SELL at 0.0
    gives the position away for nothing; 0.9996 rounds to 1.0, and a BUY there
    pays full face for a $1 payout. So clamp into [tick, 1 - tick] rather than
    trusting each call site to re-check. Callers that need "there is no real
    price here" must test the raw input (see chainarb._unwind_leg).
    """
    if tick <= 0:
        return price
    snapped = round(round(price / tick) * tick, 6)
    return min(max(snapped, tick), round(1.0 - tick, 6))


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
        # Positions live on the funder (proxy accounts) or, for a plain EOA
        # with no funder set, on the signer address itself — without this
        # fallback an EOA account could never see its own positions
        # (redeemer scan and the idempotency reconcile would both be blind).
        try:
            signer = self._client.get_address()
        except Exception:
            signer = None
        self._funder = funder or signer
        self._orders = TokenBucket(cfg.ratelimit.orders_per_sec,
                                   cfg.ratelimit.orders_burst)
        self._reads = TokenBucket(cfg.ratelimit.reads_per_sec,
                                  cfg.ratelimit.reads_burst)
        # HARD CONSTRAINT: the MM WS fastlane places orders from the websocket
        # recv thread (main.on_tick -> react_to_tick -> _place -> buy_limit), and
        # a stalled recv loop trips risk.ws_staleness_kill_sec. So the wait here
        # must stay a small FRACTION of that kill threshold, never a match for
        # it. A requote can issue a cancel plus two places, so budget for
        # several waits inside one tick.
        self._order_wait_sec = max(
            0.1, min(1.0, cfg.risk.ws_staleness_kill_sec / 20.0))

    def _limit_order(self, side, token_id: str, price: float, size: float,
                     neg_risk: bool, order_type: str = "GTC") -> dict:
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions

        # A dropped order is safe (callers read a missing orderID as "no fill");
        # a Cloudflare ban mid-session is not. So shed rather than queue: with a
        # 100-order burst allowance a dry bucket means genuine flooding, and
        # waiting it out would stall the WS recv thread (see _order_wait_sec).
        if not self._orders.acquire(1.0, timeout=self._order_wait_sec):
            log.error("rate limit: order NOT placed for %s (local bucket "
                      "exhausted) — treated as no fill", token_id[:16])
            return {}
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
        # A cancel REDUCES risk: take a token if one is free, but never refuse to
        # cancel because the bucket is dry — a live order left behind is worse.
        if not self._orders.try_acquire(1.0):
            log.warning("rate limit: cancel %s proceeding without a token "
                        "(risk reduction is never throttled away)", order_id[:16])
        self._client.cancel(order_id)

    def cancel_all(self) -> None:
        """Bulk-cancel all orders (emergency kill-switch action).

        Deliberately NOT rate limited: this is the kill-switch. Throttling the
        one call that flattens the book is how a safety system kills you.
        """
        self._orders.try_acquire(1.0)   # account for it, but never wait
        self._client.cancel_all()

    def _throttle_read(self, what: str) -> None:
        """Raise rather than return a degraded answer — see RateLimited."""
        if not self._reads.acquire(1.0, timeout=10.0):
            raise RateLimited(f"local read bucket exhausted for {what}")

    def order_status(self, order_id: str) -> dict:
        """{'status': 'LIVE'|'MATCHED'|'CANCELED'..., 'size_matched': float}."""
        self._throttle_read(f"order_status {order_id[:16]}")
        raw = self._client.get_order(order_id) or {}
        return {
            "status": str(raw.get("status", "unknown")).lower(),
            "size_matched": float(raw.get("size_matched") or 0),
        }

    def matched_size(self, order_id: str | None, requested: float) -> float:
        """Shares actually matched for `order_id`. Never trusts the request.

        An order id coming back means "accepted", not "filled": a GTC order can
        rest untouched and a killed FOK still returns a response. Booking the
        requested size on the strength of an id creates shares the account does
        not hold — phantom PnL, and for a basket a structure booked complete
        while a leg is missing. On an unreadable status we return 0: claiming
        nothing filled is recoverable, claiming a fill that did not happen is not.
        """
        if order_id is None:
            return 0.0
        try:
            matched = float(self.order_status(order_id).get("size_matched", 0.0))
        except Exception as exc:
            log.warning("could not confirm fill for %s: %s — assuming 0",
                        order_id[:16], exc)
            return 0.0
        return max(0.0, min(matched, float(requested)))

    def open_orders(self) -> list[dict]:
        self._throttle_read("open_orders")
        return self._client.get_orders() or []

    def api_positions(self) -> list[dict]:
        """Actual wallet positions from data-api (for idempotency reconcile)."""
        if not self._funder:
            return []
        self._throttle_read("api_positions")
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
