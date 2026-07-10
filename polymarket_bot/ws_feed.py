"""WS-стаканы: авто-reconnect, heartbeat, gap-detect.

Websocket-поток CLOB — механизм «мгновенно»: изменения топа книги приходят
за миллисекунды вместо REST-поллинга. На каждом обновлении дёргается
callback (MM-репрайс, мгновенные выходы). При потере потока дольше
staleness_kill_sec вызывается on_disconnect — kill-switch снимает котировки:
торговать вслепую нельзя.

Логика разбора сообщений вынесена в BookStore — тестируется без сети.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Callable

from pydantic import BaseModel

log = logging.getLogger(__name__)


class TopOfBook(BaseModel):
    bid: float = 0.0
    bid_size: float = 0.0
    ask: float = 0.0
    ask_size: float = 0.0
    ts: float = 0.0

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.bid or self.ask

    @property
    def microprice(self) -> float:
        """Микроцена: mid, взвешенный объёмами сторон — честнее для fair value."""
        total = self.bid_size + self.ask_size
        if self.bid > 0 and self.ask > 0 and total > 0:
            return (self.bid * self.ask_size + self.ask * self.bid_size) / total
        return self.mid


class BookStore:
    """L2-книги по токенам из WS-сообщений CLOB (event_type: book / price_change)."""

    def __init__(self) -> None:
        self._bids: dict[str, dict[float, float]] = {}
        self._asks: dict[str, dict[float, float]] = {}
        self._lock = threading.Lock()

    def handle(self, msg: dict) -> str | None:
        """Обрабатывает одно сообщение; возвращает token_id, если книга изменилась."""
        token = str(msg.get("asset_id") or msg.get("market") or "")
        if not token:
            return None
        event = msg.get("event_type")
        with self._lock:
            if event == "book":
                self._bids[token] = self._levels(msg.get("bids") or msg.get("buys"))
                self._asks[token] = self._levels(msg.get("asks") or msg.get("sells"))
                return token
            if event == "price_change":
                bids = self._bids.setdefault(token, {})
                asks = self._asks.setdefault(token, {})
                changed = False
                for ch in msg.get("changes") or []:
                    try:
                        price, size = float(ch["price"]), float(ch["size"])
                        side = str(ch.get("side", "")).upper()
                    except (KeyError, TypeError, ValueError):
                        continue
                    levels = bids if side == "BUY" else asks
                    if size <= 0:
                        levels.pop(price, None)
                    else:
                        levels[price] = size
                    changed = True
                return token if changed else None
        return None

    @staticmethod
    def _levels(raw) -> dict[float, float]:
        out: dict[float, float] = {}
        for level in raw or []:
            try:
                price, size = float(level["price"]), float(level["size"])
            except (KeyError, TypeError, ValueError):
                continue
            if size > 0:
                out[price] = size
        return out

    def top(self, token: str) -> TopOfBook | None:
        with self._lock:
            bids = self._bids.get(token)
            asks = self._asks.get(token)
            if not bids and not asks:
                return None
            bid = max(bids) if bids else 0.0
            ask = min(asks) if asks else 0.0
            return TopOfBook(
                bid=bid, bid_size=bids.get(bid, 0.0) if bids else 0.0,
                ask=ask, ask_size=asks.get(ask, 0.0) if asks else 0.0,
                ts=time.time(),
            )


class WSFeed(threading.Thread):
    """Поток websocket-подписки на стаканы выбранных токенов."""

    daemon = True

    def __init__(self, url: str,
                 on_update: Callable[[str, TopOfBook], None] | None = None,
                 on_disconnect: Callable[[float], None] | None = None,
                 staleness_kill_sec: float = 10.0,
                 ping_interval_sec: float = 10.0):
        super().__init__(name="ws-feed")
        self._url = url
        self._on_update = on_update
        self._on_disconnect = on_disconnect
        self._staleness = staleness_kill_sec
        self._ping_interval = ping_interval_sec
        self.store = BookStore()
        self._desired: set[str] = set()
        self._resubscribe = threading.Event()
        self._stop = threading.Event()
        self._last_msg_ts = 0.0
        self._outage_reported = False

    # --- публичный интерфейс ---

    def watch(self, tokens: set[str]) -> None:
        tokens = set(tokens)
        if tokens != self._desired:
            self._desired = tokens
            self._resubscribe.set()

    def top(self, token: str) -> TopOfBook | None:
        return self.store.top(token)

    @property
    def healthy(self) -> bool:
        return bool(self._last_msg_ts) and (time.time() - self._last_msg_ts) < self._staleness

    def stop(self) -> None:
        self._stop.set()
        self._resubscribe.set()

    # --- цикл потока ---

    def run(self) -> None:  # pragma: no cover — сетевой цикл, логика в BookStore
        try:
            import websockets  # noqa: F401
        except ImportError:
            log.error("ws_feed: пакет websockets не установлен — поток выключен")
            return
        backoff = 1.0
        while not self._stop.is_set():
            if not self._desired:
                time.sleep(1.0)
                continue
            try:
                asyncio.run(self._session())
                backoff = 1.0
            except Exception as exc:
                log.warning("ws_feed: %s — reconnect через %.0fс", exc, backoff)
                self._check_outage()
                time.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _session(self) -> None:  # pragma: no cover
        import websockets

        self._resubscribe.clear()
        tokens = sorted(self._desired)
        async with websockets.connect(self._url, ping_interval=self._ping_interval,
                                      close_timeout=5) as ws:
            await ws.send(json.dumps({"type": "market", "assets_ids": tokens}))
            log.info("ws_feed: подписка на %d токенов", len(tokens))
            self._last_msg_ts = time.time()
            self._outage_reported = False
            while not self._stop.is_set() and not self._resubscribe.is_set():
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=self._staleness)
                except asyncio.TimeoutError:
                    self._check_outage()
                    continue
                self._last_msg_ts = time.time()
                self._outage_reported = False
                self._dispatch(raw)

    def _dispatch(self, raw) -> None:
        try:
            payload = json.loads(raw)
        except (TypeError, ValueError):
            return
        messages = payload if isinstance(payload, list) else [payload]
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            token = self.store.handle(msg)
            if token and self._on_update is not None:
                top = self.store.top(token)
                if top is not None:
                    try:
                        self._on_update(token, top)
                    except Exception:
                        log.exception("ws_feed: on_update callback")

    def _check_outage(self) -> None:
        """Gap-detect: поток мёртв дольше порога — сообщаем kill-switch один раз."""
        gap = time.time() - self._last_msg_ts if self._last_msg_ts else 0.0
        if gap >= self._staleness and not self._outage_reported:
            self._outage_reported = True
            log.error("ws_feed: нет данных %.0fс — котировки должны быть сняты", gap)
            if self._on_disconnect is not None:
                try:
                    self._on_disconnect(gap)
                except Exception:
                    log.exception("ws_feed: on_disconnect callback")
