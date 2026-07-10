"""Сателлит (20% капитала, по умолчанию ВЫКЛЮЧЕН): T-10s TA на 5-мин BTC.

НЕ латентный арбитраж (выеден sub-100ms HFT и подавлен динамическими taker
fees). Вместо этого: технический скоринг направления за 10-30 сек до
закрытия окна; вход только при P(model) − implied − taker_fee > edge_threshold.

Слаги детерминированы: btc-updown-5m-{unix_ts}, ts кратен 300 — рынок
вычисляется по часам, а не ищется. Сайзинг: quarter-Kelly с жёстким кэпом.
Вход агрессивной ногой (FOK): маркет-ордеров на платформе нет.
"""

from __future__ import annotations

import logging
import math
import time

import httpx
from pydantic import BaseModel

from .clob import ClobReader, Trader, round_to_tick
from .config import BotConfig
from .fees import FeeModel
from .http_client import get_with_backoff, make_client
from .ledger import Ledger
from .models import Market, simple_estimate
from .portfolio import kelly_fraction

log = logging.getLogger(__name__)


def window_ts(now: float, window_sec: int = 300) -> int:
    """Начало текущего 5-минутного окна (unix ts, кратен 300)."""
    return int(now // window_sec) * window_sec


def market_slug(now: float, template: str, window_sec: int = 300) -> str:
    return template.format(ts=window_ts(now, window_sec))


def seconds_to_close(now: float, window_sec: int = 300) -> float:
    return window_ts(now, window_sec) + window_sec - now


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0


def ta_p_up(candles: list[Candle]) -> tuple[float, float]:
    """Вероятность закрытия окна вверх по минутным свечам. -> (p_up, confidence).

    Прозрачные компоненты: momentum последних минут, положение цены
    относительно короткой EMA, доля бычьих свечей. Логистическое смешение.
    """
    if len(candles) < 6:
        return 0.5, 0.0
    closes = [c.close for c in candles]
    last = closes[-1]

    # Momentum: доходность за 3 и за 1 минуту, нормированная волатильностью.
    rets = [closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))]
    vol = max((sum(r * r for r in rets) / len(rets)) ** 0.5, 1e-6)
    mom3 = (last / closes[-4] - 1.0) / (vol * math.sqrt(3))
    mom1 = rets[-1] / vol

    # EMA(5): положение цены относительно локального тренда.
    ema = closes[0]
    alpha = 2 / (5 + 1)
    for price in closes[1:]:
        ema = alpha * price + (1 - alpha) * ema
    ema_dev = (last - ema) / max(ema * vol, 1e-9)

    # Доля бычьих минут из последних шести.
    bull_frac = sum(1 for c in candles[-6:] if c.close > c.open) / 6.0

    z = 0.5 * mom1 + 0.7 * mom3 + 0.4 * ema_dev + 1.2 * (bull_frac - 0.5)
    p_up = 1.0 / (1.0 + math.exp(-z))
    confidence = min(abs(z) / 3.0, 1.0)
    return p_up, confidence


class BTC5mSatellite:
    def __init__(self, cfg: BotConfig, ledger: Ledger, clob: ClobReader,
                 trader: Trader | None, mode: str,
                 http: httpx.Client | None = None):
        self._cfg = cfg.satellite
        self._risk = cfg.risk
        self._fees = FeeModel(cfg.fees)
        self._gamma_host = cfg.runtime.gamma_host
        self._ledger = ledger
        self._clob = clob
        self._trader = trader
        self._mode = mode
        self._http = http or make_client(15.0)
        self._traded_windows: set[int] = set()

    def fetch_candles(self) -> list[Candle]:
        try:
            resp = get_with_backoff(self._http, self._cfg.candles_url, max_retries=1)
        except httpx.HTTPError as exc:
            log.warning("satellite: свечи недоступны: %s", exc)
            return []
        out = []
        for k in resp.json() or []:
            try:
                out.append(Candle(open=float(k[1]), high=float(k[2]),
                                  low=float(k[3]), close=float(k[4]),
                                  volume=float(k[5])))
            except (IndexError, TypeError, ValueError):
                continue
        return out

    def fetch_market(self, slug: str) -> Market | None:
        try:
            resp = get_with_backoff(
                self._http, f"{self._gamma_host}/markets",
                params={"slug": slug}, max_retries=1)
        except httpx.HTTPError:
            return None
        markets = resp.json() or []
        return Market.from_gamma(markets[0]) if markets else None

    def decide(self, p_up: float, implied_up: float) -> tuple[int, float] | None:
        """(outcome_index, p_model) для входа или None. Edge считается ПОСЛЕ fee."""
        fee = self._fees.taker_fee("crypto")
        edge_up = p_up - implied_up - fee
        edge_down = (1.0 - p_up) - (1.0 - implied_up) - fee
        if edge_up >= self._cfg.edge_threshold and edge_up >= edge_down:
            return 0, p_up
        if edge_down >= self._cfg.edge_threshold:
            return 1, 1.0 - p_up
        return None

    def bet_size(self, p_model: float, price: float) -> float:
        f_star = kelly_fraction(p_model, price)
        stake = self._cfg.kelly_fraction * f_star * self._risk.max_global_exposure_usd
        return min(stake, self._cfg.max_bet_usd)

    def cycle(self, now: float | None = None) -> None:
        if not self._cfg.enabled:
            return
        now = now if now is not None else time.time()
        ttc = seconds_to_close(now, self._cfg.window_sec)
        if not self._cfg.entry_to_sec <= ttc <= self._cfg.entry_from_sec:
            return
        window = window_ts(now, self._cfg.window_sec)
        if window in self._traded_windows:
            return

        candles = self.fetch_candles()
        p_up, confidence = ta_p_up(candles)
        if confidence <= 0:
            return
        slug = market_slug(now, self._cfg.slug_template, self._cfg.window_sec)
        market = self.fetch_market(slug)
        if market is None or len(market.clob_token_ids) < 2:
            log.debug("satellite: рынок %s не найден", slug)
            return

        book = self._clob.order_book(market.clob_token_ids[0])
        if book is None or book.best_ask <= 0:
            return
        implied_up = book.best_ask
        decision = self.decide(p_up, implied_up)
        if decision is None:
            return
        idx, p_model = decision
        token = market.clob_token_ids[idx]
        side_book = book if idx == 0 else self._clob.order_book(token)
        if side_book is None or side_book.best_ask <= 0:
            return
        price = round_to_tick(side_book.best_ask, market.tick_size)
        stake = self.bet_size(p_model, price)
        size = float(math.floor(stake / max(price, 1e-9)))
        if size < market.min_order_size:
            return

        self._traded_windows.add(window)
        order_id = None
        if self._trader is not None:
            try:
                resp = self._trader.buy_limit(token, price, size,
                                              neg_risk=market.neg_risk,
                                              order_type="FOK")
                order_id = (resp or {}).get("orderID")
            except Exception as exc:
                log.error("satellite: FOK не исполнился: %s", exc)
                return
        self._ledger.record_trade(
            mode=self._mode, estimate=simple_estimate(market, idx, price),
            category="crypto", side="BUY", price=price, size=size,
            order_id=order_id,
            status="filled" if self._trader else f"{self._mode}-filled",
            strategy="btc_5m",
        )
        log.info("satellite: вход %s p_model=%.3f implied=%.3f %.3f x %.0f (%s)",
                 "UP" if idx == 0 else "DOWN", p_model, implied_up, price, size, slug)
