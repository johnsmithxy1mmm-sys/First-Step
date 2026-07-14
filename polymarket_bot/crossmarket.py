"""Strategy #2: cross-platform divergence scanner (Polymarket vs Kalshi).

The same outcome is quoted with 2-10 pp gaps across venues, especially at
news moments. This module ONLY finds and alerts — auto-trading is
deliberately absent: the main risk is not market but operational (an
"identical" event resolves differently due to rule differences), and only a
human can reconcile the resolution rules.

Kalshi is the first external venue: public REST, no auth. Other venues
(Betfair etc.) plug in via an ExternalVenue implementation.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

import httpx
from pydantic import BaseModel

from .config import BotConfig
from .http_client import get_with_backoff, make_client
from .models import Market

log = logging.getLogger(__name__)

_STOPWORDS = frozenset(
    "will the a an of in on at by to be is are before after or and for during".split()
)
_CLEAN = re.compile(r"[^a-z0-9 ]+")


def title_tokens(title: str) -> frozenset[str]:
    words = _CLEAN.sub(" ", title.lower()).split()
    return frozenset(w for w in words if w not in _STOPWORDS and len(w) > 1)


def similarity(a: str, b: str) -> float:
    """Jaccard over the meaningful words of the titles."""
    ta, tb = title_tokens(a), title_tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class VenueMarket(BaseModel):
    venue: str
    title: str
    yes_price: float            # mid of Yes probability, 0..1
    url: str = ""


class ExternalVenue(Protocol):
    name: str
    def fetch_markets(self) -> list[VenueMarket]: ...


class KalshiVenue:
    """Public Kalshi market data (no auth)."""

    name = "kalshi"
    HOST = "https://api.elections.kalshi.com/trade-api/v2"

    def __init__(self, timeout_sec: float = 30.0, max_pages: int = 5):
        self._client = make_client(timeout_sec)
        self._max_pages = max_pages

    def fetch_markets(self) -> list[VenueMarket]:
        out: list[VenueMarket] = []
        cursor = None
        for _ in range(self._max_pages):
            params: dict = {"status": "open", "limit": 200}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = get_with_backoff(self._client, f"{self.HOST}/markets",
                                        params=params, max_retries=2)
            except httpx.HTTPError as exc:
                log.warning("kalshi: %s", exc)
                break
            data = resp.json()
            for m in data.get("markets") or []:
                bid = float(m.get("yes_bid") or 0) / 100.0
                ask = float(m.get("yes_ask") or 0) / 100.0
                if bid <= 0 or ask <= 0:
                    continue
                out.append(VenueMarket(
                    venue=self.name,
                    title=m.get("title") or "",
                    yes_price=(bid + ask) / 2,
                    url=f"https://kalshi.com/markets/{m.get('ticker', '')}",
                ))
            cursor = data.get("cursor")
            if not cursor:
                break
        log.info("kalshi: fetched %d markets", len(out))
        return out


class Divergence(BaseModel):
    poly_market: Market
    venue_market: VenueMarket
    poly_price: float
    venue_price: float
    similarity: float

    @property
    def gap(self) -> float:
        return abs(self.poly_price - self.venue_price)

    def describe(self) -> str:
        cheaper = "Polymarket" if self.poly_price < self.venue_price else self.venue_market.venue
        return (f"DIVERGENCE {self.gap * 100:.1f} pp (cheaper on {cheaper}, "
                f"sim={self.similarity:.2f}):\n"
                f"  Polymarket {self.poly_price:.3f}: {self.poly_market.question[:80]}\n"
                f"  {self.venue_market.venue} {self.venue_price:.3f}: "
                f"{self.venue_market.title[:80]}\n"
                f"  CHECK BOTH VENUES' RESOLUTION RULES BEFORE ENTERING")


class CrossMarketScanner:
    def __init__(self, cfg: BotConfig, venues: list[ExternalVenue] | None = None):
        self._cfg = cfg.crossmarket
        self._venues = venues if venues is not None else [KalshiVenue()]

    def find_divergences(self, poly_markets: list[Market],
                         venue_markets: list[VenueMarket]) -> list[Divergence]:
        c = self._cfg
        liquid = [m for m in poly_markets
                  if m.volume_24h_usd >= c.min_volume_24h_usd
                  and m.outcome_prices and not m.closed]
        out: list[Divergence] = []
        for vm in venue_markets:
            best, best_sim = None, 0.0
            for pm in liquid:
                sim = similarity(pm.question, vm.title)
                if sim > best_sim:
                    best, best_sim = pm, sim
            if best is None or best_sim < c.min_similarity:
                continue
            poly_price = best.outcome_prices[0]
            gap = abs(poly_price - vm.yes_price)
            if gap >= c.min_divergence:
                out.append(Divergence(
                    poly_market=best, venue_market=vm,
                    poly_price=poly_price, venue_price=vm.yes_price,
                    similarity=best_sim,
                ))
        out.sort(key=lambda d: d.gap, reverse=True)
        return out[: c.max_alerts_per_cycle]

    def cycle(self, poly_markets: list[Market]) -> list[Divergence]:
        if not self._cfg.enabled:
            return []
        divergences: list[Divergence] = []
        for venue in self._venues:
            try:
                venue_markets = venue.fetch_markets()
            except Exception as exc:
                log.warning("venue %s: %s", venue.name, exc)
                continue
            divergences.extend(self.find_divergences(poly_markets, venue_markets))
        for d in divergences:
            log.info(d.describe())
        return divergences
