"""Gamma Markets API: метаданные рынков и событий."""

from __future__ import annotations

import logging
from typing import Iterator

import httpx

from .config import BotConfig
from .http_client import get_with_backoff, make_client
from .models import Market

log = logging.getLogger(__name__)

PAGE_SIZE = 500
MAX_PAGES = 80


class GammaClient:
    def __init__(self, cfg: BotConfig, client: httpx.Client | None = None):
        self._cfg = cfg
        self._client = client or make_client(cfg.runtime.request_timeout_sec)

    def _paginate(self, path: str, params: dict) -> Iterator[dict]:
        offset = 0
        for _ in range(MAX_PAGES):
            page_params = {**params, "limit": PAGE_SIZE, "offset": offset}
            resp = get_with_backoff(
                self._client,
                f"{self._cfg.runtime.gamma_host}{path}",
                params=page_params,
                max_retries=self._cfg.runtime.max_retries,
            )
            batch = resp.json()
            if not batch:
                return
            yield from batch
            if len(batch) < PAGE_SIZE:
                return
            offset += PAGE_SIZE

    def fetch_active_markets(self) -> list[Market]:
        """Активные рынки через /events — контекст события нужен когерентности."""
        markets: list[Market] = []
        seen: set[str] = set()
        for event in self._paginate("/events", {"active": "true", "closed": "false"}):
            for raw in event.get("markets") or []:
                market = Market.from_gamma(raw, event)
                if market is not None and market.id not in seen:
                    seen.add(market.id)
                    markets.append(market)
        log.info("gamma: активных рынков %d", len(markets))
        return markets

    def fetch_closed_markets(self, max_markets: int) -> list[Market]:
        """Закрытые рынки для бэктеста калибровки."""
        markets: list[Market] = []
        for raw in self._paginate("/markets", {"closed": "true", "order": "endDate", "ascending": "false"}):
            market = Market.from_gamma(raw)
            if market is None or not market.closed:
                continue
            markets.append(market)
            if len(markets) >= max_markets:
                break
        log.info("gamma: закрытых рынков для бэктеста %d", len(markets))
        return markets
