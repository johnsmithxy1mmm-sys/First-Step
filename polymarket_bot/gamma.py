"""Работа с Gamma API Polymarket: выгрузка активных рынков."""

from __future__ import annotations

import json
import time
from typing import Iterator

import requests

from .config import BotConfig

PAGE_SIZE = 500
MAX_PAGES = 60  # предохранитель: 30 000 рынков хватит с запасом


def iter_active_markets(cfg: BotConfig, session: requests.Session | None = None) -> Iterator[dict]:
    """Постранично отдаёт все активные незакрытые рынки."""
    session = session or requests.Session()
    offset = 0
    for _ in range(MAX_PAGES):
        resp = session.get(
            f"{cfg.gamma_host}/markets",
            params={
                "active": "true",
                "closed": "false",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            return
        yield from batch
        if len(batch) < PAGE_SIZE:
            return
        offset += PAGE_SIZE
        time.sleep(cfg.request_delay_sec)


def iter_active_events(cfg: BotConfig, session: requests.Session | None = None) -> Iterator[dict]:
    """Постранично отдаёт активные события (группы рынков) — нужны для арбитража."""
    session = session or requests.Session()
    offset = 0
    for _ in range(MAX_PAGES):
        resp = session.get(
            f"{cfg.gamma_host}/events",
            params={
                "active": "true",
                "closed": "false",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            timeout=30,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            return
        yield from batch
        if len(batch) < PAGE_SIZE:
            return
        offset += PAGE_SIZE
        time.sleep(cfg.request_delay_sec)


def market_by_token(cfg: BotConfig, token_id: str,
                    session: requests.Session | None = None) -> dict | None:
    """Находит рынок по ID токена CLOB (нужно при продаже позиции: тик, neg-risk)."""
    session = session or requests.Session()
    resp = session.get(
        f"{cfg.gamma_host}/markets",
        params={"clob_token_ids": token_id},
        timeout=30,
    )
    if resp.status_code != 200:
        return None
    markets = resp.json()
    return markets[0] if markets else None


def parse_json_list(value) -> list:
    """Gamma отдаёт outcomes/outcomePrices/clobTokenIds строками с JSON внутри."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def market_float(market: dict, *keys: str) -> float:
    """Достаёт число из рынка, пробуя несколько имён полей (liquidityNum/liquidity и т.п.)."""
    for key in keys:
        value = market.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return 0.0
