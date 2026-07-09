"""Общие фикстуры: фабрики рынков/кандидатов, конфиг с временной БД."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from polymarket_bot.config import BotConfig
from polymarket_bot.ledger import Ledger
from polymarket_bot.models import BookLevel, Candidate, Market, OrderBook


def make_market(**overrides) -> Market:
    end = datetime.now(timezone.utc) + timedelta(days=30)
    defaults = dict(
        id="m1",
        question="Will a magnitude 8.0 earthquake strike this year?",
        slug="quake-m8",
        description="Resolves YES if USGS reports a magnitude 8.0+ earthquake "
                    "anywhere in the world before the end date, per official data.",
        outcomes=["Yes", "No"],
        outcome_prices=[0.01, 0.99],
        clob_token_ids=["tok-yes-m1", "tok-no-m1"],
        volume_usd=500_000.0,
        volume_24h_usd=20_000.0,
        liquidity_usd=50_000.0,
        end_date=end,
        resolution_source="https://usgs.gov",
        enable_order_book=True,
        tick_size=0.001,
        min_order_size=5.0,
    )
    defaults.update(overrides)
    return Market(**defaults)


def make_candidate(market: Market | None = None, outcome_index: int = 0,
                   **market_overrides) -> Candidate:
    m = market or make_market(**market_overrides)
    return Candidate(
        market=m,
        outcome_index=outcome_index,
        token_id=m.clob_token_ids[outcome_index],
        p_mkt=m.outcome_prices[outcome_index],
    )


def make_book(best_bid: float = 0.009, best_ask: float = 0.012,
              depth: float = 100_000.0) -> OrderBook:
    return OrderBook(
        bids=[BookLevel(price=best_bid, size=depth),
              BookLevel(price=best_bid * 0.7, size=depth)],
        asks=[BookLevel(price=best_ask, size=depth),
              BookLevel(price=best_ask * 1.5, size=depth)],
    )


def gamma_raw_market(**overrides) -> dict:
    end = datetime.now(timezone.utc) + timedelta(days=30)
    raw = {
        "id": "g1",
        "question": "Will X happen?",
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.02", "0.98"]),
        "clobTokenIds": json.dumps(["t-yes", "t-no"]),
        "volume24hr": 10_000,
        "volumeNum": 100_000,
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "enableOrderBook": True,
        "orderPriceMinTickSize": 0.001,
        "orderMinSize": 5,
        "resolutionSource": "official source",
    }
    raw.update(overrides)
    return raw


@pytest.fixture
def cfg(tmp_path) -> BotConfig:
    c = BotConfig()
    c.runtime.db_path = str(tmp_path / "ledger.sqlite")
    c.runtime.log_path = str(tmp_path / "bot.jsonl")
    c.runtime.llm_cache_path = str(tmp_path / "llm_cache.json")
    return c


@pytest.fixture
def ledger(cfg) -> Ledger:
    led = Ledger(cfg.runtime.db_path)
    yield led
    led.close()
