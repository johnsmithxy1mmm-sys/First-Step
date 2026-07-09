"""Офлайн-тесты бота: стратегия, конфиг, учёт ставок. Сеть не нужна."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from polymarket_bot.clob import round_to_tick, shares_for_stake
from polymarket_bot.config import BotConfig
from polymarket_bot.storage import BetLog
from polymarket_bot.strategy import find_candidates, plan_bets


def make_market(**overrides) -> dict:
    end = datetime.now(timezone.utc) + timedelta(days=30)
    market = {
        "id": "1",
        "question": "Will bitcoin hit $500k this year?",
        "slug": "btc-500k",
        "enableOrderBook": True,
        "outcomes": json.dumps(["Yes", "No"]),
        "outcomePrices": json.dumps(["0.005", "0.995"]),
        "clobTokenIds": json.dumps(["token-yes", "token-no"]),
        "liquidityNum": 50_000,
        "volumeNum": 200_000,
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "negRisk": False,
        "orderPriceMinTickSize": 0.001,
        "orderMinSize": 5,
    }
    market.update(overrides)
    return market


def cfg(**kw) -> BotConfig:
    return BotConfig(**kw)


def test_finds_cheap_outcome():
    found = find_candidates([make_market()], cfg())
    assert len(found) == 1
    c = found[0]
    assert c.outcome == "Yes"
    assert c.token_id == "token-yes"
    assert c.price == 0.005
    assert c.payout_multiple == pytest.approx(200)


def test_skips_expensive_and_dead_prices():
    market = make_market(outcomePrices=json.dumps(["0.05", "0.0"]))
    assert find_candidates([market], cfg()) == []


def test_filters_by_liquidity_and_volume():
    assert find_candidates([make_market(liquidityNum=10)], cfg()) == []
    assert find_candidates([make_market(volumeNum=10)], cfg()) == []


def test_filters_by_resolution_window():
    soon = datetime.now(timezone.utc) + timedelta(hours=2)
    far = datetime.now(timezone.utc) + timedelta(days=365)
    assert find_candidates([make_market(endDate=soon.isoformat())], cfg()) == []
    assert find_candidates([make_market(endDate=far.isoformat())], cfg()) == []


def test_keyword_filters():
    include = cfg(include_keywords=["ethereum"])
    exclude = cfg(exclude_keywords=["bitcoin"])
    assert find_candidates([make_market()], include) == []
    assert find_candidates([make_market()], exclude) == []
    assert len(find_candidates([make_market()], cfg(include_keywords=["bitcoin"]))) == 1


def test_skips_already_bet_tokens():
    assert find_candidates([make_market()], cfg(), skip_token_ids={"token-yes"}) == []


def test_skips_markets_without_orderbook():
    assert find_candidates([make_market(enableOrderBook=False)], cfg()) == []


def test_max_bets_per_market():
    market = make_market(outcomePrices=json.dumps(["0.005", "0.008"]))
    assert len(find_candidates([market], cfg())) == 1
    assert len(find_candidates([market], cfg(max_bets_per_market=2))) == 2


def test_plan_respects_budget():
    markets = [make_market(id=str(i), clobTokenIds=json.dumps([f"t{i}", f"n{i}"]))
               for i in range(20)]
    found = find_candidates(markets, cfg())
    planned = plan_bets(found, cfg(stake_usd=15, total_budget_usd=60, max_bets=100))
    assert len(planned) == 4  # 60 / 15


def test_candidates_sorted_by_liquidity():
    low = make_market(id="low", liquidityNum=2_000, clobTokenIds=json.dumps(["a", "b"]))
    high = make_market(id="high", liquidityNum=90_000, clobTokenIds=json.dumps(["c", "d"]))
    found = find_candidates([low, high], cfg())
    assert [c.market_id for c in found] == ["high", "low"]


def test_round_to_tick_and_sizing():
    assert round_to_tick(0.0049, 0.001) == pytest.approx(0.005)
    assert shares_for_stake(15, 0.005, min_order_size=5) == 3000
    assert shares_for_stake(1, 0.5, min_order_size=5) == 5  # не меньше минимума биржи


def test_config_validation_and_overrides(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"max_price": 0.02, "stake_usd": 10}), encoding="utf-8")
    loaded = BotConfig.load(path, stake_usd=20)
    assert loaded.max_price == 0.02
    assert loaded.stake_usd == 20  # CLI важнее файла

    path.write_text(json.dumps({"nope": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        BotConfig.load(path)
    with pytest.raises(ValueError):
        BotConfig(max_price=1.5).validate()


def test_bet_log_dedupe_and_budget(tmp_path):
    log = BetLog(tmp_path / "bets.json")
    candidate = find_candidates([make_market()], cfg())[0]
    log.record(candidate=candidate, price=0.005, size=3000, live=True, status="matched")
    log.record(candidate=candidate, price=0.005, size=3000, live=False, status="dry-run")

    reloaded = BetLog(tmp_path / "bets.json")
    assert reloaded.live_token_ids() == {"token-yes"}  # dry-run дубли не блокирует
    assert reloaded.spent_usd() == pytest.approx(15.0)
    assert reloaded.bets[0]["payout_if_win"] == 3000
