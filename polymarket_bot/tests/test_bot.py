"""Офлайн-тесты бота: стратегия, скоринг, конфиг, учёт ставок. Сеть не нужна."""

import json
from datetime import datetime, timedelta, timezone

import pytest

from polymarket_bot.clob import BookQuote, compute_entry_price, round_to_tick, shares_for_stake
from polymarket_bot.config import BotConfig
from polymarket_bot.storage import BetLog
from polymarket_bot.strategy import find_candidates, plan_bets, stake_for


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
        "volume24hr": 5_000,
        "oneDayPriceChange": 0.0,
        "endDate": end.isoformat().replace("+00:00", "Z"),
        "negRisk": False,
        "orderPriceMinTickSize": 0.001,
        "orderMinSize": 5,
    }
    market.update(overrides)
    return market


def cfg(**kw) -> BotConfig:
    return BotConfig(**kw)


# --- отбор кандидатов ---

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
    planned = plan_bets(found, cfg(stake_usd=15, total_budget_usd=60, max_bets=100,
                                   stake_scaling=False))
    assert len(planned) == 4  # 60 / 15


# --- скоринг ---

def test_momentum_boosts_score():
    quiet = make_market(id="quiet", clobTokenIds=json.dumps(["a", "b"]))
    pumping = make_market(id="pump", oneDayPriceChange=0.003,
                          clobTokenIds=json.dumps(["c", "d"]))
    found = find_candidates([quiet, pumping], cfg())
    assert [c.market_id for c in found] == ["pump", "quiet"]
    pump = found[0]
    assert pump.score_parts["momentum"] == 1.0  # +0.003 к цене 0.005 = сильный сигнал


def test_momentum_sign_flips_for_no_outcome():
    # Дешёвый No: падение Yes = рост No = позитивный моментум.
    market = make_market(outcomePrices=json.dumps(["0.995", "0.005"]),
                         oneDayPriceChange=-0.003)
    found = find_candidates([market], cfg())
    assert found[0].outcome == "No"
    assert found[0].score_parts["momentum"] == 1.0


def test_min_score_filter():
    assert find_candidates([make_market()], cfg(min_score=0.99)) == []


def test_stake_scaling():
    c = find_candidates([make_market()], cfg())[0]
    fixed = stake_for(c, cfg(stake_scaling=False, stake_usd=15))
    scaled = stake_for(c, cfg(stake_scaling=True, stake_usd=15))
    assert fixed == 15
    assert 7.5 <= scaled <= 22.5  # 0.5x..1.5x


# --- цены и размеры ордеров ---

def test_round_to_tick_and_sizing():
    assert round_to_tick(0.0049, 0.001) == pytest.approx(0.005)
    assert shares_for_stake(15, 0.005, min_order_size=5) == 3000
    assert shares_for_stake(1, 0.5, min_order_size=5) == 5  # не меньше минимума биржи


def _candidate():
    return find_candidates([make_market()], cfg())[0]


def test_entry_price_taker_takes_ask():
    quote = BookQuote(best_ask=0.006, ask_depth=1000, best_bid=0.004, bid_depth=1000)
    price = compute_entry_price(quote, _candidate(), cfg(entry_mode="taker"))
    assert price == pytest.approx(0.006)


def test_entry_price_maker_sits_inside_spread():
    # Широкий спред: встаём на тик выше бида, а не платим ask.
    quote = BookQuote(best_ask=0.009, ask_depth=1000, best_bid=0.002, bid_depth=1000)
    price = compute_entry_price(quote, _candidate(), cfg(entry_mode="maker"))
    assert price == pytest.approx(0.003)
    # Узкий спред: не пересекаем ask (остаёмся maker).
    quote = BookQuote(best_ask=0.005, ask_depth=1000, best_bid=0.004, bid_depth=1000)
    price = compute_entry_price(quote, _candidate(), cfg(entry_mode="maker"))
    assert price == pytest.approx(0.004)


def test_entry_price_rejects_runaway_market():
    # Gamma говорила 0.005, а в стакане уже 0.08 — цена убежала, пропускаем.
    quote = BookQuote(best_ask=0.08, ask_depth=1000, best_bid=0.06, bid_depth=1000)
    assert compute_entry_price(quote, _candidate(), cfg()) is None


def test_entry_price_never_exceeds_max_price():
    quote = BookQuote(best_ask=0.02, ask_depth=1000, best_bid=0.018, bid_depth=1000)
    price = compute_entry_price(quote, _candidate(), cfg(entry_mode="taker"))
    assert price is None  # taker по 0.02 > max_price 0.01 — не покупаем дороже порога


# --- конфиг ---

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
    with pytest.raises(ValueError):
        BotConfig(entry_mode="yolo").validate()
    with pytest.raises(ValueError):
        BotConfig(take_profit_fraction=0).validate()


# --- журнал ставок ---

def test_bet_log_dedupe_and_budget(tmp_path):
    log = BetLog(tmp_path / "bets.json")
    c = _candidate()
    kwargs = dict(market_id=c.market_id, question=c.question, slug=c.slug,
                  outcome=c.outcome, token_id=c.token_id, price=0.005, size=3000)
    log.record(live=True, status="matched", **kwargs)
    log.record(live=False, status="dry-run", **kwargs)

    reloaded = BetLog(tmp_path / "bets.json")
    assert reloaded.live_token_ids() == {"token-yes"}  # dry-run дубли не блокирует
    assert reloaded.spent_usd() == pytest.approx(15.0)
    assert reloaded.spent_today_usd() == pytest.approx(15.0)
    assert reloaded.bets[0]["payout_if_win"] == 3000


def test_bet_log_canceled_orders_return_budget(tmp_path):
    log = BetLog(tmp_path / "bets.json")
    c = _candidate()
    log.record(market_id=c.market_id, question=c.question, slug=c.slug,
               outcome=c.outcome, token_id=c.token_id, price=0.005, size=3000,
               live=True, order_id="ord-1", status="live")
    assert log.spent_usd() == pytest.approx(15.0)
    assert len(log.open_orders()) == 1

    log.update_status("ord-1", "canceled")
    assert log.spent_usd() == 0.0  # снятый ордер размораживает бюджет
    assert log.open_orders() == []
    assert log.live_token_ids() == set()  # можно ставить на этот токен снова


def test_bet_log_sell_side_not_counted_as_spend(tmp_path):
    log = BetLog(tmp_path / "bets.json")
    log.record(market_id="1", question="q", slug="s", outcome="Yes", token_id="tok",
               price=0.05, size=1500, live=True, side="SELL", order_id="ord-2", status="live")
    assert log.spent_usd() == 0.0
    assert log.has_open_sell("tok")
    assert not log.has_open_sell("other")
