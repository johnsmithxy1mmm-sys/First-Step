"""Тесты автопилота и арбитража на замоканной сети."""

import json
from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from polymarket_bot import autopilot
from polymarket_bot.arb import find_arbs
from polymarket_bot.clob import BookQuote
from polymarket_bot.config import BotConfig
from polymarket_bot.storage import BetLog

from .test_bot import make_market


def make_event(asks: list[float], neg_risk: bool = True, **overrides) -> dict:
    event = {
        "id": "ev1",
        "title": "Who will win the election?",
        "negRisk": neg_risk,
        "markets": [
            {
                "question": f"Candidate {i}",
                "closed": False,
                "enableOrderBook": True,
                "bestAsk": ask,
                "clobTokenIds": json.dumps([f"yes-{i}", f"no-{i}"]),
                "orderPriceMinTickSize": 0.001,
                "orderMinSize": 5,
            }
            for i, ask in enumerate(asks)
        ],
    }
    event.update(overrides)
    return event


# --- арбитраж ---

def test_arb_detected_when_set_costs_less_than_dollar():
    arbs = find_arbs([make_event([0.40, 0.35, 0.20])], BotConfig())  # сумма 0.95
    assert len(arbs) == 1
    assert arbs[0].edge == pytest.approx(0.05)
    assert len(arbs[0].legs) == 3


def test_arb_ignores_fair_and_non_negrisk_events():
    fair = make_event([0.50, 0.49])              # сумма 0.99 < порога 0.98? нет: 1-0.02=0.98, 0.99 >= 0.98
    not_negrisk = make_event([0.40, 0.35], neg_risk=False)
    assert find_arbs([fair, not_negrisk], BotConfig(arb_min_edge=0.02)) == []


def test_arb_sets_sizing_respects_min_order():
    arb = find_arbs([make_event([0.40, 0.35, 0.20])], BotConfig())[0]
    assert arb.sets_for_stake(95.0) == 100   # 95 / 0.95
    assert arb.sets_for_stake(2.0) == 0      # 2 комплекта < min_order_size 5


# --- бюджет автопилота ---

def test_remaining_budget_daily_cap(tmp_path):
    cfg = BotConfig(total_budget_usd=1000, daily_budget_usd=100)
    log = BetLog(tmp_path / "bets.json")
    log.record(market_id="1", question="q", slug="s", outcome="Yes", token_id="t1",
               price=0.01, size=6000, live=True, status="matched")  # $60 сегодня
    assert autopilot.remaining_budget(cfg, log) == pytest.approx(40.0)


# --- сквозной dry-run цикл ---

def _fake_positions():
    return [
        {   # выигрыш готов к выводу
            "title": "Won market", "outcome": "Yes", "asset": "tok-won",
            "redeemable": True, "currentValue": 1500.0, "size": 1500,
            "avgPrice": 0.01, "curPrice": 1.0, "cashPnl": 1485.0,
        },
        {   # 12x от входа — пора фиксировать
            "title": "Pumped market", "outcome": "Yes", "asset": "tok-pump",
            "redeemable": False, "size": 3000,
            "avgPrice": 0.005, "curPrice": 0.06, "cashPnl": 165.0,
        },
        {   # обычная позиция, трогать не надо
            "title": "Flat market", "outcome": "Yes", "asset": "tok-flat",
            "redeemable": False, "size": 3000,
            "avgPrice": 0.005, "curPrice": 0.006, "cashPnl": 3.0,
        },
    ]


def test_full_dry_run_cycle(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYMARKET_FUNDER", "0xdeadbeef")
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    cfg = BotConfig(stake_usd=15, total_budget_usd=60, stake_scaling=False)
    log = BetLog(tmp_path / "bets.json")

    markets = [make_market(id=str(i), clobTokenIds=json.dumps([f"t{i}", f"n{i}"]))
               for i in range(6)]
    quote = BookQuote(best_ask=0.006, ask_depth=100000, best_bid=0.004, bid_depth=100000)

    with mock.patch.object(autopilot, "iter_active_markets", return_value=iter(markets)), \
         mock.patch.object(autopilot, "iter_active_events",
                           return_value=iter([make_event([0.40, 0.35, 0.20])])), \
         mock.patch.object(autopilot.clob, "get_quote", return_value=quote), \
         mock.patch.object(autopilot, "fetch_positions", return_value=_fake_positions()), \
         mock.patch.object(autopilot, "market_by_token", return_value=make_market()):
        report = autopilot.run_cycle(cfg, log, trader=None)

    assert report.placed == 4                    # бюджет 60 / 15
    assert report.spent == pytest.approx(4 * 0.005 * 3000)  # maker-цена 0.005
    assert report.arbs_found == 1
    assert len(report.redeemable) == 1
    assert "Won market" in report.redeemable[0]
    # dry-run: ничего live не записано, но план сохранён
    assert log.live_token_ids() == set()
    assert len(log.bets) == 4


def test_cycle_survives_step_failures(tmp_path, monkeypatch):
    """Упавший шаг не убивает цикл — автономность важнее одного скана."""
    monkeypatch.delenv("POLYMARKET_FUNDER", raising=False)
    monkeypatch.delenv("POLYMARKET_ADDRESS", raising=False)
    cfg = BotConfig()
    log = BetLog(tmp_path / "bets.json")
    with mock.patch.object(autopilot, "iter_active_markets", side_effect=RuntimeError("api down")), \
         mock.patch.object(autopilot, "iter_active_events", side_effect=RuntimeError("api down")):
        report = autopilot.run_cycle(cfg, log, trader=None)
    assert any("place_bets" in e for e in report.errors)
    assert any("scan_arbs" in e for e in report.errors)


def test_take_profit_places_sell_with_live_trader(tmp_path, monkeypatch):
    monkeypatch.setenv("POLYMARKET_FUNDER", "0xdeadbeef")
    cfg = BotConfig(take_profit_multiple=10, take_profit_fraction=0.5)
    log = BetLog(tmp_path / "bets.json")
    report = autopilot.CycleReport()

    trader = mock.Mock()
    trader.sell_limit.return_value = {"orderID": "sell-1", "status": "live"}
    quote = BookQuote(best_ask=0.065, ask_depth=5000, best_bid=0.06, bid_depth=5000)

    with mock.patch.object(autopilot, "fetch_positions", return_value=_fake_positions()), \
         mock.patch.object(autopilot, "market_by_token", return_value=make_market()), \
         mock.patch.object(autopilot.clob, "get_quote", return_value=quote):
        autopilot.manage_positions(cfg, log, trader, session=mock.Mock(), report=report)

    trader.sell_limit.assert_called_once()
    token_id, price, size = trader.sell_limit.call_args[0]
    assert token_id == "tok-pump"
    assert price == pytest.approx(0.06)
    assert size == 1500  # половина позиции 3000
    assert report.take_profits == 1
    assert log.has_open_sell("tok-pump")

    # Повторный цикл: продажа уже висит, дубля не будет.
    with mock.patch.object(autopilot, "fetch_positions", return_value=_fake_positions()), \
         mock.patch.object(autopilot, "market_by_token", return_value=make_market()), \
         mock.patch.object(autopilot.clob, "get_quote", return_value=quote):
        autopilot.manage_positions(cfg, log, trader, session=mock.Mock(), report=report)
    trader.sell_limit.assert_called_once()


def test_cancel_stale_orders(tmp_path):
    cfg = BotConfig(max_order_age_hours=0)  # всё старше «сейчас» — зависшее
    log = BetLog(tmp_path / "bets.json")
    log.record(market_id="1", question="stale", slug="s", outcome="Yes", token_id="t1",
               price=0.005, size=3000, live=True, order_id="ord-stale", status="live")
    # Подделываем время записи: ордер висит 13 часов.
    bets = log.bets
    old_ts = (datetime.now(timezone.utc) - timedelta(hours=13)).isoformat(timespec="seconds")
    log._bets[0]["ts"] = old_ts
    log._save()

    trader = mock.Mock()
    report = autopilot.CycleReport()
    autopilot.cancel_stale_orders(BotConfig(max_order_age_hours=12), log, trader, report)
    trader.cancel.assert_called_once_with("ord-stale")
    assert report.canceled == 1
    assert log.spent_usd() == 0.0
