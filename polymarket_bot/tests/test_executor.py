"""Исполнитель: идемпотентность, maker-цены, edge-cap, дробление, dry-run fill."""

from unittest import mock

import pytest

from polymarket_bot.estimator.ensemble import combine
from polymarket_bot.executor import Executor
from polymarket_bot.models import Signal, TradePlan

from .conftest import make_book, make_candidate


def make_plan(size_usd=50.0, cap=0.02, **overrides) -> TradePlan:
    c = make_candidate(**overrides)
    est = combine(c, [Signal(name="s", p_est=0.04, confidence=1e9)], 1e-9)
    return TradePlan(estimate=est, category="nature", size_usd=size_usd,
                     limit_price_cap=cap)


def make_executor(cfg, ledger, book=None, trader=None) -> Executor:
    clob = mock.Mock()
    clob.order_book.return_value = book if book is not None else make_book()
    return Executor(cfg, ledger, clob, trader, "dry-run" if trader is None else "live")


# --- dry-run исполнение ---

def test_dry_run_fills_at_maker_price_and_records(cfg, ledger):
    ex = make_executor(cfg, ledger, book=make_book(best_bid=0.009, best_ask=0.012))
    plan = make_plan(size_usd=50.0, cap=0.02)
    result = ex.execute(plan)
    assert result.status == "filled"
    # ask (0.012) <= cap (0.02): встаём на тик ниже ask.
    assert result.avg_price == pytest.approx(0.011)
    positions = ledger.open_positions("dry-run")
    assert len(positions) == 1
    assert positions[0].size == result.filled_size


def test_maker_price_never_crosses_ask(cfg, ledger):
    book = make_book(best_bid=0.010, best_ask=0.011)  # спред в один тик
    ex = make_executor(cfg, ledger, book=book)
    result = ex.execute(make_plan(cap=0.05))
    assert result.status == "filled"
    assert result.avg_price <= book.best_ask - 0.001 + 1e-9  # maker, не taker


def test_edge_cap_limits_price(cfg, ledger):
    # Книга дорогая: bid 0.030, ask 0.035, а edge исчезает выше 0.02.
    ex = make_executor(cfg, ledger, book=make_book(best_bid=0.030, best_ask=0.035))
    result = ex.execute(make_plan(cap=0.02))
    # Цена зажата кэпом 0.02 — бид ниже рынка, честный resting-ордер.
    assert result.status == "filled"
    assert result.avg_price <= 0.02 + 1e-9


def test_order_splitting_into_children(cfg, ledger):
    cfg.executor.max_child_order_usd = 20.0
    clob = mock.Mock()
    clob.order_book.return_value = make_book()
    ex = Executor(cfg, ledger, clob, None, "dry-run")
    result = ex.execute(make_plan(size_usd=50.0))
    assert result.status == "filled"
    # 50 / 20 -> 3 ребёнка -> 3 запроса книги (по одному на ребёнка).
    assert clob.order_book.call_count == 3


# --- идемпотентность ---

def test_idempotency_skips_existing_position(cfg, ledger):
    ex = make_executor(cfg, ledger)
    plan = make_plan()
    assert ex.execute(plan).status == "filled"
    second = ex.execute(plan)
    assert second.status == "skipped"
    assert "idempotency" in second.detail
    assert len(ledger.open_positions("dry-run")) == 1  # дубля нет


def test_idempotency_checks_api_positions_in_live(cfg, ledger):
    trader = mock.Mock()
    plan = make_plan()
    trader.api_positions.return_value = [{"asset": plan.token_id, "size": 100}]
    trader.open_orders.return_value = []
    ex = make_executor(cfg, ledger, trader=trader)
    assert ex.execute(plan).status == "skipped"


def test_reconcile_failure_is_fail_safe(cfg, ledger):
    trader = mock.Mock()
    trader.api_positions.side_effect = RuntimeError("api down")
    ex = make_executor(cfg, ledger, trader=trader)
    # Сверка не удалась -> считаем, что позиция есть, ордер не шлём.
    assert ex.execute(make_plan()).status == "skipped"


# --- live-путь с моками ---

def test_live_reprices_then_gives_up(cfg, ledger):
    cfg.executor.max_reprices = 2
    cfg.executor.fill_timeout_sec = 0.01
    cfg.executor.poll_interval_sec = 0.01
    trader = mock.Mock()
    trader.api_positions.return_value = []
    trader.open_orders.return_value = []
    trader.buy_limit.return_value = {"orderID": "o1"}
    trader.order_status.return_value = {"status": "live", "size_matched": 0.0}
    ex = make_executor(cfg, ledger, trader=trader)

    result = ex.execute(make_plan(size_usd=50.0))
    assert result.status == "canceled"
    assert trader.buy_limit.call_count == cfg.executor.max_reprices + 1
    assert trader.cancel.call_count == cfg.executor.max_reprices + 1
    assert ledger.open_positions("live") == []


def test_live_partial_fill_recorded(cfg, ledger):
    cfg.executor.fill_timeout_sec = 0.01
    cfg.executor.poll_interval_sec = 0.01
    trader = mock.Mock()
    trader.api_positions.return_value = []
    trader.open_orders.return_value = []
    trader.buy_limit.return_value = {"orderID": "o1"}
    trader.order_status.return_value = {"status": "canceled", "size_matched": 1000.0}
    ex = make_executor(cfg, ledger, trader=trader)

    result = ex.execute(make_plan(size_usd=50.0))
    assert result.status == "filled"
    assert result.filled_size == 1000.0


# --- take-profit продажа ---

def test_execute_sell_respects_min_price(cfg, ledger):
    ex = make_executor(cfg, ledger, book=make_book(best_bid=0.05, best_ask=0.06))
    plan = make_plan()
    assert ex.execute_sell(plan, size=500, min_price=0.055).status == "skipped"
    result = ex.execute_sell(plan, size=500, min_price=0.04)
    assert result.status == "filled"
    assert result.avg_price == pytest.approx(0.05)
