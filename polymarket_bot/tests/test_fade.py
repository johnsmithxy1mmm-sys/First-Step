"""Фейдинг переоценённых хвостов: условие, bias-поправка, NO-сторона, кэпы."""

from unittest import mock

import pytest

from polymarket_bot.estimator.ensemble import combine
from polymarket_bot.executor import Executor
from polymarket_bot.fade import FadeStrategy
from polymarket_bot.models import Signal
from polymarket_bot.portfolio import Portfolio

from .conftest import make_book, make_candidate, make_market


def make_estimate(p_mkt=0.05, p_est=None, signals=None, **market_over):
    """Оценка YES-хвоста. p_est=None -> нет сигнала (p_est=p_mkt через якорь)."""
    c = make_candidate(outcome_prices=[p_mkt, 1 - p_mkt], **market_over)
    if signals is None:
        signals = [] if p_est is None else [Signal(name="s", p_est=p_est, confidence=1e9)]
    return combine(c, signals, market_anchor_confidence=1e-9 if signals else 0.85)


def make_fade(cfg, ledger, trader=None):
    cfg.fade.enabled = True
    clob = mock.Mock()
    clob.order_book.return_value = None
    portfolio = Portfolio(cfg, ledger, "dry-run")
    executor = Executor(cfg, ledger, clob, trader, "dry-run")
    return FadeStrategy(cfg, ledger, portfolio, executor, "dry-run"), clob


# --- условие фейда ---

def test_systematic_fade_fires_without_signal(cfg, ledger):
    """Даже без сигнала bias_discount делает хвост фейдабельным."""
    fade, _ = make_fade(cfg, ledger)
    plan = fade.plan(make_estimate(p_mkt=0.05, p_est=None))
    assert plan is not None
    # Покупаем NO (индекс 1) недооценённого рынка.
    assert plan.estimate.candidate.outcome_index == 1
    assert plan.estimate.candidate.p_mkt == pytest.approx(0.95)   # entry No = 1 - 0.05
    # Честная Yes = 0.05·0.7 = 0.035 -> честная No = 0.965.
    assert plan.estimate.p_est == pytest.approx(0.965)
    assert plan.limit_price_cap == pytest.approx(0.965 - 0.005)


def test_signal_deepens_fade(cfg, ledger):
    """Сигнал (base rate ниже рынка) даёт более сильный фейд, чем один bias."""
    fade, _ = make_fade(cfg, ledger)
    # Оценщик говорит Yes=0.01 (< 0.05·0.7=0.035) -> честная No=0.99.
    plan = fade.plan(make_estimate(p_mkt=0.05, p_est=0.01))
    assert plan is not None
    assert plan.estimate.p_est == pytest.approx(0.99)


def test_no_fade_for_underpriced_tail(cfg, ledger):
    """Если оценщик считает хвост НЕдооценённым — это лонгшот, не фейд."""
    fade, _ = make_fade(cfg, ledger)
    assert fade.plan(make_estimate(p_mkt=0.05, p_est=0.15)) is None


def test_no_fade_above_max_price(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    assert fade.plan(make_estimate(p_mkt=0.30, p_est=None)) is None    # не хвост


def test_thin_edge_rejected(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.bias_discount = 0.05        # edge = 0.05·0.05 = 0.0025 < порога 0.005
    assert fade.plan(make_estimate(p_mkt=0.05, p_est=None)) is None


# --- сайзинг и кэпы (переиспользуют портфель) ---

def test_fade_size_respects_market_cap(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    plan = fade.plan(make_estimate(p_mkt=0.05, p_est=None))
    assert plan is not None
    assert plan.size_usd <= cfg.portfolio.max_market_pct * cfg.portfolio.bankroll_usd + 1e-9


# --- сквозное исполнение через executor (dry-run) ---

def test_fade_cycle_enters_and_tags_strategy(cfg, ledger):
    fade, clob = make_fade(cfg, ledger)
    # Книга NO-токена: ask 0.95, bid 0.94 — maker-бид внутри cap.
    clob.order_book.return_value = make_book(best_bid=0.94, best_ask=0.96, depth=100_000)
    entered = fade.cycle([make_estimate(p_mkt=0.05, p_est=None)])
    assert entered == 1
    positions = ledger.open_positions("dry-run")
    assert len(positions) == 1
    # Записано как стратегия 'fade', покупка NO-исхода.
    row = ledger._conn.execute("SELECT strategy, side FROM trades").fetchone()
    assert row["strategy"] == "fade" and row["side"] == "BUY"


def test_fade_disabled_is_silent(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.enabled = False
    assert fade.cycle([make_estimate(p_mkt=0.05, p_est=None)]) == 0
