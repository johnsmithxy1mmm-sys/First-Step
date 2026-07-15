"""Fading overpriced tails: condition, bias correction, NO side, caps."""

from unittest import mock

import pytest

from polymarket_bot.estimator.ensemble import combine
from polymarket_bot.executor import Executor
from polymarket_bot.fade import FadeStrategy
from polymarket_bot.models import Signal
from polymarket_bot.portfolio import Portfolio

from .conftest import make_book, make_candidate, make_market


def make_estimate(p_mkt=0.05, p_est=None, signals=None, **market_over):
    """Estimate for a YES tail. p_est=None -> no signal (p_est=p_mkt via anchor)."""
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


# --- fade condition ---

def test_systematic_fade_fires_without_signal(cfg, ledger):
    """Even with no signal, bias_discount alone makes the tail fadeable."""
    fade, _ = make_fade(cfg, ledger)
    plan = fade.plan(make_estimate(p_mkt=0.05, p_est=None))
    assert plan is not None
    # Buy NO (index 1) of the underpriced market.
    assert plan.estimate.candidate.outcome_index == 1
    assert plan.estimate.candidate.p_mkt == pytest.approx(0.95)   # entry No = 1 - 0.05
    # Fair Yes = 0.05*0.7 = 0.035 -> fair No = 0.965.
    assert plan.estimate.p_est == pytest.approx(0.965)
    assert plan.limit_price_cap == pytest.approx(0.965 - 0.005)


def test_signal_deepens_fade(cfg, ledger):
    """A signal (base rate below market) fades harder than bias alone."""
    fade, _ = make_fade(cfg, ledger)
    # Estimator says Yes=0.01 (< 0.05*0.7=0.035) -> fair No=0.99.
    plan = fade.plan(make_estimate(p_mkt=0.05, p_est=0.01))
    assert plan is not None
    assert plan.estimate.p_est == pytest.approx(0.99)


def test_no_fade_for_underpriced_tail(cfg, ledger):
    """If the estimator thinks the tail is UNDERpriced — that's a longshot, not a fade."""
    fade, _ = make_fade(cfg, ledger)
    assert fade.plan(make_estimate(p_mkt=0.05, p_est=0.15)) is None    # ratio 3.0 >= 2.0


def test_mild_drift_above_market_still_fades(cfg, ledger):
    """p_est a hair above market (< veto ratio) is anchor noise — still fade."""
    fade, _ = make_fade(cfg, ledger)
    plan = fade.plan(make_estimate(p_mkt=0.03, p_est=0.042))           # ratio 1.4 < 2.0
    assert plan is not None
    assert plan.estimate.candidate.outcome_index == 1                 # buy NO


def test_genuine_longshot_vetoes_fade(cfg, ledger):
    """p_est at the veto threshold (2x market) is a real longshot — no fade."""
    fade, _ = make_fade(cfg, ledger)
    assert fade.plan(make_estimate(p_mkt=0.03, p_est=0.06)) is None    # ratio 2.0 >= 2.0


def test_no_fade_above_max_price(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    assert fade.plan(make_estimate(p_mkt=0.30, p_est=None)) is None    # not a tail


def test_thin_edge_rejected(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.bias_discount = 0.05        # edge = 0.05*0.05 = 0.0025 < threshold 0.005
    assert fade.plan(make_estimate(p_mkt=0.05, p_est=None)) is None


def test_reject_reason_matches_plan(cfg, ledger):
    """reject_reason mirrors plan()'s pre-sizing gates (used by --mode diagnose)."""
    fade, _ = make_fade(cfg, ledger)
    assert fade.reject_reason(make_estimate(p_mkt=0.05, p_est=None)) is None
    assert "tail price" in fade.reject_reason(make_estimate(p_mkt=0.30))
    assert "longshot" in fade.reject_reason(make_estimate(p_mkt=0.05, p_est=0.15))


# --- sizing and caps (reuse the portfolio) ---

def test_fade_size_respects_market_cap(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    plan = fade.plan(make_estimate(p_mkt=0.05, p_est=None))
    assert plan is not None
    assert plan.size_usd <= cfg.portfolio.max_market_pct * cfg.portfolio.bankroll_usd + 1e-9


# --- end-to-end execution via the executor (dry-run) ---

def test_fade_cycle_enters_and_tags_strategy(cfg, ledger):
    fade, clob = make_fade(cfg, ledger)
    # NO-token book: ask 0.95, bid 0.94 — maker bid within cap.
    clob.order_book.return_value = make_book(best_bid=0.94, best_ask=0.96, depth=100_000)
    with mock.patch("polymarket_bot.fade.alert") as a:
        entered = fade.cycle([make_estimate(p_mkt=0.05, p_est=None)])
    assert entered == 1
    positions = ledger.open_positions("dry-run")
    assert len(positions) == 1
    # Recorded as strategy 'fade', buying the NO outcome.
    row = ledger._conn.execute("SELECT strategy, side FROM trades").fetchone()
    assert row["strategy"] == "fade" and row["side"] == "BUY"
    # The fill is announced to Telegram.
    a.assert_called_once()
    assert "FADE fill" in a.call_args[0][0]


def test_fade_disabled_is_silent(cfg, ledger):
    fade, _ = make_fade(cfg, ledger)
    cfg.fade.enabled = False
    assert fade.cycle([make_estimate(p_mkt=0.05, p_est=None)]) == 0
