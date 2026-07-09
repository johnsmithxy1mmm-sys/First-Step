"""Портфель: Келли, кэпы, kill-switch, правило выхода."""

import pytest

from polymarket_bot.estimator.ensemble import combine
from polymarket_bot.models import Position, Signal
from polymarket_bot.portfolio import (Portfolio, classify_category, correlation,
                                      kelly_fraction)

from .conftest import make_candidate


def make_estimate(p_mkt=0.01, p_est=0.03, **overrides):
    c = make_candidate(outcome_prices=[p_mkt, 1 - p_mkt], **overrides)
    # Один сигнал с гигантским весом задаёт p_est точно.
    return combine(c, [Signal(name="s", p_est=p_est, confidence=1e9)], 1e-9)


def record_buy(ledger, est, category, usd):
    size = usd / est.p_mkt
    ledger.record_trade(mode="dry-run", estimate=est, category=category,
                        side="BUY", price=est.p_mkt, size=size,
                        order_id=None, status="sim-filled")


# --- Келли ---

def test_kelly_formula():
    # p=0.03, price=0.01: f* = 0.02/0.99
    assert kelly_fraction(0.03, 0.01) == pytest.approx(0.02 / 0.99)
    assert kelly_fraction(0.005, 0.01) == 0.0  # отрицательный edge
    assert kelly_fraction(0.5, 0.0) == 0.0


def test_size_is_fractional_kelly_capped_by_market_limit(cfg, ledger):
    p = Portfolio(cfg, ledger, "dry-run")
    est = make_estimate(p_mkt=0.01, p_est=0.03)
    plan = p.size_trade(est)
    assert plan is not None
    kelly_usd = cfg.portfolio.kelly_fraction * kelly_fraction(0.03, 0.01) \
        * cfg.portfolio.bankroll_usd
    market_cap = cfg.portfolio.max_market_pct * cfg.portfolio.bankroll_usd
    assert plan.size_usd == pytest.approx(min(kelly_usd, market_cap), abs=0.01)
    assert plan.size_usd <= market_cap + 1e-9


def test_no_plan_when_no_edge(cfg, ledger):
    p = Portfolio(cfg, ledger, "dry-run")
    assert p.size_trade(make_estimate(p_mkt=0.02, p_est=0.02)) is None


# --- кэпы категории и общей экспозиции ---

def test_category_cap_blocks_after_fill(cfg, ledger):
    p = Portfolio(cfg, ledger, "dry-run")
    cat_cap = cfg.portfolio.max_category_pct * cfg.portfolio.bankroll_usd  # $500
    est_old = make_estimate(p_mkt=0.01, p_est=0.05)
    record_buy(ledger, est_old, "nature", cat_cap)  # категория заполнена

    est_new = make_estimate(p_mkt=0.01, p_est=0.05)  # тот же землетрясенческий вопрос
    assert p.size_trade(est_new) is None


def test_correlated_category_reduces_room(cfg, ledger):
    p = Portfolio(cfg, ledger, "dry-run")
    # Геополитика заполнена: экономика (rho=0.5) получает вдвое меньший запас.
    record_buy(ledger, make_estimate(), "geopolitics",
               cfg.portfolio.max_category_pct * cfg.portfolio.bankroll_usd)
    est = make_estimate(
        p_mkt=0.01, p_est=0.05,
        question="Will the government shutdown trigger a recession?",
        description="Resolves YES per official BEA data on GDP contraction rules.")
    plan = p.size_trade(est)
    cat_cap = cfg.portfolio.max_category_pct * cfg.portfolio.bankroll_usd
    effective_room = cat_cap - correlation("economy", "geopolitics") * cat_cap
    assert plan is not None
    assert plan.size_usd <= effective_room + 1e-6


def test_total_exposure_cap(cfg, ledger):
    cfg.portfolio.max_total_exposure_pct = 0.001  # $5 на всё
    p = Portfolio(cfg, ledger, "dry-run")
    record_buy(ledger, make_estimate(), "sports", 5.0)
    assert p.size_trade(make_estimate(p_mkt=0.01, p_est=0.05)) is None


# --- kill-switch ---

def test_drawdown_kill_switch(cfg, ledger):
    p = Portfolio(cfg, ledger, "dry-run")
    assert not p.observe_only()

    # Покупка на $1000, резолюция в ноль: банк 5000 -> 4000 (просадка 20%).
    est = make_estimate(p_mkt=0.01, p_est=0.05)
    record_buy(ledger, est, "nature", 1000.0)
    ledger.record_resolution(est.candidate.token_id, est.candidate.market.id, won=False)
    assert p.drawdown() == pytest.approx(0.20, abs=0.01)
    assert not p.observe_only()  # 20% < 25%

    # Ещё -500: просадка 30% >= 25% -> стоп.
    est2 = make_estimate(p_mkt=0.01, p_est=0.05, id="m2",
                         clob_token_ids=["tok2-yes", "tok2-no"])
    record_buy(ledger, est2, "nature", 500.0)
    ledger.record_resolution("tok2-yes", "m2", won=False)
    assert p.observe_only()


# --- выход ---

def test_exit_rule_takes_partial_profit(cfg, ledger):
    p = Portfolio(cfg, ledger, "dry-run")
    pos = Position(token_id="t", market_id="m", question="q", outcome="Yes",
                   category="nature", size=1000, avg_price=0.01)
    assert p.exit_plan(pos, current_price=0.05) is None       # 5x < 7x
    plan = p.exit_plan(pos, current_price=0.08)               # 8x >= 7x
    assert plan is not None
    size, min_price = plan
    assert size == 600                                        # 60% позиции
    assert min_price == pytest.approx(0.01 * 7 * 0.8)


# --- классификатор категорий ---

def test_classify_category():
    assert classify_category("Will bitcoin hit $500k?") == "crypto"
    assert classify_category("Will Russia invade Moldova?") == "geopolitics"
    assert classify_category("Will it rain tomorrow in Paris?") == "other"
    assert correlation("crypto", "crypto") == 1.0
    assert correlation("geopolitics", "economy") == 0.5
