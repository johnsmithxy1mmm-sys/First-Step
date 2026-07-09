"""Оценщик: ансамбль, edge-фильтр, когерентность, base rates, momentum."""

import math
from datetime import datetime, timedelta, timezone

import pytest

from polymarket_bot.estimator import Estimator
from polymarket_bot.estimator.base_rates import (BaseRateEntry, BaseRatesSignal,
                                                 probability_before)
from polymarket_bot.estimator.coherence import CoherenceSignal, normalize_question
from polymarket_bot.estimator.ensemble import combine
from polymarket_bot.estimator.momentum import MomentumSignal
from polymarket_bot.models import Signal

from .conftest import make_candidate, make_market


# --- ансамбль ---

def test_ensemble_geometric_mean_math():
    c = make_candidate()  # p_mkt = 0.01
    signals = [Signal(name="s1", p_est=0.04, confidence=0.5)]
    est = combine(c, signals, market_anchor_confidence=0.5)
    # Равные веса: geo-mean(0.01, 0.04) = sqrt(0.01*0.04) = 0.02
    assert est.p_est == pytest.approx(math.sqrt(0.01 * 0.04), rel=1e-6)
    assert est.edge_ratio == pytest.approx(2.0, rel=1e-6)


def test_ensemble_defaults_to_market_without_signals():
    est = combine(make_candidate(), [], market_anchor_confidence=0.85)
    assert est.p_est == pytest.approx(0.01)
    assert est.edge_ratio == pytest.approx(1.0)


def test_ensemble_ignores_abstaining_signals():
    signals = [Signal(name="quiet", p_est=None, confidence=0.9)]
    est = combine(make_candidate(), signals, market_anchor_confidence=0.85)
    assert est.edge_ratio == pytest.approx(1.0)


# --- edge-фильтр ---

def test_edge_filter_threshold(cfg):
    estimator = Estimator(cfg)
    c = make_candidate()
    strong = combine(c, [Signal(name="s", p_est=0.2, confidence=2.0)], 0.5)
    weak = combine(c, [Signal(name="s", p_est=0.012, confidence=0.5)], 0.85)
    assert estimator.qualifies(strong)
    assert not estimator.qualifies(weak)


def test_edge_filter_rejects_expensive_market(cfg):
    estimator = Estimator(cfg)
    c = make_candidate(outcome_prices=[0.20, 0.80])  # p_mkt > max_p_mkt
    est = combine(c, [Signal(name="s", p_est=0.9, confidence=5.0)], 0.5)
    assert not estimator.qualifies(est)


# --- когерентность ---

def _negrisk_market(mid: str, yes_price: float, event_id: str = "ev1"):
    return make_market(
        id=mid, outcome_prices=[yes_price, 1 - yes_price],
        clob_token_ids=[f"{mid}-yes", f"{mid}-no"],
        event_id=event_id, event_neg_risk=True, event_title="Election",
    )


def test_negrisk_basket_underpriced_rescales_up():
    # Сумма Yes = 0.90 < 1: каждый исход недооценён, fair = p / 0.9.
    markets = [_negrisk_market("a", 0.02), _negrisk_market("b", 0.38),
               _negrisk_market("c", 0.50)]
    signal = CoherenceSignal(markets).evaluate(
        make_candidate(market=markets[0]))
    assert signal is not None
    assert signal.p_est == pytest.approx(0.02 / 0.90, rel=1e-6)
    assert signal.confidence == 0.9


def test_negrisk_basket_fair_sum_is_silent():
    markets = [_negrisk_market("a", 0.02), _negrisk_market("b", 0.48),
               _negrisk_market("c", 0.50)]  # сумма 1.00
    assert CoherenceSignal(markets).evaluate(make_candidate(market=markets[0])) is None


def test_calendar_chain_violation_detected():
    early_end = datetime.now(timezone.utc) + timedelta(days=20)
    late_end = datetime.now(timezone.utc) + timedelta(days=80)
    early = make_market(id="e", question="Will X resign by March 31?",
                        outcome_prices=[0.05, 0.95], end_date=early_end,
                        clob_token_ids=["e-yes", "e-no"])
    late = make_market(id="l", question="Will X resign by June 30?",
                       outcome_prices=[0.02, 0.98], end_date=late_end,
                       clob_token_ids=["l-yes", "l-no"])
    assert normalize_question(early.question) == normalize_question(late.question)

    signal = CoherenceSignal([early, late]).evaluate(make_candidate(market=late))
    assert signal is not None            # поздний дешевле раннего — нарушение
    assert signal.p_est == pytest.approx(0.05)

    # Ранний рынок нарушения не имеет (нет более ранних дороже него).
    assert CoherenceSignal([early, late]).evaluate(make_candidate(market=early)) is None


# --- base rates ---

def test_probability_before_transform():
    assert probability_before(0.5, 365) == pytest.approx(0.5)
    assert probability_before(0.5, 0) == 0.0
    # Полгода при годовой 0.5: 1 - 0.5^0.5 ~ 0.2929
    assert probability_before(0.5, 182.5) == pytest.approx(1 - 0.5**0.5, rel=1e-3)


def test_base_rates_signal_matches_keywords():
    entry = BaseRateEntry(name="quake", keywords_all=["earthquake"],
                          keywords_any=["8.0", "magnitude 8"],
                          annual_probability=0.63, confidence=0.5)
    signal = BaseRatesSignal([entry]).evaluate(make_candidate())
    assert signal is not None
    assert 0 < signal.p_est < 0.63
    # No-сторона не оценивается базовыми частотами.
    assert BaseRatesSignal([entry]).evaluate(make_candidate(outcome_index=1)) is None


# --- momentum ---

def test_momentum_fires_on_price_rise_with_volume():
    c = make_candidate(one_day_price_change=0.01)  # +100% к цене 0.01
    signal = MomentumSignal().evaluate(c)
    assert signal is not None
    assert signal.p_est > c.p_mkt


def test_momentum_silent_on_decline_or_flat():
    assert MomentumSignal().evaluate(make_candidate(one_day_price_change=-0.005)) is None
    assert MomentumSignal().evaluate(make_candidate(one_day_price_change=0.0)) is None


# --- интеграция estimator ---

def test_estimator_full_pass(cfg):
    estimator = Estimator(cfg)
    markets = [_negrisk_market("a", 0.02), _negrisk_market("b", 0.28),
               _negrisk_market("c", 0.40)]  # сумма 0.70 — сильный сигнал корзины
    candidates = [make_candidate(market=markets[0])]
    estimates = estimator.estimate_all(candidates, markets)
    assert len(estimates) == 1
    assert estimates[0].p_est > estimates[0].p_mkt
    names = {s.name for s in estimates[0].signals}
    assert "market" in names and "coherence" in names
