"""Estimator: ensemble, edge filter, coherence, base rates, momentum."""

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


# --- ensemble ---

def test_ensemble_geometric_mean_math():
    c = make_candidate()  # p_mkt = 0.01
    signals = [Signal(name="s1", p_est=0.04, confidence=0.5)]
    est = combine(c, signals, market_anchor_confidence=0.5)
    # Equal weights: geo-mean(0.01, 0.04) = sqrt(0.01*0.04) = 0.02
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


# --- edge filter ---

def test_edge_filter_threshold(cfg):
    estimator = Estimator(cfg)
    c = make_candidate()
    strong = combine(c, [Signal(name="s", p_est=0.2, confidence=1.0)], 0.5)
    weak = combine(c, [Signal(name="s", p_est=0.012, confidence=0.5)], 0.85)
    assert estimator.qualifies(strong)
    assert not estimator.qualifies(weak)


def test_edge_filter_rejects_expensive_market(cfg):
    estimator = Estimator(cfg)
    c = make_candidate(outcome_prices=[0.20, 0.80])  # p_mkt > max_p_mkt
    est = combine(c, [Signal(name="s", p_est=0.9, confidence=1.0)], 0.5)
    assert not estimator.qualifies(est)


# --- coherence ---

def _negrisk_market(mid: str, yes_price: float, event_id: str = "ev1"):
    return make_market(
        id=mid, outcome_prices=[yes_price, 1 - yes_price],
        clob_token_ids=[f"{mid}-yes", f"{mid}-no"],
        event_id=event_id, event_neg_risk=True, event_title="Election",
    )


def test_negrisk_basket_underpriced_rescales_up():
    # Sum of Yes = 0.90 < 1: every outcome underpriced, fair = p / 0.9.
    markets = [_negrisk_market("a", 0.02), _negrisk_market("b", 0.38),
               _negrisk_market("c", 0.50)]
    signal = CoherenceSignal(markets).evaluate(
        make_candidate(market=markets[0]))
    assert signal is not None
    assert signal.p_est == pytest.approx(0.02 / 0.90, rel=1e-6)
    assert signal.confidence == 0.9


def test_negrisk_basket_fair_sum_is_silent():
    markets = [_negrisk_market("a", 0.02), _negrisk_market("b", 0.48),
               _negrisk_market("c", 0.50)]  # sum 1.00
    assert CoherenceSignal(markets).evaluate(make_candidate(market=markets[0])) is None


def _calendar_pair(q_early="Will X resign by March 31, 2027?",
                   q_late="Will X resign by June 30, 2027?",
                   end_early=None, end_late=None):
    end_early = end_early or datetime(2027, 3, 31, tzinfo=timezone.utc)
    end_late = end_late or datetime(2027, 6, 30, tzinfo=timezone.utc)
    early = make_market(id="e", question=q_early, outcome_prices=[0.05, 0.95],
                        end_date=end_early, clob_token_ids=["e-yes", "e-no"])
    late = make_market(id="l", question=q_late, outcome_prices=[0.02, 0.98],
                       end_date=end_late, clob_token_ids=["l-yes", "l-no"])
    return early, late


def test_calendar_chain_violation_detected():
    early, late = _calendar_pair()
    assert normalize_question(early.question) == normalize_question(late.question)

    signal = CoherenceSignal([early, late]).evaluate(make_candidate(market=late))
    assert signal is not None            # later cheaper than earlier — violation
    assert signal.p_est == pytest.approx(0.05)

    # The earlier market has no violation (no earlier one pricier than it).
    assert CoherenceSignal([early, late]).evaluate(make_candidate(market=early)) is None


def test_calendar_chain_refuses_inverted_metadata():
    """endDate contradicting the wording (the real GPT-6 Gamma bug) must not
    produce a signal — a 0.85-confidence boost on a lie buys the wrong side."""
    early, late = _calendar_pair(
        end_early=datetime(2027, 9, 30, tzinfo=timezone.utc),   # metadata LATE
        end_late=datetime(2027, 4, 1, tzinfo=timezone.utc))     # metadata EARLY
    assert CoherenceSignal([early, late]).evaluate(make_candidate(market=late)) is None


def test_calendar_chain_refuses_different_thresholds():
    """normalize_question strips numbers, so '$150k by June' and '$200k by
    December' share a chain key — but a different threshold breaks the
    implication and must not read as a calendar violation."""
    early, late = _calendar_pair(
        q_early="Will Bitcoin reach $150,000 by June 30, 2027?",
        q_late="Will Bitcoin reach $200,000 by December 31, 2027?",
        end_early=datetime(2027, 6, 30, tzinfo=timezone.utc),
        end_late=datetime(2027, 12, 31, tzinfo=timezone.utc))
    assert normalize_question(early.question) == normalize_question(late.question)
    assert CoherenceSignal([early, late]).evaluate(make_candidate(market=late)) is None


def test_calendar_chain_refuses_undated_wording():
    """No parseable year in the text -> direction unverifiable -> no signal."""
    early, late = _calendar_pair(q_early="Will X resign by March 31?",
                                 q_late="Will X resign by June 30?",
                                 end_early=datetime.now(timezone.utc) + timedelta(days=20),
                                 end_late=datetime.now(timezone.utc) + timedelta(days=80))
    assert CoherenceSignal([early, late]).evaluate(make_candidate(market=late)) is None


# --- base rates ---

def test_probability_before_transform():
    assert probability_before(0.5, 365) == pytest.approx(0.5)
    assert probability_before(0.5, 0) == 0.0
    # Half a year at annual 0.5: 1 - 0.5^0.5 ~ 0.2929
    assert probability_before(0.5, 182.5) == pytest.approx(1 - 0.5**0.5, rel=1e-3)


def test_base_rates_signal_matches_keywords():
    entry = BaseRateEntry(name="quake", keywords_all=["earthquake"],
                          keywords_any=["8.0", "magnitude 8"],
                          annual_probability=0.63, confidence=0.5)
    signal = BaseRatesSignal([entry]).evaluate(make_candidate())
    assert signal is not None
    assert 0 < signal.p_est < 0.63
    # The No side is not estimated by base rates.
    assert BaseRatesSignal([entry]).evaluate(make_candidate(outcome_index=1)) is None


# --- momentum ---

def test_momentum_fires_on_price_rise_with_volume():
    c = make_candidate(one_day_price_change=0.01)  # +100% on a 0.01 price
    signal = MomentumSignal().evaluate(c)
    assert signal is not None
    assert signal.p_est > c.p_mkt


def test_momentum_silent_on_decline_or_flat():
    assert MomentumSignal().evaluate(make_candidate(one_day_price_change=-0.005)) is None
    assert MomentumSignal().evaluate(make_candidate(one_day_price_change=0.0)) is None


# --- estimator integration ---

def test_estimator_full_pass(cfg):
    estimator = Estimator(cfg)
    markets = [_negrisk_market("a", 0.02), _negrisk_market("b", 0.28),
               _negrisk_market("c", 0.40)]  # sum 0.70 — strong basket signal
    candidates = [make_candidate(market=markets[0])]
    estimates = estimator.estimate_all(candidates, markets)
    assert len(estimates) == 1
    assert estimates[0].p_est > estimates[0].p_mkt
    names = {s.name for s in estimates[0].signals}
    assert "market" in names and "coherence" in names
