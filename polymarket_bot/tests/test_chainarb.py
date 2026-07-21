"""Chain (ladder) arbitrage: monotonic-constraint classification and payoff."""

from datetime import datetime, timedelta, timezone
from unittest import mock

import pytest

from polymarket_bot.chainarb import (ChainArbitrage, ChainLeg, ChainPair,
                                     classify_pair)
from polymarket_bot.models import BookLevel, OrderBook

from .conftest import make_market


def dated(**overrides):
    return make_market(**overrides)


def book(ask=0.5, ask_size=1000.0, bid=0.48, bid_size=1000.0) -> OrderBook:
    return OrderBook(bids=[BookLevel(price=bid, size=bid_size)],
                     asks=[BookLevel(price=ask, size=ask_size)])


# --- classify_pair: DATE ladder ---

def test_date_ladder_classifies_later_deadline_as_superset(cfg):
    a = dated(id="fed-june", question="Will the Fed cut rates by June 2026?",
             end_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
             clob_token_ids=["a-y", "a-n"])
    b = dated(id="fed-july", question="Will the Fed cut rates by July 2026?",
             end_date=datetime(2026, 7, 31, tzinfo=timezone.utc),
             clob_token_ids=["b-y", "b-n"])
    result = classify_pair(a, b)
    assert result is not None
    subset, superset, kind = result
    assert kind == "date"
    assert subset.id == "fed-june"       # earlier deadline = subset
    assert superset.id == "fed-july"     # later deadline = superset


def test_date_ladder_order_independent(cfg):
    a = dated(id="fed-june", question="Will the Fed cut rates by June 2026?",
             end_date=datetime(2026, 6, 30, tzinfo=timezone.utc))
    b = dated(id="fed-july", question="Will the Fed cut rates by July 2026?",
             end_date=datetime(2026, 7, 31, tzinfo=timezone.utc))
    r1 = classify_pair(a, b)
    r2 = classify_pair(b, a)
    assert r1 is not None and r2 is not None
    assert r1[0].id == r2[0].id == "fed-june"
    assert r1[1].id == r2[1].id == "fed-july"


def test_date_ladder_rejects_non_absorbing_wording(cfg):
    """'in June' is a specific window, not a cumulative deadline -> refuse."""
    a = dated(question="Will the Fed cut rates in June 2026?",
             end_date=datetime(2026, 6, 30, tzinfo=timezone.utc))
    b = dated(question="Will the Fed cut rates in July 2026?",
             end_date=datetime(2026, 7, 31, tzinfo=timezone.utc))
    assert classify_pair(a, b) is None


def test_date_ladder_rejects_unrelated_questions(cfg):
    a = dated(question="Will the Fed cut rates by June 2026?",
             end_date=datetime(2026, 6, 30, tzinfo=timezone.utc))
    b = dated(question="Will Bitcoin reach $150,000 by June 2026?",
             end_date=datetime(2026, 6, 30, tzinfo=timezone.utc))
    assert classify_pair(a, b) is None


def test_no_end_date_rejected(cfg):
    a = dated(question="Will the Fed cut rates by June 2026?", end_date=None)
    b = dated(question="Will the Fed cut rates by July 2026?",
             end_date=datetime(2026, 7, 31, tzinfo=timezone.utc))
    assert classify_pair(a, b) is None


# --- classify_pair: VALUE ladder ---

def test_value_ladder_classifies_higher_threshold_as_subset(cfg):
    same_date = datetime(2026, 12, 31, tzinfo=timezone.utc)
    a = dated(id="btc-150k", question="Will Bitcoin reach $150,000 by Dec 2026?",
             end_date=same_date, clob_token_ids=["a-y", "a-n"])
    b = dated(id="btc-200k", question="Will Bitcoin reach $200,000 by Dec 2026?",
             end_date=same_date, clob_token_ids=["b-y", "b-n"])
    result = classify_pair(a, b)
    assert result is not None
    subset, superset, kind = result
    assert kind == "value"
    assert subset.id == "btc-200k"       # higher threshold = subset
    assert superset.id == "btc-150k"     # lower threshold = superset


def test_value_ladder_rejects_reversed_direction_wording(cfg):
    """'stays below $X' has the OPPOSITE monotonicity -> refuse, don't guess."""
    same_date = datetime(2026, 12, 31, tzinfo=timezone.utc)
    a = dated(question="Will Bitcoin stay below $150,000 through Dec 2026?", end_date=same_date)
    b = dated(question="Will Bitcoin stay below $200,000 through Dec 2026?", end_date=same_date)
    assert classify_pair(a, b) is None


def test_value_ladder_requires_same_date(cfg):
    a = dated(question="Will Bitcoin reach $150,000 by Dec 2026?",
             end_date=datetime(2026, 12, 31, tzinfo=timezone.utc))
    b = dated(question="Will Bitcoin reach $200,000 by Mar 2027?",
             end_date=datetime(2027, 3, 31, tzinfo=timezone.utc))
    # Both value AND date differ -> ambiguous, must not guess a direction.
    assert classify_pair(a, b) is None


def test_identical_markets_not_a_ladder(cfg):
    same_date = datetime(2026, 12, 31, tzinfo=timezone.utc)
    a = dated(question="Will Bitcoin reach $150,000 by Dec 2026?", end_date=same_date)
    b = dated(question="Will Bitcoin reach $150,000 by Dec 2026?", end_date=same_date)
    assert classify_pair(a, b) is None


# --- ChainPair payoff math ---

def test_payoff_is_worst_case_one_dollar_when_cost_below_one():
    """The core proof: cost < $1 with a guaranteed >= $1 payout = no loss."""
    subset_leg = ChainLeg(market=dated(id="s"), outcome_index=1,
                          token_id="s-n", ask=0.30, depth=1000)   # buy No(subset)
    superset_leg = ChainLeg(market=dated(id="p"), outcome_index=0,
                            token_id="p-y", ask=0.60, depth=1000)  # buy Yes(superset)
    pair = ChainPair(event_id="e", event_title="t", kind="date",
                     subset=subset_leg, superset=superset_leg, taker_fee=0.0)
    assert pair.cost_per_set == pytest.approx(0.90)
    assert pair.payout_per_set == 1.0
    assert pair.profit_per_set == pytest.approx(0.10)
    assert pair.net_profit_pct == pytest.approx(0.10 / 0.90)


def test_fee_reduces_net_but_not_gross():
    subset_leg = ChainLeg(market=dated(id="s"), outcome_index=1, token_id="s-n",
                          ask=0.30, depth=1000)
    superset_leg = ChainLeg(market=dated(id="p"), outcome_index=0, token_id="p-y",
                            ask=0.60, depth=1000)
    pair = ChainPair(event_id="e", event_title="t", kind="date",
                     subset=subset_leg, superset=superset_leg, taker_fee=0.05)
    assert pair.fee_per_set == pytest.approx(0.05 * 0.90)
    assert pair.net_profit_per_set < pair.profit_per_set


def test_max_sets_limited_by_thinnest_leg():
    subset_leg = ChainLeg(market=dated(id="s"), outcome_index=1, token_id="s-n",
                          ask=0.30, depth=50)
    superset_leg = ChainLeg(market=dated(id="p"), outcome_index=0, token_id="p-y",
                            ask=0.60, depth=1000)
    pair = ChainPair(event_id="e", event_title="t", kind="date",
                     subset=subset_leg, superset=superset_leg, taker_fee=0.0)
    assert pair.max_sets_by_depth() == 50


# --- ChainArbitrage: prefilter / verify / cycle / execute ---

def make_chain(cfg, ledger, mode="dry-run", clob=None):
    cfg.chain_arb.enabled = True
    return ChainArbitrage(cfg, ledger, clob or mock.Mock(), None, mode)


def test_prefilter_finds_gamma_level_violation(cfg, ledger):
    same_date = datetime.now(timezone.utc) + timedelta(days=60)
    subset_m = dated(id="btc-200k", event_id="ev1",
                     question="Will Bitcoin reach $200,000 by Dec 2026?",
                     end_date=same_date, outcome_prices=[0.10, 0.90],
                     clob_token_ids=["s-y", "s-n"], volume_24h_usd=50_000)
    superset_m = dated(id="btc-150k", event_id="ev1",
                       question="Will Bitcoin reach $150,000 by Dec 2026?",
                       end_date=same_date, outcome_prices=[0.05, 0.95],  # priced BELOW subset -> violation
                       clob_token_ids=["p-y", "p-n"], volume_24h_usd=50_000)
    chain = make_chain(cfg, ledger)
    pairs = chain.prefilter_pairs([subset_m, superset_m])
    assert len(pairs) == 1
    subset, superset, kind = pairs[0]
    assert kind == "value" and subset.id == "btc-200k" and superset.id == "btc-150k"


def test_prefilter_ignores_pair_without_violation(cfg, ledger):
    same_date = datetime.now(timezone.utc) + timedelta(days=60)
    subset_m = dated(id="btc-200k", event_id="ev1",
                     question="Will Bitcoin reach $200,000 by Dec 2026?",
                     end_date=same_date, outcome_prices=[0.05, 0.95],
                     clob_token_ids=["s-y", "s-n"], volume_24h_usd=50_000)
    superset_m = dated(id="btc-150k", event_id="ev1",
                       question="Will Bitcoin reach $150,000 by Dec 2026?",
                       end_date=same_date, outcome_prices=[0.10, 0.90],  # correctly higher -> no violation
                       clob_token_ids=["p-y", "p-n"], volume_24h_usd=50_000)
    chain = make_chain(cfg, ledger)
    assert chain.prefilter_pairs([subset_m, superset_m]) == []


def test_verify_builds_pair_from_live_books(cfg, ledger):
    subset_m = dated(id="s", clob_token_ids=["s-y", "s-n"])
    superset_m = dated(id="p", clob_token_ids=["p-y", "p-n"])
    clob = mock.Mock()
    clob.order_book.side_effect = lambda tok: {
        "p-y": book(ask=0.55),   # superset Yes ask
        "s-n": book(ask=0.35),   # subset No ask
    }[tok]
    chain = make_chain(cfg, ledger, clob=clob)
    pair = chain.verify(subset_m, superset_m, "date")
    assert pair is not None
    assert pair.superset.ask == 0.55 and pair.subset.ask == 0.35
    assert pair.cost_per_set == pytest.approx(0.90)


def test_verify_returns_none_on_missing_book(cfg, ledger):
    subset_m = dated(id="s", clob_token_ids=["s-y", "s-n"])
    superset_m = dated(id="p", clob_token_ids=["p-y", "p-n"])
    clob = mock.Mock()
    clob.order_book.return_value = None
    chain = make_chain(cfg, ledger, clob=clob)
    assert chain.verify(subset_m, superset_m, "date") is None


def test_cycle_respects_min_net_edge_and_haircut(cfg, ledger):
    same_date = datetime.now(timezone.utc) + timedelta(days=60)
    subset_m = dated(id="btc-200k", event_id="ev1",
                     question="Will Bitcoin reach $200,000 by Dec 2026?",
                     end_date=same_date, outcome_prices=[0.30, 0.70],
                     clob_token_ids=["s-y", "s-n"], volume_24h_usd=50_000)
    superset_m = dated(id="btc-150k", event_id="ev1",
                       question="Will Bitcoin reach $150,000 by Dec 2026?",
                       end_date=same_date, outcome_prices=[0.10, 0.90],
                       clob_token_ids=["p-y", "p-n"], volume_24h_usd=50_000)
    clob = mock.Mock()
    clob.order_book.side_effect = lambda tok: {
        "p-y": book(ask=0.10),   # superset cheap -> big violation vs subset's 0.30
        "s-n": book(ask=0.70),
    }[tok]
    cfg.chain_arb.min_net_edge = 0.03
    cfg.chain_arb.classification_haircut = 0.02
    chain = make_chain(cfg, ledger, clob=clob)
    found = chain.cycle([subset_m, superset_m])
    assert len(found) == 1
    assert found[0].net_profit_pct - cfg.chain_arb.classification_haircut >= cfg.chain_arb.min_net_edge


def test_cycle_executes_and_tags_both_legs(cfg, ledger):
    same_date = datetime.now(timezone.utc) + timedelta(days=60)
    subset_m = dated(id="btc-200k", event_id="ev1",
                     question="Will Bitcoin reach $200,000 by Dec 2026?",
                     end_date=same_date, outcome_prices=[0.30, 0.70],
                     clob_token_ids=["s-y", "s-n"], volume_24h_usd=50_000,
                     min_order_size=1.0, tick_size=0.001)
    superset_m = dated(id="btc-150k", event_id="ev1",
                       question="Will Bitcoin reach $150,000 by Dec 2026?",
                       end_date=same_date, outcome_prices=[0.10, 0.90],
                       clob_token_ids=["p-y", "p-n"], volume_24h_usd=50_000,
                       min_order_size=1.0, tick_size=0.001)
    clob = mock.Mock()
    clob.order_book.side_effect = lambda tok: {
        "p-y": book(ask=0.10, ask_size=100),
        "s-n": book(ask=0.70, ask_size=100),
    }[tok]
    cfg.chain_arb.execute = True
    cfg.chain_arb.max_stake_usd = 1000
    chain = make_chain(cfg, ledger, clob=clob)
    chain.cycle([subset_m, superset_m])
    positions = ledger.open_positions("dry-run")
    assert len(positions) == 2
    tokens = {p.token_id for p in positions}
    assert tokens == {"p-y", "s-n"}


def test_execute_returns_zero_below_min_size(cfg, ledger):
    subset_m = dated(id="s", clob_token_ids=["s-y", "s-n"], min_order_size=100.0)
    superset_m = dated(id="p", clob_token_ids=["p-y", "p-n"], min_order_size=100.0)
    subset_leg = ChainLeg(market=subset_m, outcome_index=1, token_id="s-n", ask=0.30, depth=5)
    superset_leg = ChainLeg(market=superset_m, outcome_index=0, token_id="p-y", ask=0.60, depth=5)
    pair = ChainPair(event_id="e", event_title="t", kind="date",
                     subset=subset_leg, superset=superset_leg, taker_fee=0.0)
    chain = make_chain(cfg, ledger)
    assert chain.execute(pair) == 0.0
