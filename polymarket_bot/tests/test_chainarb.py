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
                     subset=subset_leg, superset=superset_leg, taker_coef=0.0)
    assert pair.cost_per_set == pytest.approx(0.90)
    assert pair.payout_per_set == 1.0
    assert pair.profit_per_set == pytest.approx(0.10)
    assert pair.net_profit_pct == pytest.approx(0.10 / 0.90)


def test_fee_reduces_net_but_not_gross():
    """Fee is theta*p*(1-p) per leg, not theta*cost — the latter overstated a
    two-leg ladder by ~2x and pushed real pairs below the min-edge gate."""
    subset_leg = ChainLeg(market=dated(id="s"), outcome_index=1, token_id="s-n",
                          ask=0.30, depth=1000)
    superset_leg = ChainLeg(market=dated(id="p"), outcome_index=0, token_id="p-y",
                            ask=0.60, depth=1000)
    pair = ChainPair(event_id="e", event_title="t", kind="date",
                     subset=subset_leg, superset=superset_leg, taker_coef=0.05)
    expected = 0.05 * (0.30 * 0.70 + 0.60 * 0.40)
    assert pair.fee_per_set == pytest.approx(expected)
    assert pair.fee_per_set < 0.05 * 0.90          # cheaper than the flat model
    assert pair.net_profit_per_set < pair.profit_per_set


def test_max_sets_limited_by_thinnest_leg():
    subset_leg = ChainLeg(market=dated(id="s"), outcome_index=1, token_id="s-n",
                          ask=0.30, depth=50)
    superset_leg = ChainLeg(market=dated(id="p"), outcome_index=0, token_id="p-y",
                            ask=0.60, depth=1000)
    pair = ChainPair(event_id="e", event_title="t", kind="date",
                     subset=subset_leg, superset=superset_leg, taker_coef=0.0)
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
                     subset=subset_leg, superset=superset_leg, taker_coef=0.0)
    cfg.chain_arb.spoof_screen = False   # isolate the min-size guard from the book screen
    chain = make_chain(cfg, ledger)
    assert chain.execute(pair) == 0.0


def test_allow_execute_false_detects_but_places_nothing(cfg, ledger):
    """Kill-switch/observe-only/breaker: alerts keep flowing, orders do not."""
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
    found = chain.cycle([subset_m, superset_m], allow_execute=False)
    assert len(found) == 1                            # still detected
    assert ledger.open_positions("dry-run") == []     # but nothing traded


def test_dust_depth_produces_no_alert(cfg, ledger):
    """A 2-share ask can fake an edge nobody can trade — below min_sets, skip."""
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
        "p-y": book(ask=0.10, ask_size=2),            # dust
        "s-n": book(ask=0.70, ask_size=100),
    }[tok]
    chain = make_chain(cfg, ledger, clob=clob)
    assert chain.cycle([subset_m, superset_m]) == []


# --- date-ladder direction from TEXT, not endDate (the GPT-6 inversion bug) ---

def test_date_ladder_uses_text_deadline_not_endDate():
    """endDate INVERTED vs the wording (the real Gamma bug): refuse, don't
    fabricate an arbitrage from a mis-ordered pair."""
    # Wording: July (earlier) should be subset, Sept (later) superset.
    # But endDate is inverted: the July market carries a LATER endDate.
    july = dated(id="gpt-jul", event_id="ev",
                 question="Will GPT-6 be released by July 31, 2026?",
                 end_date=datetime(2026, 9, 30, tzinfo=timezone.utc),   # WRONG/late
                 clob_token_ids=["jy", "jn"])
    sept = dated(id="gpt-sep", event_id="ev",
                 question="Will GPT-6 be released by September 30, 2026?",
                 end_date=datetime(2026, 8, 1, tzinfo=timezone.utc),    # WRONG/early
                 clob_token_ids=["sy", "sn"])
    assert classify_pair(july, sept) is None       # text vs metadata disagree -> refuse


def test_date_ladder_text_and_metadata_agree_orders_by_deadline():
    july = dated(id="gpt-jul", event_id="ev",
                 question="Will GPT-6 be released by July 31, 2026?",
                 end_date=datetime(2026, 7, 31, tzinfo=timezone.utc),
                 clob_token_ids=["jy", "jn"])
    sept = dated(id="gpt-sep", event_id="ev",
                 question="Will GPT-6 be released by September 30, 2026?",
                 end_date=datetime(2026, 9, 30, tzinfo=timezone.utc),
                 clob_token_ids=["sy", "sn"])
    subset, superset, kind = classify_pair(july, sept)
    assert kind == "date"
    assert subset.id == "gpt-jul"                  # earlier text deadline = subset
    assert superset.id == "gpt-sep"


def test_deadline_parser_ignores_year_only_as_day():
    """'June 2026' (no day) must not read '20' out of '2026' as the day."""
    from polymarket_bot.chainarb import _extract_deadline
    m = dated(question="Will the Fed cut rates by June 2026?",
              end_date=datetime(2026, 6, 30, tzinfo=timezone.utc))
    d = _extract_deadline(m)
    assert d is not None and d.month == 6 and d.year == 2026


def test_date_ladder_refuses_without_parseable_deadline():
    """No year in the wording -> cannot verify direction from text -> refuse."""
    a = dated(question="Will the Fed cut rates by summer?",
              end_date=datetime(2026, 6, 30, tzinfo=timezone.utc))
    b = dated(question="Will the Fed cut rates by autumn?",
              end_date=datetime(2026, 9, 30, tzinfo=timezone.utc))
    assert classify_pair(a, b) is None


def test_inverted_pair_yields_no_arb_through_cycle(cfg, ledger):
    """End to end: the mis-ordered GPT-6 pair the bot spammed must produce
    zero chain-arb candidates now (it is refused at classification)."""
    july = dated(id="gpt-jul", event_id="ev",
                 question="Will GPT-6 be released by July 31, 2026?",
                 end_date=datetime(2026, 9, 30, tzinfo=timezone.utc),
                 outcome_prices=[0.006, 0.994], clob_token_ids=["jy", "jn"],
                 volume_24h_usd=50_000)
    sept = dated(id="gpt-sep", event_id="ev",
                 question="Will GPT-6 be released by September 30, 2026?",
                 end_date=datetime(2026, 8, 1, tzinfo=timezone.utc),
                 outcome_prices=[0.74, 0.26], clob_token_ids=["sy", "sn"],
                 volume_24h_usd=50_000)
    chain = make_chain(cfg, ledger)
    assert chain.prefilter_pairs([july, sept]) == []
    assert chain.cycle([july, sept]) == []


# --- focus-batch audit fixes: negation, implausible, leg unwind ---

def test_negation_wording_refuses_pair():
    """A negated predicate flips monotonicity; both wordings negated must not
    classify as a same-direction ladder."""
    a = dated(id="a", event_id="ev",
              question="Will GPT-6 NOT be released by July 31, 2027?",
              end_date=datetime(2027, 7, 31, tzinfo=timezone.utc),
              clob_token_ids=["ay", "an"])
    b = dated(id="b", event_id="ev",
              question="Will GPT-6 NOT be released by September 30, 2027?",
              end_date=datetime(2027, 9, 30, tzinfo=timezone.utc),
              clob_token_ids=["by", "bn"])
    assert classify_pair(a, b) is None


def test_implausible_net_flagged_and_not_executed(cfg, ledger):
    same = datetime.now(timezone.utc) + timedelta(days=60)
    subset_m = dated(id="s", event_id="ev1",
                     question="Will Bitcoin reach $200,000 by Dec 2026?",
                     end_date=same, outcome_prices=[0.30, 0.70],
                     clob_token_ids=["s-y", "s-n"], volume_24h_usd=50_000,
                     min_order_size=1.0, tick_size=0.001)
    superset_m = dated(id="p", event_id="ev1",
                       question="Will Bitcoin reach $150,000 by Dec 2026?",
                       end_date=same, outcome_prices=[0.10, 0.90],
                       clob_token_ids=["p-y", "p-n"], volume_24h_usd=50_000,
                       min_order_size=1.0, tick_size=0.001)
    # Both legs almost free -> cost << $1 -> NET hundreds of % -> implausible.
    clob = mock.Mock()
    clob.order_book.side_effect = lambda tok: {
        "p-y": book(ask=0.01, ask_size=100), "s-n": book(ask=0.01, ask_size=100),
    }[tok]
    cfg.chain_arb.execute = True
    cfg.chain_arb.max_stake_usd = 1000
    chain = make_chain(cfg, ledger, clob=clob)
    trader = mock.Mock()
    chain._trader = trader
    found = chain.cycle([subset_m, superset_m])
    assert len(found) == 1 and found[0].implausible is True
    trader.buy_limit.assert_not_called()             # never auto-executed


def test_leg_unwind_when_second_leg_killed(cfg, ledger):
    """First leg fills, second is FOK-killed -> the first is unwound (sold back),
    not left as a silent directional position."""
    from polymarket_bot.chainarb import ChainLeg, ChainPair
    m = dated(id="m", clob_token_ids=["m-y", "m-n"], min_order_size=1.0, tick_size=0.001)
    superset_leg = ChainLeg(market=m, outcome_index=0, token_id="p-y", ask=0.30, depth=100)
    subset_leg = ChainLeg(market=m, outcome_index=1, token_id="s-n", ask=0.30, depth=100)
    pair = ChainPair(event_id="ev", event_title="t", kind="date",
                     subset=subset_leg, superset=superset_leg, taker_coef=0.0)
    cfg.chain_arb.execute = True
    cfg.chain_arb.max_stake_usd = 1000
    cfg.chain_arb.spoof_screen = False
    clob = mock.Mock()
    clob.order_book.return_value = book(ask=0.31, ask_size=100, bid=0.29, bid_size=100)
    trader = mock.Mock()
    # superset buy fills (orderID); subset buy is FOK-killed (no orderID).
    trader.buy_limit.side_effect = [{"orderID": "sup"}, {}]
    chain = make_chain(cfg, ledger, clob=clob)
    chain._trader = trader
    spent = chain.execute(pair)
    assert spent == 0.0                              # aborted
    trader.sell_limit.assert_called_once()           # the filled leg was unwound
    assert trader.sell_limit.call_args.args[0] == "p-y"
