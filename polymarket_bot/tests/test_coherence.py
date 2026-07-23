"""Coherence graph: implication chains, isotonic projection, violations."""

from datetime import datetime, timezone

import pytest

from polymarket_bot.coherence import (analyze_event, coherent_probs,
                                       event_chain, incoherences,
                                       isotonic_nondecreasing)

from .conftest import make_market


def threshold_market(mid, value, prob, event="ev"):
    return make_market(
        id=mid, event_id=event, event_title="BTC 2026",
        question=f"Will Bitcoin reach ${value:,} by Dec 2026?",
        end_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
        outcome_prices=[prob, 1 - prob], clob_token_ids=[f"{mid}y", f"{mid}n"])


# --- isotonic regression (pure) ---

def test_isotonic_already_sorted_unchanged():
    assert isotonic_nondecreasing([0.1, 0.2, 0.5]) == [0.1, 0.2, 0.5]


def test_isotonic_pools_violators():
    # A dip is pooled to the average, restoring non-decreasing order.
    out = isotonic_nondecreasing([0.1, 0.4, 0.2, 0.5])
    assert out == pytest.approx([0.1, 0.3, 0.3, 0.5])   # 0.4 & 0.2 pooled to 0.3
    assert all(out[i] <= out[i + 1] for i in range(len(out) - 1))


def test_isotonic_monotone_output_on_descending():
    out = isotonic_nondecreasing([0.9, 0.5, 0.1])
    assert out == [0.5, 0.5, 0.5]             # all pooled to the mean


# --- chain ordering (rarest -> commonest) ---

def test_event_chain_orders_by_implication():
    # $250k (rarest) implies $200k implies $150k (most likely).
    ms = [threshold_market("a", 150_000, 0.30),
          threshold_market("b", 250_000, 0.05),
          threshold_market("c", 200_000, 0.12)]
    chain = event_chain(ms)
    assert [m.id for m in chain] == ["b", "c", "a"]   # 250k, 200k, 150k


def test_event_chain_none_for_unrelated_markets():
    a = make_market(id="a", event_id="ev", question="Will team A win?",
                    clob_token_ids=["ay", "an"])
    b = make_market(id="b", event_id="ev", question="Will it rain Friday?",
                    clob_token_ids=["by", "bn"])
    assert event_chain([a, b]) is None


# --- projection + violation ---

def test_coherent_projection_leaves_coherent_prices_alone():
    ms = [threshold_market("a", 150_000, 0.30),
          threshold_market("b", 200_000, 0.12),
          threshold_market("c", 250_000, 0.05)]
    got = coherent_probs(ms)
    assert got["a"] == 0.30 and got["b"] == 0.12 and got["c"] == 0.05


def test_incoherence_detected_when_prices_violate_order():
    # $200k priced ABOVE $150k — impossible (a higher bar can't be more likely).
    ms = [threshold_market("a", 150_000, 0.10),   # commonest, but priced LOW
          threshold_market("b", 200_000, 0.25),   # rarer, but priced HIGH
          threshold_market("c", 250_000, 0.03)]
    incs = incoherences(ms, min_gap=0.02)
    assert len(incs) == 1
    inc = incs[0]
    assert inc.max_gap > 0.1                       # 0.25 vs 0.10 is a big gap
    # The projection pulls the violating pair toward a common coherent value.
    proj = {n.market_id: n.p_coherent for n in inc.nodes}
    assert proj["c"] <= proj["b"] <= proj["a"]     # non-decreasing rarest->commonest


def test_no_incoherence_below_threshold():
    ms = [threshold_market("a", 150_000, 0.121),
          threshold_market("b", 200_000, 0.12)]    # tiny 0.001 violation
    assert incoherences(ms, min_gap=0.02) == []


def test_analyze_event_projection_is_monotone():
    ms = [threshold_market("a", 150_000, 0.10),
          threshold_market("b", 200_000, 0.25),
          threshold_market("c", 250_000, 0.03)]
    inc = analyze_event(ms)
    pc = [n.p_coherent for n in inc.nodes]
    assert all(pc[i] <= pc[i + 1] for i in range(len(pc) - 1))


# --- mixed events: an unrelated market must not veto the chain beside it ---

def test_unrelated_market_does_not_kill_the_chain():
    """Real events mix ladders with other markets ('dip to $50k' beside the
    'reach $X' ladder). The chain must still be found in its component."""
    ms = [threshold_market("a", 150_000, 0.30),
          threshold_market("b", 200_000, 0.12),
          threshold_market("c", 250_000, 0.05),
          make_market(id="dip", event_id="ev", event_title="BTC 2026",
                      question="Will Bitcoin dip below $50,000 in 2026?",
                      outcome_prices=[0.2, 0.8], clob_token_ids=["dy", "dn"])]
    chain = event_chain(ms)
    assert chain is not None
    assert [m.id for m in chain] == ["c", "b", "a"]
    assert "dip" not in {m.id for m in chain}


def test_violation_found_despite_unrelated_sibling():
    ms = [threshold_market("a", 150_000, 0.10),    # violated ladder...
          threshold_market("b", 200_000, 0.25),
          make_market(id="dip", event_id="ev", event_title="BTC 2026",
                      question="Will Bitcoin dip below $50,000 in 2026?",
                      outcome_prices=[0.2, 0.8], clob_token_ids=["dy", "dn"])]
    incs = incoherences(ms, min_gap=0.02)          # ...next to an unrelated market
    assert len(incs) == 1 and incs[0].max_gap > 0.1


def test_two_independent_chains_in_one_event():
    """Two separate ladders in one event: both are analyzed, not just the largest."""
    from polymarket_bot.coherence import coherent_probs
    ms = [threshold_market("a", 150_000, 0.30),
          threshold_market("b", 200_000, 0.12),
          make_market(id="e1", event_id="ev", event_title="BTC 2026",
                      question="Will Ethereum reach $10,000 by Dec 2026?",
                      end_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
                      outcome_prices=[0.20, 0.80], clob_token_ids=["e1y", "e1n"]),
          make_market(id="e2", event_id="ev", event_title="BTC 2026",
                      question="Will Ethereum reach $15,000 by Dec 2026?",
                      end_date=datetime(2026, 12, 31, tzinfo=timezone.utc),
                      outcome_prices=[0.08, 0.92], clob_token_ids=["e2y", "e2n"])]
    probs = coherent_probs(ms)
    assert {"a", "b", "e1", "e2"} <= set(probs)    # both chains projected
