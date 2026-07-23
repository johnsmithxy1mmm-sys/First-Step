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
