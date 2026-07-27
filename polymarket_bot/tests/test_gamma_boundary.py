"""F-014: the Gamma REST feed is a trust boundary, and it was never fuzzed.

The earlier audit hardened the websocket book against NaN/Infinity (F-001) but
left the REST parser — the OTHER source of every price and tick size — accepting
whatever it was handed. All nine hand-built hostile payloads were accepted.

The sharpest consequence was in the order path: `orderPriceMinTickSize` flows
straight into `clob.round_to_tick`, where

  * a NaN tick RAISED ValueError on `round(price / tick)`, killing whichever
    strategy job was mid-execution, and
  * a NEGATIVE tick fell into the `tick <= 0` early return, which hands back the
    price UNCLAMPED — silently re-opening the 0/1 order-price hole that clamp
    exists to close.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from polymarket_bot.clob import round_to_tick
from polymarket_bot.models import Market


def _raw(**overrides) -> dict:
    base = {"id": "m1", "question": "q", "outcomes": '["Yes","No"]',
            "outcomePrices": '["0.5","0.5"]', "clobTokenIds": '["a","b"]'}
    base.update(overrides)
    return base


# --- prices: a probability is finite and in [0, 1] or it is not a price ---

@pytest.mark.parametrize("prices", ['["NaN","1"]', '["Infinity","0"]',
                                    '["-0.5","1.5"]', '["7","1"]'])
def test_unusable_prices_reject_the_market(prices):
    assert Market.from_gamma(_raw(outcomePrices=prices)) is None


def test_sane_market_still_parses():
    m = Market.from_gamma(_raw())
    assert m is not None and m.outcome_prices == [0.5, 0.5]


# --- sizes: zero/negative/non-finite means "field unusable", not "unusual" ---

@pytest.mark.parametrize("tick", [-0.01, 0.0, float("nan"), float("inf")])
def test_unusable_tick_falls_back_to_the_default(tick):
    m = Market.from_gamma(_raw(orderPriceMinTickSize=tick))
    assert m is not None
    assert m.tick_size == 0.001


@pytest.mark.parametrize("size", [-100, 0, float("nan")])
def test_unusable_min_order_size_falls_back(size):
    m = Market.from_gamma(_raw(orderMinSize=size))
    assert m is not None and m.min_order_size == 5.0


def test_non_finite_numerics_are_treated_as_absent():
    m = Market.from_gamma(_raw(volumeNum=float("inf"),
                               rewardsMaxSpread=float("inf"),
                               bestBid=float("nan")))
    assert m is not None
    for value in (m.volume_usd, m.rewards_max_spread, m.best_bid):
        assert math.isfinite(value)


# --- defence in depth: round_to_tick must survive what slips past ---

@pytest.mark.parametrize("tick", [-0.01, 0.0, float("nan"), float("inf")])
def test_round_to_tick_never_raises_on_a_bad_tick(tick):
    out = round_to_tick(0.44, tick)
    assert math.isfinite(out) and 0.0 <= out <= 1.0


def test_round_to_tick_never_returns_an_untradable_price_on_a_bad_tick():
    """A negative tick used to return the price unclamped — the exact hole the
    clamp was added to close."""
    assert 0.0 <= round_to_tick(1.5, -0.01) <= 1.0
    assert 0.0 <= round_to_tick(-3.0, -0.01) <= 1.0
    assert math.isfinite(round_to_tick(float("nan"), 0.001))


# --- property: no Gamma payload yields a market that breaks order pricing ---

@given(
    tick=st.one_of(st.floats(allow_nan=True, allow_infinity=True),
                   st.just(0.0), st.integers(-5, 5)),
    price=st.one_of(
        st.floats(min_value=-2, max_value=2, allow_nan=False,
                  allow_infinity=False),
        st.sampled_from([float("nan"), float("inf"), float("-inf")]),
    ),
)
@settings(max_examples=300, deadline=None)
def test_property_parsed_market_always_prices_orders_safely(tick, price):
    m = Market.from_gamma(_raw(orderPriceMinTickSize=tick))
    assert m is not None
    assert math.isfinite(m.tick_size) and m.tick_size > 0
    out = round_to_tick(price, m.tick_size)
    assert math.isfinite(out) and 0.0 <= out <= 1.0


def test_tick_above_one_is_rejected_not_merely_positive():
    """Found by the property test above, not by reading the code: a tick >= 1
    makes `1 - tick` negative, so the clamp `min(max(p, tick), 1 - tick)`
    inverts and returns a NEGATIVE order price."""
    m = Market.from_gamma(_raw(orderPriceMinTickSize=2.0))
    assert m is not None and m.tick_size == 0.001
    # And defence in depth, if one ever reaches the function directly:
    assert round_to_tick(0.0, 2.0) >= 0.0
    assert 0.0 <= round_to_tick(0.44, 1.0) <= 1.0
