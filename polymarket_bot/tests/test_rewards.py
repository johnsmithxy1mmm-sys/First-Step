"""Liquidity-rewards scoring: the published formula, pinned against its source.

The fee bug survived 400 self-consistent tests because no external constant was
ever checked against its documentation. These tests exist to make the rewards
formula the opposite case.
"""

import pytest

from polymarket_bot.models import BookLevel, OrderBook
from polymarket_bot.rewards import (ONE_SIDED_DIVISOR, book_q, our_q, q_min,
                                    reward_share, score_fraction, spread_score)


# --- S(v, s) = ((v - s)/v)^2 * b ---

def test_spread_score_matches_the_published_quadratic():
    # At the midpoint the order keeps its full size as score.
    assert spread_score(size=100, spread=0.0, max_spread=0.03) == pytest.approx(100)
    # Half-way out keeps a quarter.
    assert spread_score(100, 0.015, 0.03) == pytest.approx(25)
    # Exactly at (and past) the band edge scores nothing.
    assert spread_score(100, 0.03, 0.03) == 0.0
    assert spread_score(100, 0.05, 0.03) == 0.0


def test_score_is_quadratic_not_linear():
    """The whole point: doubling the distance quarters the score."""
    near = spread_score(100, 0.005, 0.04)
    far = spread_score(100, 0.010, 0.04)
    assert near / far == pytest.approx(((0.035 / 0.04) ** 2) / ((0.030 / 0.04) ** 2))
    assert far < near / 1.3            # decisively worse, not marginally


def test_score_fraction_table():
    """Pins the cost of widening — the numbers the MM trades against."""
    v = 0.04
    assert score_fraction(0.00 * v, v) == pytest.approx(1.00)
    assert score_fraction(0.10 * v, v) == pytest.approx(0.81)
    assert score_fraction(0.25 * v, v) == pytest.approx(0.5625)
    assert score_fraction(0.50 * v, v) == pytest.approx(0.25)
    # The natural-looking "stay inside the band" clamp earns 1% of the maximum.
    assert score_fraction(0.90 * v, v) == pytest.approx(0.01)


def test_degenerate_inputs_score_zero():
    assert spread_score(0, 0.01, 0.03) == 0.0
    assert spread_score(100, 0.01, 0.0) == 0.0
    assert spread_score(-5, 0.01, 0.03) == 0.0
    assert score_fraction(0.05, 0.0) == 0.0


# --- Q_min = max(min(Q1, Q2), max(Q1, Q2) / 3) ---

def test_q_min_rewards_balanced_two_sided_liquidity():
    assert q_min(100, 100, mid=0.5) == pytest.approx(100)


def test_q_min_still_pays_one_sided_at_a_third():
    assert q_min(90, 0, mid=0.5) == pytest.approx(90 / ONE_SIDED_DIVISOR)
    # Balanced beats lopsided for the same total size (100+100 vs 190+10).
    assert q_min(100, 100, 0.5) > q_min(190, 10, 0.5)


def test_one_sided_scores_nothing_outside_the_middle_band():
    """Below 0.10 / above 0.90 the program requires two-sided liquidity."""
    for mid in (0.05, 0.95):
        assert q_min(500, 0, mid) == 0.0
        assert q_min(100, 100, mid) == pytest.approx(100)


def test_q_min_clamps_negative_inputs():
    assert q_min(-10, -10, 0.5) == 0.0


# --- competition and share ---

def _book(bid_px, ask_px, size=500.0):
    return OrderBook(bids=[BookLevel(price=bid_px, size=size)],
                     asks=[BookLevel(price=ask_px, size=size)])


def test_reward_share_falls_as_competition_grows():
    thin = _book(0.49, 0.51, size=50)
    thick = _book(0.49, 0.51, size=5000)
    args = dict(size=100.0, half_spread=0.01, max_spread=0.04, mid=0.50)
    assert reward_share(book=thin, **args) > reward_share(book=thick, **args)
    assert 0.0 <= reward_share(book=thick, **args) <= 1.0


def test_thin_reward_market_is_where_a_small_quote_wins():
    """A $10-scale quote can own a real share of a thin pool and ~none of a
    crowded one — the ranking signal the turnover proxy could not express."""
    share_thin = reward_share(100, 0.005, 0.04, 0.5, _book(0.495, 0.505, size=20))
    assert share_thin > 0.5


def test_book_q_excludes_our_own_resting_size():
    """Our own quote must not make the market look crowded to us."""
    book = _book(0.495, 0.505, size=100)
    with_us = book_q(book, max_spread=0.04, mid=0.5, exclude_size=0.0)
    without_us = book_q(book, max_spread=0.04, mid=0.5, exclude_size=100.0)
    assert without_us < with_us
    assert without_us == 0.0


def test_reward_share_is_zero_when_out_of_band():
    assert reward_share(100, 0.05, 0.04, 0.5, _book(0.49, 0.51)) == 0.0


def test_our_q_two_sided_beats_one_sided():
    two = our_q(100, 0.01, 0.04, 0.5, two_sided=True)
    one = our_q(100, 0.01, 0.04, 0.5, two_sided=False)
    assert two == pytest.approx(one * ONE_SIDED_DIVISOR)


def test_no_book_means_we_own_the_pool():
    assert reward_share(100, 0.01, 0.04, 0.5, None) == pytest.approx(1.0)
