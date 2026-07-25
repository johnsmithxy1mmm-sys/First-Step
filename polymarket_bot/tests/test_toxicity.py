"""Predictive adverse-selection scoring from the recorded quote tape."""

import pytest

from polymarket_bot.toxicity import (ToxicityModel, continuation, mid_series,
                                     _returns)


def book_rows(mids, start_ts=1000.0, half=0.005):
    """tickstore rows (ts, bid, ask, bid_size, ask_size) around a mid path."""
    return [(start_ts + i, m - half, m + half, 100.0, 100.0)
            for i, m in enumerate(mids)]


# --- series extraction ---

def test_mid_series_skips_crossed_and_empty_books():
    rows = [(1.0, 0.44, 0.46, 10, 10),      # ok -> 0.45
            (2.0, 0.0, 0.46, 10, 10),       # no bid
            (3.0, 0.44, 0.0, 10, 10),       # no ask
            (4.0, 0.50, 0.48, 10, 10),      # crossed
            (5.0, 0.49, 0.51, 10, 10)]      # ok -> 0.50
    assert mid_series(rows) == [(1.0, pytest.approx(0.45)), (5.0, pytest.approx(0.50))]


def test_flat_ticks_are_dropped_before_correlating():
    """A prediction-market tape sits still for long stretches; counting those
    flat ticks would drag every autocorrelation to ~0 and hide real toxicity."""
    series = [(float(i), 0.50) for i in range(50)]
    assert _returns(series, min_move=0.001) == []


# --- continuation ---

def test_trending_tape_scores_positive():
    """A steady drift is the signature of informed flow: moves keep going."""
    mids = [0.40 + 0.002 * i for i in range(60)]
    moves = _returns(mid_series(book_rows(mids)), 0.001)
    assert continuation(moves) > 0.5


def test_mean_reverting_tape_scores_negative():
    """Alternating moves revert — a maker's fills come back in their favour."""
    mids = [0.50 + (0.004 if i % 2 else -0.004) for i in range(60)]
    moves = _returns(mid_series(book_rows(mids)), 0.001)
    assert continuation(moves) < -0.5


def test_continuation_is_undecidable_without_data():
    assert continuation([]) == 0.0
    assert continuation([0.01, 0.01]) == 0.0        # fewer than 3 moves
    assert continuation([0.0] * 10) == 0.0          # no path travelled at all


def test_constant_drift_is_maximally_toxic():
    """The case that breaks lag-1 autocorrelation: identical one-way moves have
    zero variance, so the textbook metric scores 0 on the most toxic tape there
    is. The efficiency ratio must call it +1."""
    assert continuation([0.01] * 30) == pytest.approx(1.0)
    assert continuation([-0.01] * 30) == pytest.approx(1.0)


def test_random_walk_sits_near_zero():
    import random
    rng = random.Random(7)
    moves = [rng.choice((-0.01, 0.01)) for _ in range(400)]
    assert abs(continuation(moves)) < 0.5


def test_continuation_is_bounded():
    mids = [0.30 + 0.003 * i for i in range(80)]
    rho = continuation(_returns(mid_series(book_rows(mids)), 0.001))
    assert -1.0 <= rho <= 1.0


# --- model ---

def test_toxic_market_widens_the_spread():
    trending = [0.40 + 0.002 * i for i in range(60)]
    model = ToxicityModel()
    model.fit({"toxic": book_rows(trending)})
    assert model.multiplier("toxic") > 1.2


def test_friendly_market_is_neutral_never_a_discount():
    """Being wrong about 'safe' costs money; being wrong about 'dangerous' costs
    a few missed fills. So a calm tape earns 1.0, never less."""
    reverting = [0.50 + (0.004 if i % 2 else -0.004) for i in range(60)]
    model = ToxicityModel()
    model.fit({"calm": book_rows(reverting)})
    assert model.multiplier("calm") == 1.0


def test_thin_tape_is_silent_rather_than_confident():
    """Too few moves must yield no opinion at all — not a fabricated one."""
    model = ToxicityModel(min_moves=20)
    model.fit({"thin": book_rows([0.40, 0.42, 0.44])})
    assert model.continuation_of("thin") is None
    assert model.multiplier("thin") == 1.0


def test_multiplier_respects_its_ceiling():
    trending = [0.20 + 0.004 * i for i in range(100)]
    model = ToxicityModel(max_multiplier=1.5)
    model.fit({"t": book_rows(trending)})
    assert 1.0 <= model.multiplier("t") <= 1.5


def test_unknown_token_is_neutral():
    assert ToxicityModel().multiplier("never-seen") == 1.0


# --- blending with measurement ---

def test_measured_markout_overrides_the_estimate():
    """Once a market has really spoken, evidence beats inference — even when the
    estimate is more alarming."""
    trending = [0.40 + 0.002 * i for i in range(60)]
    model = ToxicityModel()
    model.fit({"tok": book_rows(trending)})
    assert model.multiplier("tok") > 1.2
    assert model.blend("tok", realized_mult=1.0, realized_fills=50) == 1.0


def test_cold_start_takes_the_more_cautious_of_the_two():
    trending = [0.40 + 0.002 * i for i in range(60)]
    model = ToxicityModel()
    model.fit({"tok": book_rows(trending)})
    blended = model.blend("tok", realized_mult=1.0, realized_fills=0)
    assert blended == model.multiplier("tok") > 1.0
    # A harsher realized value still wins on the cautious side. (This tape is a
    # pure drift, so the estimate is already at the 2.0 ceiling — go above it.)
    assert model.blend("tok", realized_mult=2.5, realized_fills=1) == 2.5
