"""MM 2.0 microstructure: realized vol, queue-ahead, fill probability."""

from polymarket_bot.microstructure import (RealizedVol, fill_probability,
                                           queue_ahead_usd)
from polymarket_bot.models import BookLevel


def test_realized_vol_starts_flat_then_moves():
    v = RealizedVol(alpha=0.5)
    assert v.update("m", 0.50) == 0.0        # no prior mid
    v.update("m", 0.55)                       # a jump
    assert v.sigma("m") > 0
    hi = v.sigma("m")
    for _ in range(20):
        v.update("m", 0.55)                   # flat -> vol decays
    assert v.sigma("m") < hi


def test_queue_ahead_buy_side():
    bids = [BookLevel(price=0.50, size=100), BookLevel(price=0.49, size=200),
            BookLevel(price=0.51, size=50)]
    # A BUY at 0.50: bids priced >= 0.50 are ahead of us.
    assert queue_ahead_usd(bids, 0.50, "BUY") == 0.50 * 100 + 0.51 * 50


def test_queue_ahead_sell_side():
    asks = [BookLevel(price=0.60, size=100), BookLevel(price=0.61, size=200)]
    assert queue_ahead_usd(asks, 0.60, "SELL") == 0.60 * 100


def test_fill_probability_monotonic():
    assert fill_probability(100, 10, 0) == 0.0        # no flow -> no fill
    p_low = fill_probability(1000, 10, 100)           # deep queue, little flow
    p_high = fill_probability(100, 10, 1000)          # shallow queue, lots of flow
    assert 0.0 < p_low < p_high < 1.0
