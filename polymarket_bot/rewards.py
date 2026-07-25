"""Polymarket liquidity-rewards scoring — the real formula, not a turnover proxy.

Verified against docs.polymarket.com/market-makers/liquidity-rewards:

    S(v, s) = ((v - s) / v)^2 * b

for one resting order, where `v` is the market's max spread (rewards_max_spread),
`s` is the order's distance from the adjusted midpoint and `b` is its size.
Orders outside the band (s >= v) score zero.

Per side the scores are summed into Q_one (bids) and Q_two (asks), combined as

    Q_min = max( min(Q_one, Q_two), max(Q_one, Q_two) / c )      c = 3

so balanced two-sided liquidity scores up to 3x what the same size earns
one-sided. When the midpoint sits outside [0.10, 0.90] liquidity MUST be
two-sided to score at all. A maker's daily payout is their share of the sum of
Q_min over that market, paid at midnight UTC (minimum $1).

WHY THIS MATTERS MORE THAN IT LOOKS: the score is QUADRATIC in the distance
from the midpoint, so the band's outer edge is nearly worthless --

    s = 0.10v -> 81% of max      s = 0.50v -> 25%
    s = 0.25v -> 56%             s = 0.90v ->  1%

Quoting at 0.9 * max_spread (a natural-looking "stay inside the band" clamp)
earns one percent of what the same size earns at the midpoint. This module
exists so that trade-off is priced explicitly instead of being stumbled into.

This module is pure arithmetic on inputs the caller supplies. It deliberately
does NOT invent a pool size: absolute dollar rewards need the market's daily
allocation, which we do not fetch. Everything here is RELATIVE (share of Q),
which is what ranking markets and placing a quote actually require.
"""

from __future__ import annotations

from .models import OrderBook

# Single-sided divisor from the published Q_min formula.
ONE_SIDED_DIVISOR = 3.0
# Outside this midpoint band the program requires two-sided liquidity to score.
TWO_SIDED_ONLY_BELOW = 0.10
TWO_SIDED_ONLY_ABOVE = 0.90


def spread_score(size: float, spread: float, max_spread: float) -> float:
    """S(v, s) = ((v - s)/v)^2 * b for one resting order. 0 outside the band."""
    if size <= 0 or max_spread <= 0 or spread < 0 or spread >= max_spread:
        return 0.0
    return ((max_spread - spread) / max_spread) ** 2 * size


def q_min(q_one: float, q_two: float, mid: float) -> float:
    """Combine the two sides exactly as the rewards program does.

    Balanced two-sided liquidity earns min(Q_one, Q_two); a lopsided or
    single-sided book still earns max(...)/3. Outside [0.10, 0.90] one-sided
    liquidity scores nothing at all.
    """
    q_one, q_two = max(q_one, 0.0), max(q_two, 0.0)
    if mid < TWO_SIDED_ONLY_BELOW or mid > TWO_SIDED_ONLY_ABOVE:
        if q_one <= 0 or q_two <= 0:
            return 0.0
        return min(q_one, q_two)
    return max(min(q_one, q_two), max(q_one, q_two) / ONE_SIDED_DIVISOR)


def our_q(size: float, half_spread: float, max_spread: float, mid: float,
          two_sided: bool = True) -> float:
    """Q_min our own quote would earn: `size` on each side at `half_spread`."""
    side = spread_score(size, half_spread, max_spread)
    return q_min(side, side if two_sided else 0.0, mid)


def book_q(book: OrderBook | None, max_spread: float, mid: float,
           exclude_size: float = 0.0) -> float:
    """Q_min already resting in the book — our competition for the pool.

    `exclude_size` removes our own quote from each side when we are already
    quoting, so a market is not judged less attractive because we are in it.
    """
    if book is None or max_spread <= 0 or mid <= 0:
        return 0.0
    q_bid = sum(spread_score(max(level.size - exclude_size, 0.0),
                             abs(mid - level.price), max_spread)
                for level in book.bids)
    q_ask = sum(spread_score(max(level.size - exclude_size, 0.0),
                             abs(level.price - mid), max_spread)
                for level in book.asks)
    return q_min(q_bid, q_ask, mid)


def reward_share(size: float, half_spread: float, max_spread: float, mid: float,
                 book: OrderBook | None, two_sided: bool = True) -> float:
    """Our expected fraction of this market's daily reward pool, in [0, 1].

    Pool dollars are unknown (not fetched), so this is a share, not an amount.
    Comparing shares across markets is exactly what market selection needs, and
    a thin reward market where we are a large fraction of Q beats a crowded one
    where the same size disappears -- the only place a small account wins by
    formula rather than by out-trading informed flow.
    """
    mine = our_q(size, half_spread, max_spread, mid, two_sided)
    if mine <= 0:
        return 0.0
    return mine / (mine + book_q(book, max_spread, mid, exclude_size=size))


def score_fraction(half_spread: float, max_spread: float) -> float:
    """Fraction of the maximum per-share score kept at this distance from mid.

    ((v - s)/v)^2. Handy for reasoning about the cost of widening: this is the
    multiplier applied to reward income when the quote steps away from mid.
    """
    if max_spread <= 0 or half_spread >= max_spread:
        return 0.0
    return ((max_spread - max(half_spread, 0.0)) / max_spread) ** 2
