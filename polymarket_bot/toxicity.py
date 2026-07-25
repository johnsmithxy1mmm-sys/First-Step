"""Predictive adverse-selection scoring from the recorded quote tape.

THE GAP THIS FILLS: `MarkoutFeedback` widens the spread in markets where our
fills have already gone against us. That is measured and therefore trustworthy,
but it is also purely retrospective — a market we have never quoted carries
multiplier 1.0 no matter how obviously toxic its tape looks, so the only way to
learn a market is dangerous is to lose money in it first.

The tick store holds the top of book for every market we WATCH, traded or not.
That tape is enough to estimate the one thing a maker actually fears: after the
price moves, does it keep going? A resting quote is picked off precisely when
the move that filled it continues. So:

    toxicity = continuation of mid-moves over a short horizon

measured as the lag-1 autocorrelation of successive mid returns.

    rho > 0  trending  -> informed flow, fills are adverse      -> quote wider
    rho ~ 0  random walk -> fills are noise                     -> neutral
    rho < 0  mean-reverting -> fills revert in our favour       -> friendly

This is an ESTIMATE from prices, so it never overrides measurement: `blend()`
hands over to realized markout as soon as a market has enough real fills. Its
job is to cover the cold start, not to argue with evidence.

Deliberately conservative: it can only ever WIDEN the quote (multiplier >= 1.0).
A friendly-looking tape earns no tightening — being wrong about "safe" costs
money, being wrong about "dangerous" costs a few missed fills.
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger(__name__)


def mid_series(rows: list[tuple]) -> list[tuple[float, float]]:
    """(ts, mid) from tickstore book rows, skipping crossed/empty books."""
    out: list[tuple[float, float]] = []
    for row in rows:
        if len(row) < 3:
            continue
        ts, bid, ask = float(row[0]), float(row[1]), float(row[2])
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        out.append((ts, (bid + ask) / 2.0))
    return out


def _returns(series: list[tuple[float, float]], min_move: float) -> list[float]:
    """Successive mid changes, dropping flat ticks.

    Flat ticks dominate a prediction-market tape (the mid sits still for long
    stretches) and would crush the autocorrelation toward zero, making every
    market look benign. Only actual moves carry information about continuation.
    """
    moves: list[float] = []
    for (_, prev), (_, cur) in zip(series, series[1:]):
        delta = cur - prev
        if abs(delta) >= min_move:
            moves.append(delta)
    return moves


def continuation(moves: list[float]) -> float:
    """How much the mid keeps going, in [-1, 1]. 0.0 when undecidable.

    Efficiency ratio |net displacement| / total path length, re-centred on the
    random-walk baseline. Lag-1 autocorrelation is the textbook choice and the
    wrong one here: a steady one-directional drift — the most toxic tape there
    is — has ZERO variance in its move sizes, so the correlation is undefined
    and scores 0. The efficiency ratio calls it 1.0, which is the truth.

        pure drift        -> +1   every move extends the last
        random walk       ->  0   path wanders, net displacement ~ sqrt(n)
        strict alternation -> -1   the path cancels itself out

    A random walk of n moves covers |net| ~ sqrt(2n/pi) of its path length, so
    that is the zero point; distances above and below it are scaled separately
    to keep both ends at exactly +/-1.
    """
    n = len(moves)
    if n < 3:
        return 0.0
    path = sum(abs(m) for m in moves)
    if path <= 0:
        return 0.0
    ratio = abs(sum(moves)) / path
    baseline = min(math.sqrt(2.0 / (math.pi * n)), 0.99)
    if ratio >= baseline:
        score = (ratio - baseline) / (1.0 - baseline)
    else:
        score = (ratio - baseline) / baseline
    return max(-1.0, min(1.0, score))


class ToxicityModel:
    """Per-token adverse-selection estimate from the recorded tape."""

    def __init__(self, min_move: float = 0.001, min_moves: int = 20,
                 max_multiplier: float = 2.0):
        self._min_move = min_move
        self._min_moves = min_moves          # below this the estimate is noise
        self._max_mult = max_multiplier
        self._rho: dict[str, float] = {}
        self._n: dict[str, int] = {}

    def fit(self, book_series: dict[str, list[tuple]]) -> None:
        """Score every key that has enough recorded movement to judge.

        Keys are whatever the caller scores by and are never interpreted here.
        The MM keys by market id (to line up with markout feedback) while the
        tick store records by token, so the caller does that translation once
        when it builds this mapping — mixing the two key spaces would silently
        score nothing.
        """
        self._rho.clear()
        self._n.clear()
        for token, rows in book_series.items():
            moves = _returns(mid_series(rows), self._min_move)
            self._n[token] = len(moves)
            if len(moves) < self._min_moves:
                continue                      # not enough evidence — stay silent
            self._rho[token] = continuation(moves)
        if self._rho:
            log.info("toxicity: scored %d/%d tokens (median continuation %.3f)",
                     len(self._rho), len(book_series), self._median())

    def _median(self) -> float:
        vals = sorted(self._rho.values())
        return vals[len(vals) // 2] if vals else 0.0

    def continuation_of(self, token: str) -> float | None:
        """Estimated continuation, or None when there is not enough tape."""
        return self._rho.get(token)

    def multiplier(self, token: str) -> float:
        """Spread multiplier in [1.0, max_multiplier]. Never tightens.

        Only positive continuation (trending = informed) widens. A calm or
        mean-reverting tape returns exactly 1.0 rather than a discount: being
        wrong about "safe" costs money, being wrong about "dangerous" costs a
        few missed fills.
        """
        rho = self._rho.get(token)
        if rho is None or rho <= 0:
            return 1.0
        return 1.0 + rho * (self._max_mult - 1.0)

    def blend(self, token: str, realized_mult: float, realized_fills: int,
              min_fills: int = 20) -> float:
        """Combine the estimate with measured markout — measurement wins.

        Below `min_fills` the realized multiplier rests on too few observations
        to trust on its own, so we take the more cautious of the two. At or above
        it, the market has spoken: use the realized value and drop the estimate.
        """
        if realized_fills >= min_fills:
            return realized_mult
        return max(realized_mult, self.multiplier(token))
