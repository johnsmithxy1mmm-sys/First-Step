"""Spoof guard: refuse to cross into a book that looks painted.

The way an arbitrageur is fleeced is a fake wall: a large resting order at the
touch that we price our "riskless" fill against, which vanishes the instant we
try to take it — leaving one leg filled at a bad price and the arb turned into
a directional loss. This is a pure, book-shape screen run right before an arb
execution (arbitrage / chain), independent of the edge math:

  * touch-vs-behind — a big top-of-book order with almost nothing behind it is
    the classic spoof shape (real liquidity has depth behind the touch);
  * one-sided cliff — displayed size at the touch dwarfs the opposite side, a
    lure that pushes you to lift it;
  * transient wall (optional) — a touch order much larger than its own recent
    average, i.e. it just appeared, is more likely to disappear.

Conservative by construction: a flagged book blocks EXECUTION only (detection
and alerting continue), so a false positive costs a missed fill, never money.
"""

from __future__ import annotations

from pydantic import BaseModel

from .models import OrderBook


class SpoofVerdict(BaseModel):
    suspicious: bool
    reasons: list[str] = []


def _depth_behind(levels: list, best_price: float, side: str,
                  band: float) -> float:
    """Size resting within `band` BEHIND the touch (worse prices than best)."""
    total = 0.0
    for lvl in levels:
        if lvl.size <= 0:
            continue
        if side == "ask" and best_price < lvl.price <= best_price + band:
            total += lvl.size
        elif side == "bid" and best_price - band <= lvl.price < best_price:
            total += lvl.size
    return total


def screen_ask(book: OrderBook, *, band: float = 0.05, min_wall_size: float = 500.0,
               behind_ratio: float = 0.25, cliff_ratio: float = 6.0,
               recent_avg_touch: float | None = None,
               transient_ratio: float = 5.0) -> SpoofVerdict:
    """Screen the ASK side we would BUY into (arb legs are taker buys).

    Only a LARGE touch (>= min_wall_size) can be a spoof — a small top-of-book
    order with nothing behind it is just thin liquidity, not manipulation, and
    must not block an otherwise-valid arb.
    """
    reasons: list[str] = []
    ask = book.best_ask
    if ask <= 0:
        return SpoofVerdict(suspicious=True, reasons=["no ask"])
    touch = next((l.size for l in book.asks if l.price == ask and l.size > 0), 0.0)
    if touch <= 0:
        return SpoofVerdict(suspicious=True, reasons=["empty touch"])
    if touch < min_wall_size:
        return SpoofVerdict(suspicious=False)   # small touch = thin, not painted

    behind = _depth_behind(book.asks, ask, "ask", band)
    if behind < behind_ratio * touch:
        reasons.append("thin behind the touch (wall shape)")

    bid_touch = next((l.size for l in book.bids if l.price == book.best_bid
                      and l.size > 0), 0.0)
    if bid_touch > 0 and touch > cliff_ratio * bid_touch:
        reasons.append("one-sided size cliff at the ask")

    if recent_avg_touch is not None and recent_avg_touch > 0 \
            and touch > transient_ratio * recent_avg_touch:
        reasons.append("touch much larger than its recent average (transient)")

    return SpoofVerdict(suspicious=bool(reasons), reasons=reasons)
